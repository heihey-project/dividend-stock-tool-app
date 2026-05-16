# main_free_ver1_0.py
# 高配当安定株 一括スクリーニング 無料版
# - stock_timing.py は分析エンジンとして import するだけ
# - 詳細画面は main 側では表示しない
# - 表の「開く」から stock_timing.py に ticker を渡して開く
# - 無料版は手入力のみ / 最大10銘柄 / 買い候補のみ表示
# - 有料版機能のUIには「有料版で開放」と表示し、機能は実行しない

from __future__ import annotations

import re
from typing import Any, Dict, List
from urllib.parse import quote

import pandas as pd
import streamlit as st

import stock_timing as engine


# ローカル運用：stock_timing.py を 8502 で起動しておく
# Web公開時に pages/stock_timing.py 構成へ移す場合は、環境に合わせて変更
DETAIL_BASE_URL = "https://dividend-stock-tool-app-3747amibf23otamsq7sda7.streamlit.app"

FREE_MAX_TICKERS = 10

DEFAULT_TICKERS = """9432
8593
8058
8316
2914
9101"""


# -----------------------------
# 基本ユーティリティ
# -----------------------------
def normalize_code(raw: str) -> str:
    return engine.normalize_ticker(str(raw).strip())


def parse_tickers(text: str) -> List[str]:
    parts = re.split(r"[\n,、\s]+", text.strip())
    tickers: List[str] = []
    seen = set()
    for p in parts:
        if not p:
            continue
        t = normalize_code(p)
        if t and t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers


def pct(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x) * 100:.2f}%"


def num(x) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{float(x):.1f}"


def detail_url(ticker: str) -> str:
    base = DETAIL_BASE_URL.rstrip("/")
    return f"{base}/?ticker={quote(ticker)}"


def render_detail_buttons(view: pd.DataFrame) -> None:
    """検索結果の下に、確実に詳細ページへ飛べるボタンを表示する。"""
    if view.empty:
        return

    st.markdown("### 詳細を開く")
    for _, row in view.iterrows():
        ticker = str(row.get("銘柄コード", ""))
        judgment = str(row.get("総合判定", ""))
        url = str(row.get("詳細", ""))

        if not url or url == "-":
            continue

        c1, c2, c3 = st.columns([1.2, 1.2, 5])
        c1.write(ticker)
        c2.write(judgment)
        c3.link_button("この銘柄を詳細表示", url, use_container_width=True)


# -----------------------------
# 分析処理
# -----------------------------
@st.cache_data(ttl=60 * 60, show_spinner=False)
def analyze_one(ticker: str) -> Dict[str, Any]:
    try:
        df = engine.download_price_data(ticker)
        if df.empty or len(df) < 700:
            return {
                "詳細": detail_url(ticker),
                "銘柄コード": ticker,
                "総合判定": "取得不可",
                "現在値": "-",
                "配当利回り": "-",
                "60日期待": "-",
                "180日期待": "-",
                "360日期待": "-",
                "180日勝率": "-",
                "理由": "株価データ不足",
                "エラー": "",
            }

        valuation = engine.fetch_valuation_data(ticker, df)
        factor_score = engine.calc_factor_score(valuation)
        result_df, _ = engine.calc_all_horizons(df)
        signal, _ = engine.judge_signal(df, result_df, factor_score, valuation)

        r60 = result_df[result_df["保有日数"] == 60].iloc[0]
        r180 = result_df[result_df["保有日数"] == 180].iloc[0]
        r360 = result_df[result_df["保有日数"] == 360].iloc[0]

        state = engine.calc_bargain_timing_state(df)
        state_label: List[str] = []
        if state.get("enough_discount"):
            state_label.append("高値から下落済み")
        if state.get("bottoming"):
            state_label.append("下げ止まり気味")
        if state.get("updating_low"):
            state_label.append("安値更新中")
        if state.get("ma25_recovered"):
            state_label.append("25日線回復")
        elif state.get("near_ma25"):
            state_label.append("25日線付近")

        reason = " / ".join(state_label) if state_label else "判定条件を確認"
        reason += f" / 180日期待{pct(r180['期待リターン'])} / 360日期待{pct(r360['期待リターン'])}"

        return {
            "詳細": detail_url(ticker),
            "銘柄コード": ticker,
            "総合判定": signal,
            "現在値": num(df["Close"].iloc[-1]),
            "配当利回り": pct(valuation.dividend_yield),
            "60日期待": pct(r60["期待リターン"]),
            "180日期待": pct(r180["期待リターン"]),
            "360日期待": pct(r360["期待リターン"]),
            "180日勝率": pct(r180["勝率"]),
            "理由": reason,
            "エラー": "",
        }
    except Exception as e:
        return {
            "詳細": detail_url(ticker),
            "銘柄コード": ticker,
            "総合判定": "エラー",
            "現在値": "-",
            "配当利回り": "-",
            "60日期待": "-",
            "180日期待": "-",
            "360日期待": "-",
            "180日勝率": "-",
            "理由": "分析中にエラー",
            "エラー": str(e),
        }


def run_screening(tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(tickers, start=1):
        status.write(f"検索中: {ticker}  ({i}/{len(tickers)})")
        rows.append(analyze_one(ticker))
        progress.progress(i / len(tickers))

    status.empty()
    progress.empty()

    df = pd.DataFrame(rows)
    if not df.empty:
        # 無料版は買い候補のみ表示するが、内部では検索件数の確認用に全結果を保持する
        order = {"買い候補": 0, "打診買い候補": 1, "待ち": 2, "危険": 3, "取得不可": 4, "エラー": 5}
        df["_sort"] = df["総合判定"].map(order).fillna(99)
        df = df.sort_values(["_sort", "銘柄コード"]).drop(columns=["_sort"])
    return df


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="高配当安定株 一括スクリーニング 無料版", layout="wide")
    st.title("高配当安定株 一括スクリーニング 無料版")

    ticker_text = st.text_area(
        "銘柄コードを入力（無料版は最大10銘柄）",
        value=DEFAULT_TICKERS,
        height=170,
    )
    parsed_tickers = parse_tickers(ticker_text)

    over_limit = len(parsed_tickers) > FREE_MAX_TICKERS
    tickers = parsed_tickers[:FREE_MAX_TICKERS]

    if over_limit:
        st.warning(f"無料版は最大{FREE_MAX_TICKERS}銘柄までです。先頭{FREE_MAX_TICKERS}銘柄のみ検索します。")

    c1, c2, c3 = st.columns([1, 1, 4])
    run_batch = c1.button("一括検索", type="primary", use_container_width=True)
    c2.button("全銘柄検索（有料版で開放）", use_container_width=True, disabled=True)
    clear_cache = c3.button("キャッシュ削除")

    if clear_cache:
        st.cache_data.clear()
        st.success("キャッシュを削除しました。")

    st.write(f"対象: {len(tickers)}銘柄")

    st.info(
        "無料版では、手入力した銘柄のうち『買い候補』のみ表示します。"
        "全銘柄検索、打診買い候補/待ち/危険の表示、CSV保存は有料版で開放されます。"
    )

    if run_batch:
        if not tickers:
            st.warning("銘柄コードを入力してください。")
            return
        st.session_state["screening_result_free"] = run_screening(tickers)

    result = st.session_state.get("screening_result_free")
    if isinstance(result, pd.DataFrame) and not result.empty:
        buy_view = result[result["総合判定"] == "買い候補"].copy()

        locked_c1, locked_c2 = st.columns([1, 1])
        locked_c1.button("表示判定の切替（有料版で開放）", disabled=True, use_container_width=True)
        locked_c2.button("結果CSVを保存（有料版で開放）", disabled=True, use_container_width=True)

        st.subheader("検索結果（無料版：買い候補のみ）")

        if buy_view.empty:
            st.warning("無料版で表示できる『買い候補』はありませんでした。打診買い候補・待ち・危険の確認は有料版で開放されます。")
        else:
            st.dataframe(
                buy_view,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "詳細": st.column_config.LinkColumn("詳細", display_text="開く"),
                    "理由": st.column_config.TextColumn("理由", width="large"),
                    "エラー": st.column_config.TextColumn("エラー", width="medium"),
                },
            )

            render_detail_buttons(buy_view)

        with st.expander("有料版で開放される機能"):
            st.write("・全銘柄検索")
            st.write("・買い候補 / 打診買い候補 / 待ち / 危険 の全表示")
            st.write("・CSV保存")
            st.write("・検索銘柄数の制限解除")


if __name__ == "__main__":
    main()
