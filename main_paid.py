# main_ver1_3.py
# 高配当安定株 一括スクリーニング専用メイン画面
# - stock_timing.py は分析エンジンとして import するだけ
# - 詳細画面は main 側では表示しない
# - 表の「開く」から stock_timing.py に ticker を渡して開く
# - 全銘柄検索は JPX の東証上場銘柄一覧 Excel を自動取得して実行

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote

import pandas as pd
import streamlit as st

import stock_timing as engine


# ローカル運用：stock_timing.py を 8502 で起動しておく
# Webで同一アプリ内ページにする場合は、例：DETAIL_BASE_URL = "/stock_timing"
DETAIL_BASE_URL = "http://localhost:8502"

# JPX公式「東証上場銘柄一覧」Excel
# 公式ページ: https://www.jpx.co.jp/markets/statistics-equities/misc/01.html
JPX_LISTED_ISSUES_XLS = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

DEFAULT_TICKERS = """9432
8593
8058
8316
2914
9101"""

LOCAL_TICKER_FILES = [
    "all_tickers.csv",
    "jpx_tickers.csv",
    "tickers.csv",
    "銘柄一覧.csv",
]


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


# -----------------------------
# 全銘柄リスト取得
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def load_jpx_all_tickers_from_web() -> pd.DataFrame:
    """
    JPX公式Excelから東証上場銘柄一覧を取得する。
    注意：.xls のため、環境によって xlrd が必要。
    """
    try:
        df = pd.read_excel(JPX_LISTED_ISSUES_XLS)
    except ImportError as e:
        raise RuntimeError(
            "JPXのExcel(.xls)を読むために xlrd が必要です。\n"
            "ターミナルで `pip install xlrd` を実行してください。"
        ) from e
    except Exception as e:
        raise RuntimeError(f"JPX銘柄一覧を取得できませんでした: {e}") from e

    # JPX Excelの代表的な列名
    code_col = "コード" if "コード" in df.columns else df.columns[1]
    name_col = "銘柄名" if "銘柄名" in df.columns else None
    market_col = "市場・商品区分" if "市場・商品区分" in df.columns else None
    industry_col = "33業種区分" if "33業種区分" in df.columns else None

    out = pd.DataFrame()
    out["コード"] = df[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
    out["銘柄コード"] = out["コード"].map(lambda x: f"{x}.T" if pd.notna(x) else None)
    out["銘柄名"] = df[name_col] if name_col else ""
    out["市場・商品区分"] = df[market_col] if market_col else ""
    out["33業種区分"] = df[industry_col] if industry_col else ""
    out = out.dropna(subset=["銘柄コード"]).drop_duplicates(subset=["銘柄コード"])
    return out.reset_index(drop=True)


def load_local_all_tickers() -> List[str]:
    base = Path.cwd()
    for name in LOCAL_TICKER_FILES:
        path = base / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            try:
                df = pd.read_csv(path, encoding="cp932")
            except Exception:
                continue

        candidates = ["コード", "銘柄コード", "ticker", "Ticker", "code", "Code"]
        col = next((c for c in candidates if c in df.columns), df.columns[0])
        return [normalize_code(x) for x in df[col].dropna().tolist() if normalize_code(x)]
    return []


def build_all_tickers(include_etf_reit: bool = True) -> List[str]:
    """
    優先順位：
    1. JPX公式ExcelをWeb取得
    2. 失敗したらローカルCSV
    """
    try:
        jpx_df = load_jpx_all_tickers_from_web()
        if not include_etf_reit and "市場・商品区分" in jpx_df.columns:
            # 高配当安定株ツール用途なら、ETF/ETN/REIT等は除外した方がエラーが少ない
            mask = jpx_df["市場・商品区分"].astype(str).str.contains("内国株式|外国株式|TOKYO PRO Market", regex=True, na=False)
            jpx_df = jpx_df[mask]
        return jpx_df["銘柄コード"].dropna().drop_duplicates().tolist()
    except Exception as e:
        st.warning(str(e))
        local = load_local_all_tickers()
        if local:
            st.info("ローカルCSVの銘柄一覧を使います。")
        return local


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
                "180日平均最大下落": "-",
                "チェック通過": "-",
                "品質": "-",
                "理由": "株価データ不足",
                "エラー": "",
            }

        valuation = engine.fetch_valuation_data(ticker, df)
        factor_score = engine.calc_factor_score(valuation)
        result_df, _ = engine.calc_all_horizons(df)
        signal, _ = engine.judge_signal(df, result_df, factor_score, valuation)

        checks = engine.build_auto_checklist_items(
            ticker=ticker,
            df=df,
            valuation=valuation,
            result_df=result_df,
            factor_score=factor_score,
            signal=signal,
            box_label="",
        )
        ok_count = sum(1 for _, status, _ in checks if status == "ok")
        total_count = len(checks)

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
        reason += f" / 180日期待{pct(r180['期待リターン'])} / 360日期待{pct(r360['期待リターン'])} / 品質{factor_score.total}/100"

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
            "180日平均最大下落": pct(r180["平均最大下落率"]),
            "チェック通過": f"{ok_count}/{total_count}",
            "品質": factor_score.total,
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
            "180日平均最大下落": "-",
            "チェック通過": "-",
            "品質": "-",
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
    order = {"買い候補": 0, "打診買い候補": 1, "待ち": 2, "危険": 3, "取得不可": 4, "エラー": 5}
    if not df.empty:
        df["_sort"] = df["総合判定"].map(order).fillna(99)
        df = df.sort_values(["_sort", "銘柄コード"]).drop(columns=["_sort"])
    return df


# -----------------------------
# UI
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="高配当安定株 一括スクリーニング", layout="wide")
    st.title("高配当安定株 一括スクリーニング")

    ticker_text = st.text_area("銘柄コードを入力", value=DEFAULT_TICKERS, height=170)
    tickers = parse_tickers(ticker_text)

    c1, c2, c3 = st.columns([1, 1, 4])
    run_batch = c1.button("一括検索", type="primary", use_container_width=True)
    run_all = c2.button("全銘柄検索", use_container_width=True)
    clear_cache = c3.button("キャッシュ削除")

    if clear_cache:
        st.cache_data.clear()
        st.success("キャッシュを削除しました。")

    if run_all:
        with st.spinner("JPX公式の東証上場銘柄一覧を取得中..."):
            # True: JPX掲載の全コードを対象。ETF/REIT等も含む。
            tickers = build_all_tickers(include_etf_reit=True)
        run_batch = True

    st.write(f"対象: {len(tickers)}銘柄")

    if run_all and len(tickers) > 1000:
        st.warning(
            "全銘柄検索は数千銘柄を1件ずつyfinanceで取得するため、かなり時間がかかります。"
            "途中で通信制限や取得エラーが出る場合があります。"
        )

    if run_batch:
        if not tickers:
            st.warning("銘柄コードを入力してください。")
            return
        st.session_state["screening_result"] = run_screening(tickers)

    result = st.session_state.get("screening_result")
    if isinstance(result, pd.DataFrame) and not result.empty:
        judgments = st.multiselect(
            "表示する判定",
            ["買い候補", "打診買い候補", "待ち", "危険", "取得不可", "エラー"],
            default=["買い候補", "打診買い候補", "待ち", "危険"],
        )
        view = result[result["総合判定"].isin(judgments)].copy() if judgments else result.copy()

        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "詳細": st.column_config.LinkColumn("詳細", display_text="開く"),
                "理由": st.column_config.TextColumn("理由", width="large"),
                "エラー": st.column_config.TextColumn("エラー", width="medium"),
            },
        )

        st.download_button(
            "結果CSVを保存",
            data=view.to_csv(index=False).encode("utf-8-sig"),
            file_name="screening_result.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
