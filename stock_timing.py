# stock_timing_app_ver_4_2_no_target_filter.py
# 日本株 長期保有判定ツール
#
# ベース:
# - stock_timing_app_ver_1_1.py のUI方針
#
# 方針:
# - 期待値ラインはチャート由来のみ
# - PER/PBR/配当/ROEなどは期待値に足さない
# - PERなどは「銘柄品質スコア」として別枠表示
# - 最終判定は
#   1. チャート期待値
#   2. 現在の箱
#   3. 銘柄品質スコア
#   を組み合わせる
#
# インストール:
#   pip install streamlit yfinance pandas numpy plotly
#
# 起動:
#   streamlit run stock_timing_app_ver_4_2_no_target_filter.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf


HOLDING_DAYS_LIST = [60, 180, 360]

TREND_SMOOTH_DAYS = 5
MIN_BOX_DAYS = 5
BOX_WINDOW_DAYS = 20

SIMILAR_TOP_N = 200
EXPECTED_QUANTILE = 0.60


@dataclass
class ValuationData:
    current_per: Optional[float]
    forward_per: Optional[float]
    pbr: Optional[float]
    dividend_yield: Optional[float]
    trailing_eps: Optional[float]
    forward_eps: Optional[float]
    roe: Optional[float]
    profit_margin: Optional[float]
    debt_to_equity: Optional[float]
    payout_ratio: Optional[float]
    hist_per_median: Optional[float]
    hist_per_avg: Optional[float]
    per_gap: Optional[float]
    fair_price_by_median_per: Optional[float]
    per_series: pd.Series


@dataclass
class FactorScore:
    total: int
    value: int
    dividend: int
    quality: int
    stability: int
    label: str


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .judge-card {
            padding: 18px 20px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-bottom: 12px;
        }
        .judge-title {
            font-size: 15px;
            color: #6b7280;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .judge-value {
            font-size: 30px;
            font-weight: 900;
            line-height: 1.15;
        }
        .judge-sub {
            color: #6b7280;
            font-size: 13px;
            margin-top: 6px;
        }
        .signal-buy { color: #0f766e; }
        .signal-wait { color: #b45309; }
        .signal-danger { color: #dc2626; }
        .signal-exclude { color: #7f1d1d; }
        .side-reason {
            font-size: 12px;
            color: #b45309;
            line-height: 1.35;
            margin-top: -10px;
            margin-bottom: 8px;
            padding-left: 4px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: 390px 1fr;
            gap: 16px;
            align-items: stretch;
            margin-bottom: 18px;
        }
        .main-card {
            padding: 18px 20px;
            border-radius: 16px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            min-height: 128px;
        }
        .main-card-split {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
            height: 100%;
        }
        .main-card-left {
            padding-right: 16px;
            border-right: 1px solid #e5e7eb;
        }
        .main-card-right {
            padding-left: 16px;
        }
        .card-label {
            font-size: 13px;
            color: #6b7280;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .card-value {
            font-size: 28px;
            font-weight: 900;
            line-height: 1.1;
        }
        .card-sub {
            color: #6b7280;
            font-size: 12px;
            margin-top: 8px;
            line-height: 1.35;
        }
        .expect-card {
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            background: #ffffff;
            box-shadow: 0 4px 14px rgba(0,0,0,0.05);
            overflow: hidden;
            min-height: 128px;
        }
        .expect-header {
            padding: 10px 16px;
            font-size: 13px;
            color: #6b7280;
            font-weight: 800;
            border-bottom: 1px solid #e5e7eb;
            background: #f9fafb;
        }
        .expect-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
        }
        .expect-box {
            padding: 14px 18px;
            border-right: 1px solid #e5e7eb;
        }
        .expect-box:last-child {
            border-right: none;
        }
        .expect-title {
            color: #6b7280;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .expect-pct {
            font-size: 26px;
            font-weight: 900;
            line-height: 1.1;
        }
        .expect-line {
            color: #6b7280;
            font-size: 12px;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def safe_float(value) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if np.isfinite(value):
            return value
        return None
    except Exception:
        return None



def normalize_dividend_yield(value) -> Optional[float]:
    """
    yfinanceは銘柄によって dividendYield が
    0.0356 形式、または 3.56 形式で返ることがある。
    内部では 0.0356 形式に統一する。
    """
    y = safe_float(value)
    if y is None:
        return None

    # 1を超えていれば％表記とみなして100で割る
    if y > 1:
        y = y / 100

    # 異常値ガード
    if y < 0 or y > 0.30:
        return None

    return y


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not ticker:
        return ""
    if ticker.endswith(".T"):
        return ticker
    if ticker.isdigit():
        return f"{ticker}.T"
    return ticker


@st.cache_data(ttl=60 * 60)
def download_price_data(ticker: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period="10y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    df.index = pd.to_datetime(df.index)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA25"] = df["Close"].rolling(25).mean()
    df["MA75"] = df["Close"].rolling(75).mean()

    df["RET20"] = df["Close"].pct_change(20)
    df["RET60"] = df["Close"].pct_change(60)
    df["HIGH60"] = df["Close"].rolling(60).max()
    df["DD_FROM_HIGH60"] = df["Close"] / df["HIGH60"] - 1

    daily_ret = df["Close"].pct_change()
    df["VOL20"] = daily_ret.rolling(20).std()

    df["MA25_GAP"] = df["Close"] / df["MA25"] - 1
    df["MA75_GAP"] = df["Close"] / df["MA75"] - 1
    df["MA5_25_GAP"] = df["MA5"] / df["MA25"] - 1
    df["MA25_75_GAP"] = df["MA25"] / df["MA75"] - 1

    df = add_trend_boxes(df)
    return df


def add_trend_boxes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    strong_up = (df["MA5"] > df["MA25"]) & (df["MA25"] > df["MA75"])
    weak_up = (df["MA5"] > df["MA25"]) & (df["Close"] > df["MA25"]) & ~strong_up

    strong_down = (df["MA5"] < df["MA25"]) & (df["MA25"] < df["MA75"])
    weak_down = (df["MA5"] < df["MA25"]) & (df["Close"] < df["MA25"]) & ~strong_down

    df["TREND_BOX"] = "neutral"
    df.loc[weak_up, "TREND_BOX"] = "weak_up"
    df.loc[weak_down, "TREND_BOX"] = "weak_down"
    df.loc[strong_up, "TREND_BOX"] = "strong_up"
    df.loc[strong_down, "TREND_BOX"] = "strong_down"

    df["TREND_BOX"] = smooth_trend_box(df["TREND_BOX"])
    return df


def smooth_trend_box(series: pd.Series) -> pd.Series:
    values = series.tolist()
    smoothed = values.copy()

    for i in range(TREND_SMOOTH_DAYS, len(values)):
        window = values[i - TREND_SMOOTH_DAYS + 1:i + 1]
        counts = pd.Series(window).value_counts()
        top = counts.index[0]
        if counts.iloc[0] >= 3:
            smoothed[i] = top

    return pd.Series(smoothed, index=series.index)


def get_trend_segments(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    segments = []
    trend = df["TREND_BOX"].fillna("neutral").tolist()

    if not trend:
        return segments

    start_idx = 0
    current = trend[0]

    for i in range(1, len(trend)):
        if trend[i] != current:
            if current != "neutral":
                length = i - start_idx
                if length >= MIN_BOX_DAYS:
                    segments.append((start_idx, i - 1, current))
            start_idx = i
            current = trend[i]

    if current != "neutral":
        length = len(trend) - start_idx
        if length >= MIN_BOX_DAYS:
            segments.append((start_idx, len(trend) - 1, current))

    return segments


def split_segment_to_step_boxes(start_idx: int, end_idx: int) -> List[Tuple[int, int]]:
    boxes = []
    i = start_idx
    while i <= end_idx:
        j = min(i + BOX_WINDOW_DAYS - 1, end_idx)
        if j - i + 1 >= MIN_BOX_DAYS:
            boxes.append((i, j))
        i = j + 1
    return boxes


@st.cache_data(ttl=60 * 60)
def fetch_valuation_data(ticker: str, price_df: pd.DataFrame) -> ValuationData:
    tk = yf.Ticker(ticker)

    try:
        info = tk.info or {}
    except Exception:
        info = {}

    current_per = safe_float(info.get("trailingPE"))
    forward_per = safe_float(info.get("forwardPE"))
    pbr = safe_float(info.get("priceToBook"))
    dividend_yield = normalize_dividend_yield(info.get("dividendYield"))
    trailing_eps = safe_float(info.get("trailingEps"))
    forward_eps = safe_float(info.get("forwardEps"))
    roe = safe_float(info.get("returnOnEquity"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    payout_ratio = safe_float(info.get("payoutRatio"))

    eps_series = get_annual_eps_series(tk)
    per_series = build_daily_per_series(price_df, eps_series)

    hist_per_median = None
    hist_per_avg = None
    per_gap = None
    fair_price = None

    if not per_series.empty:
        valid = per_series.replace([np.inf, -np.inf], np.nan).dropna()
        valid = valid[(valid > 0) & (valid < 200)]

        if not valid.empty:
            hist_per_median = float(valid.median())
            hist_per_avg = float(valid.mean())

    eps_for_fair = forward_eps or trailing_eps
    if eps_for_fair is not None and hist_per_median is not None and eps_for_fair > 0:
        fair_price = eps_for_fair * hist_per_median

    if current_per is not None and hist_per_median is not None and current_per > 0:
        per_gap = (hist_per_median / current_per) - 1

    return ValuationData(
        current_per=current_per,
        forward_per=forward_per,
        pbr=pbr,
        dividend_yield=dividend_yield,
        trailing_eps=trailing_eps,
        forward_eps=forward_eps,
        roe=roe,
        profit_margin=profit_margin,
        debt_to_equity=debt_to_equity,
        payout_ratio=payout_ratio,
        hist_per_median=hist_per_median,
        hist_per_avg=hist_per_avg,
        per_gap=per_gap,
        fair_price_by_median_per=fair_price,
        per_series=per_series,
    )


def get_annual_eps_series(tk: yf.Ticker) -> pd.Series:
    income = pd.DataFrame()

    try:
        income = tk.income_stmt
    except Exception:
        income = pd.DataFrame()

    if income is None or income.empty:
        try:
            income = tk.get_income_stmt(freq="yearly")
        except Exception:
            income = pd.DataFrame()

    if income is None or income.empty:
        return pd.Series(dtype=float)

    income = income.copy()

    eps = None

    possible_eps_rows = [
        "Diluted EPS",
        "Basic EPS",
        "DilutedEPS",
        "BasicEPS",
    ]

    for row in possible_eps_rows:
        if row in income.index:
            eps = income.loc[row]
            break

    if eps is None:
        net_income_row = None
        share_row = None

        for row in ["Net Income", "NetIncome", "Net Income Common Stockholders"]:
            if row in income.index:
                net_income_row = row
                break

        for row in ["Diluted Average Shares", "DilutedAverageShares", "Basic Average Shares"]:
            if row in income.index:
                share_row = row
                break

        if net_income_row is not None and share_row is not None:
            shares = income.loc[share_row].replace(0, np.nan)
            eps = income.loc[net_income_row] / shares

    if eps is None:
        return pd.Series(dtype=float)

    eps = eps.dropna()
    eps.index = pd.to_datetime(eps.index)
    eps = eps.sort_index()
    eps = eps.astype(float)
    eps = eps[eps > 0]

    return eps


def build_daily_per_series(price_df: pd.DataFrame, annual_eps: pd.Series) -> pd.Series:
    if annual_eps.empty:
        return pd.Series(dtype=float)

    per = pd.Series(index=price_df.index, dtype=float)

    for fiscal_date, eps in annual_eps.items():
        available_from = fiscal_date + pd.Timedelta(days=90)
        per.loc[per.index >= available_from] = price_df.loc[per.index >= available_from, "Close"] / eps

    per = per.ffill()
    per = per.replace([np.inf, -np.inf], np.nan)
    per = per[(per > 0) & (per < 300)]

    return per


def calc_factor_score(valuation: ValuationData) -> FactorScore:
    value = 0
    dividend = 0
    quality = 0
    stability = 0

    if valuation.per_gap is not None:
        if valuation.per_gap >= 0.20:
            value += 25
        elif valuation.per_gap >= 0.00:
            value += 18
        elif valuation.per_gap >= -0.20:
            value += 10
        else:
            value += 3
    elif valuation.current_per is not None:
        if 0 < valuation.current_per <= 15:
            value += 18
        elif valuation.current_per <= 25:
            value += 10
        else:
            value += 3

    if valuation.pbr is not None:
        if valuation.pbr <= 1.0:
            value += 10
        elif valuation.pbr <= 2.0:
            value += 6
        elif valuation.pbr <= 3.5:
            value += 3

    value = min(value, 35)

    if valuation.dividend_yield is not None:
        if valuation.dividend_yield >= 0.04:
            dividend += 20
        elif valuation.dividend_yield >= 0.03:
            dividend += 16
        elif valuation.dividend_yield >= 0.02:
            dividend += 10
        elif valuation.dividend_yield > 0:
            dividend += 5

    if valuation.payout_ratio is not None:
        if 0 < valuation.payout_ratio <= 0.60:
            dividend += 8
        elif valuation.payout_ratio <= 0.80:
            dividend += 4
        elif valuation.payout_ratio > 1.00:
            dividend -= 6

    dividend = max(0, min(dividend, 25))

    if valuation.roe is not None:
        if valuation.roe >= 0.12:
            quality += 18
        elif valuation.roe >= 0.08:
            quality += 12
        elif valuation.roe >= 0.05:
            quality += 6

    if valuation.profit_margin is not None:
        if valuation.profit_margin >= 0.12:
            quality += 10
        elif valuation.profit_margin >= 0.06:
            quality += 6
        elif valuation.profit_margin >= 0.03:
            quality += 3

    quality = min(quality, 25)

    if valuation.debt_to_equity is not None:
        if valuation.debt_to_equity <= 50:
            stability += 12
        elif valuation.debt_to_equity <= 100:
            stability += 8
        elif valuation.debt_to_equity <= 200:
            stability += 4
    else:
        stability += 4

    if valuation.trailing_eps is not None and valuation.trailing_eps > 0:
        stability += 3

    stability = min(stability, 15)

    total = int(max(0, min(100, value + dividend + quality + stability)))

    if total >= 75:
        label = "良好"
    elif total >= 55:
        label = "普通"
    elif total >= 35:
        label = "注意"
    else:
        label = "弱い"

    return FactorScore(
        total=total,
        value=int(value),
        dividend=int(dividend),
        quality=int(quality),
        stability=int(stability),
        label=label,
    )


def trend_score(a: str, b: str) -> float:
    if a == b:
        return 0.0

    up = {"strong_up", "weak_up"}
    down = {"strong_down", "weak_down"}

    if a in up and b in up:
        return 0.25
    if a in down and b in down:
        return 0.25
    if a == "neutral" or b == "neutral":
        return 0.60
    return 1.00


def find_similar_indices(df: pd.DataFrame, max_horizon: int) -> Tuple[List[int], str]:
    latest = df.iloc[-1]

    feature_cols = [
        "MA25_GAP",
        "MA75_GAP",
        "MA5_25_GAP",
        "MA25_75_GAP",
        "RET20",
        "RET60",
        "DD_FROM_HIGH60",
        "VOL20",
    ]

    valid_end = len(df) - max_horizon - 1
    candidates = df.iloc[:valid_end].dropna(subset=feature_cols).copy()

    if candidates.empty:
        return list(range(75, valid_end)), "全期間参考"

    score = np.zeros(len(candidates), dtype=float)

    weights = {
        "MA25_GAP": 1.2,
        "MA75_GAP": 1.1,
        "MA5_25_GAP": 1.0,
        "MA25_75_GAP": 1.0,
        "RET20": 0.9,
        "RET60": 1.0,
        "DD_FROM_HIGH60": 1.3,
        "VOL20": 0.8,
    }

    for col in feature_cols:
        scale = float(candidates[col].std())
        if scale == 0 or np.isnan(scale):
            scale = 1.0
        diff = (candidates[col] - float(latest[col])).abs() / scale
        score += diff.to_numpy() * weights[col]

    score += candidates["TREND_BOX"].map(
        lambda x: trend_score(str(x), str(latest["TREND_BOX"]))
    ).to_numpy() * 2.0

    candidates["SIM_SCORE"] = score
    selected = candidates.sort_values("SIM_SCORE").head(SIMILAR_TOP_N)

    return [df.index.get_loc(i) for i in selected.index], f"類似度上位{len(selected)}件"


def calc_all_horizons(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    current_price = float(df["Close"].iloc[-1])
    similar_indices, match_level = find_similar_indices(df, max(HOLDING_DAYS_LIST))

    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    rows = []

    for days in HOLDING_DAYS_LIST:
        returns = []
        drawdowns = []

        for i in similar_indices:
            if i + days >= len(df):
                continue

            entry = close[i]
            exit_price = close[i + days]
            future_low = low[i + 1:i + days + 1]

            returns.append(exit_price / entry - 1)
            drawdowns.append(float(np.min(future_low)) / entry - 1)

        if not returns:
            chart_avg = 0.0
            chart_median = 0.0
            chart_q60 = 0.0
            win_rate = 0.0
            avg_dd = 0.0
            worst_dd = 0.0
            samples = 0
        else:
            ret_np = np.array(returns)
            dd_np = np.array(drawdowns)

            chart_avg = float(ret_np.mean())
            chart_median = float(np.median(ret_np))
            chart_q60 = float(np.quantile(ret_np, EXPECTED_QUANTILE))
            win_rate = float((ret_np > 0).mean())
            avg_dd = float(dd_np.mean())
            worst_dd = float(dd_np.min())
            samples = len(returns)

        rows.append(
            {
                "保有日数": days,
                "類似局面数": samples,
                "期待リターン": chart_q60,
                "平均リターン": chart_avg,
                "中央値リターン": chart_median,
                "勝率": win_rate,
                "平均最大下落率": avg_dd,
                "最悪最大下落率": worst_dd,
                "期待値ライン": current_price * (1 + chart_q60),
                "平均下落ライン": current_price * (1 + avg_dd),
                "最悪下落ライン": current_price * (1 + worst_dd),
            }
        )

    return pd.DataFrame(rows), match_level


def calc_bargain_timing_state(df: pd.DataFrame) -> dict:
    """
    高配当安定株を「下がったところで拾う」ための状態判定。
    下がっていない優良株は危険ではなく「待ち」にする。
    """
    latest = df.iloc[-1]
    close = float(latest["Close"])

    recent_high = float(df["Close"].tail(252).max())
    drawdown_from_high = (close / recent_high) - 1 if recent_high > 0 else 0.0

    recent_low_20 = float(df["Close"].tail(20).min())
    recent_low_60 = float(df["Close"].tail(60).min())

    ma5_now = float(df["MA5"].iloc[-1])
    ma5_5d_ago = float(df["MA5"].iloc[-6]) if len(df) >= 6 and not pd.isna(df["MA5"].iloc[-6]) else ma5_now
    ma25_now = float(df["MA25"].iloc[-1])
    ma75_now = float(df["MA75"].iloc[-1])

    ma5_up = ma5_now >= ma5_5d_ago
    ma5_down = ma5_now < ma5_5d_ago
    ma25_recovered = close >= ma25_now
    near_ma25 = close >= ma25_now * 0.97

    # 直近安値を明確に更新しているか
    updating_low = close <= recent_low_20 * 1.01 and ma5_down

    # 下げ止まり気味：安値更新が止まり、短期線が横ばい〜上向き
    bottoming = (not updating_low) and ma5_up and close > recent_low_20 * 1.03

    # かなり下がっているか
    enough_discount = drawdown_from_high <= -0.08
    deep_discount = drawdown_from_high <= -0.15

    return {
        "close": close,
        "drawdown_from_high": drawdown_from_high,
        "updating_low": updating_low,
        "bottoming": bottoming,
        "enough_discount": enough_discount,
        "deep_discount": deep_discount,
        "ma25_recovered": ma25_recovered,
        "near_ma25": near_ma25,
        "ma5_up": ma5_up,
        "ma5_down": ma5_down,
        "ma25": ma25_now,
        "ma75": ma75_now,
        "recent_low_20": recent_low_20,
        "recent_low_60": recent_low_60,
    }


def judge_signal(df: pd.DataFrame, result_df: pd.DataFrame, factor_score: FactorScore, valuation: ValuationData) -> Tuple[str, str]:
    """
    高配当安定株を長期保有するための買いタイミング判定。

    思想：
    ・上昇トレンド追随ではなく、下がった優良株の拾い場を探す
    ・下がっていない優良株は「危険」ではなく「待ち」
    ・下落中でも下げ止まり気味なら「打診買い候補」
    """
    r180 = result_df[result_df["保有日数"] == 180].iloc[0]
    r360 = result_df[result_df["保有日数"] == 360].iloc[0]

    exp180 = float(r180["期待リターン"])
    exp360 = float(r360["期待リターン"])
    win180 = float(r180["勝率"])
    avg_dd180 = abs(float(r180["平均最大下落率"]))

    state = calc_bargain_timing_state(df)

    long_expect_positive = exp180 > 0 and exp360 > 0
    weak_expect_positive = exp180 > 0 or exp360 > 0

    # 長期期待値が両方マイナスなら避ける
    if exp180 < 0 and exp360 < 0:
        return "危険", "signal-danger"

    # まだ落ちるナイフっぽい
    if state["updating_low"] and state["ma5_down"] and not long_expect_positive:
        return "危険", "signal-danger"

    # 一番狙いたい：十分下がって、下げ止まり、25日線も回復
    if (
        state["enough_discount"]
        and state["bottoming"]
        and state["ma25_recovered"]
        and long_expect_positive
        and win180 >= 0.50
        and avg_dd180 <= 0.18
    ):
        return "買い候補", "signal-buy"

    # 打診買い：十分下がって、安値更新が止まりつつある
    if (
        state["enough_discount"]
        and state["bottoming"]
        and weak_expect_positive
        and avg_dd180 <= 0.22
    ):
        return "打診買い候補", "signal-buy"

    # かなり下がっていて期待値プラスなら、25日線未回復でも打診候補
    if (
        state["deep_discount"]
        and not state["updating_low"]
        and weak_expect_positive
        and avg_dd180 <= 0.25
    ):
        return "打診買い候補", "signal-buy"

    # 期待値は悪くないが、まだ買い場感が薄い/下がっていない/反転確認不足
    if weak_expect_positive:
        return "待ち", "signal-wait"

    return "危険", "signal-danger"


def format_pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


def format_num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:.2f}"


def format_price(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "-"
    return f"{x:,.1f}"


def make_display_df(result_df: pd.DataFrame) -> pd.DataFrame:
    display = result_df.copy()

    pct_cols = [
        "期待リターン",
        "平均リターン",
        "中央値リターン",
        "勝率",
        "平均最大下落率",
        "最悪最大下落率",
    ]

    for col in pct_cols:
        display[col] = display[col].map(format_pct)

    for col in ["期待値ライン", "平均下落ライン", "最悪下落ライン"]:
        display[col] = display[col].map(format_price)

    return display


def box_style(trend: str) -> Tuple[str, str]:
    if trend == "strong_up":
        return "rgba(0, 92, 230, 0.40)", "rgba(0, 66, 190, 1.0)"
    if trend == "weak_up":
        return "rgba(85, 170, 255, 0.22)", "rgba(40, 135, 220, 0.9)"
    if trend == "weak_down":
        return "rgba(255, 145, 145, 0.22)", "rgba(220, 70, 70, 0.9)"
    if trend == "strong_down":
        return "rgba(230, 35, 35, 0.40)", "rgba(170, 0, 0, 1.0)"
    return "rgba(180, 180, 180, 0.10)", "rgba(150, 150, 150, 0.6)"


def box_legend_name(trend: str) -> str:
    # 文字は出さず、正方形だけに近い凡例にする
    names = {
        "strong_up": "■",
        "weak_up": "■ ",
        "weak_down": "■  ",
        "strong_down": "■   ",
    }
    return names.get(trend, "■")

def build_box_trace(df: pd.DataFrame, target_trend: str) -> go.Scatter:
    xs = []
    ys = []

    segments = get_trend_segments(df)

    for start_idx, end_idx, trend in segments:
        if trend != target_trend:
            continue

        for box_start, box_end in split_segment_to_step_boxes(start_idx, end_idx):
            sub = df.iloc[box_start:box_end + 1]
            x0 = sub.index[0]
            x1 = sub.index[-1]

            low = float(sub["Low"].min())
            high = float(sub["High"].max())
            pad = (high - low) * 0.12 if high > low else high * 0.01

            y0 = low - pad
            y1 = high + pad

            xs.extend([x0, x1, x1, x0, x0, None])
            ys.extend([y0, y0, y1, y1, y0, None])

    fillcolor, linecolor = box_style(target_trend)

    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        fill="toself",
        fillcolor=fillcolor,
        line=dict(color=linecolor, width=1),
        name=box_legend_name(target_trend),
        hoverinfo="skip",
        legendrank=20,
        legendgroup=target_trend,
        showlegend=False,
    )




def build_box_legend_trace(target_trend: str, rank: int) -> go.Scatter:
    fillcolor, linecolor = box_style(target_trend)

    # ver1.1方式：凡例用の小さい正方形マーク。
    # 箱本体とはlegendgroupで連動させる。
    name_map = {
        "strong_up": " ",
        "weak_up": "  ",
        "weak_down": "   ",
        "strong_down": "    ",
    }

    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(
            symbol="square",
            size=16,
            color=fillcolor,
            line=dict(color=linecolor, width=1.5),
        ),
        name=name_map.get(target_trend, " "),
        legendgroup=target_trend,
        showlegend=True,
        legendrank=rank,
        hoverinfo="skip",
    )


def make_chart(
    df: pd.DataFrame,
    ticker: str,
    result_df: pd.DataFrame,
    valuation: ValuationData,
) -> go.Figure:
    latest_close = float(df["Close"].iloc[-1])

    fig = go.Figure()

    for trend in ["strong_up", "weak_up", "weak_down", "strong_down"]:
        fig.add_trace(build_box_trace(df, trend))

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="株価",
            legendrank=1,
            increasing_line_color="#009688",
            decreasing_line_color="#e53935",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA5"],
            mode="lines",
            name="MA5",
            legendrank=2,
            line=dict(color="#c084fc", width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA25"],
            mode="lines",
            name="MA25",
            legendrank=3,
            line=dict(color="#7c3aed", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA75"],
            mode="lines",
            name="MA75",
            legendrank=4,
            line=dict(color="#3b0764", width=2.8),
        )
    )

    # 箱の表示/非表示は凡例の小さい■をクリック
    fig.add_trace(build_box_legend_trace("strong_up", 20))
    fig.add_trace(build_box_legend_trace("weak_up", 21))
    fig.add_trace(build_box_legend_trace("weak_down", 22))
    fig.add_trace(build_box_legend_trace("strong_down", 23))

    fig.add_hline(
        y=latest_close,
        line_color="#374151",
        line_width=2,
        annotation_text="現在値",
    )

    line_colors = {
        60: "#60a5fa",
        180: "#2563eb",
        360: "#1e3a8a",
    }

    for _, row in result_df.iterrows():
        days = int(row["保有日数"])
        expected_price = float(row["期待値ライン"])
        fig.add_hline(
            y=expected_price,
            line_color=line_colors.get(days, "#2563eb"),
            line_width=2,
            annotation_text=f"{days}日期待値",
        )

    fig.update_layout(
        title=f"{ticker} 株価チャート",
        xaxis_title="日付",
        yaxis_title="価格",
        height=680,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            groupclick="togglegroup",
            itemsizing="constant",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=60, b=110),
    )

    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")

    return fig



def get_split_adjusted_annual_dividend(ticker: str) -> pd.Series:
    """
    日本株向け配当判定用。

    重要：
    - カレンダー年ではなく、4月〜翌3月の会計年度で集計
    - NTTのような中間配当/期末配当のズレを避ける
    - yfinance配当がすでに分割補正済みの場合は二重補正を避ける
    """
    tk = yf.Ticker(ticker)

    try:
        div = tk.dividends
    except Exception:
        return pd.Series(dtype=float)

    if div is None or div.empty:
        return pd.Series(dtype=float)

    div = div.copy()
    div.index = pd.to_datetime(div.index)

    cutoff = pd.Timestamp.today(tz=div.index.tz) - pd.DateOffset(years=7)
    div = div[div.index >= cutoff]

    if div.empty:
        return pd.Series(dtype=float)

    raw_fiscal = aggregate_dividend_by_fiscal_year(div).tail(5)

    try:
        splits = tk.splits
    except Exception:
        splits = pd.Series(dtype=float)

    if splits is None or splits.empty:
        return raw_fiscal

    splits = splits.copy()
    splits.index = pd.to_datetime(splits.index)
    splits = splits[splits.index >= cutoff]

    if splits.empty:
        return raw_fiscal

    adjusted_values = []

    for div_date, div_amount in div.items():
        factor = 1.0
        future_splits = splits[splits.index > div_date]

        for _, split_ratio in future_splits.items():
            try:
                ratio = float(split_ratio)
                if ratio > 0:
                    factor *= ratio
            except Exception:
                pass

        adjusted_values.append(float(div_amount) / factor)

    adjusted = pd.Series(adjusted_values, index=div.index)
    adjusted_fiscal = aggregate_dividend_by_fiscal_year(adjusted).tail(5)

    raw_score = dividend_series_penalty(raw_fiscal)
    adjusted_score = dividend_series_penalty(adjusted_fiscal)

    # 補正後のほうが明らかに自然な場合だけ採用。
    # 差が小さい場合はyfinanceが補正済みとみなしてrawを採用。
    if adjusted_score + 0.08 < raw_score:
        return adjusted_fiscal

    return raw_fiscal


def aggregate_dividend_by_fiscal_year(div: pd.Series) -> pd.Series:
    """
    日本株想定で、4月〜翌3月を1会計年度として集計。
    例：
      2023年9月配当 + 2024年3月配当 = 2024年3月期相当
    """
    if div.empty:
        return pd.Series(dtype=float)

    fiscal_years = []
    for dt in div.index:
        # 1〜3月はその年の3月期
        # 4〜12月は翌年3月期
        fy = dt.year if dt.month <= 3 else dt.year + 1
        fiscal_years.append(fy)

    fiscal = pd.Series(div.values, index=fiscal_years)
    annual = fiscal.groupby(level=0).sum()
    annual = annual[annual > 0]
    annual = annual.sort_index()

    return annual


def dividend_series_penalty(annual_div: pd.Series) -> float:
    """
    配当系列の不自然さをスコア化。
    大幅な落ち込み・極端なブレがあるほど高い。
    """
    if annual_div.empty or len(annual_div) < 3:
        return 999.0

    values = annual_div.values.astype(float)
    penalty = 0.0

    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]

        if prev <= 0:
            continue

        change = (curr / prev) - 1

        if change <= -0.50:
            penalty += 1.00
        elif change <= -0.25:
            penalty += 0.50
        elif change <= -0.10:
            penalty += 0.20

        if change >= 1.00:
            penalty += 0.50
        elif change >= 0.50:
            penalty += 0.20

    cv = float(np.std(values) / np.mean(values)) if np.mean(values) > 0 else 1.0
    penalty += cv * 0.2

    return penalty


def get_dividend_summary(ticker: str) -> pd.Series:
    # 既存処理名との互換用。中身は分割補正済み配当を返す。
    return get_split_adjusted_annual_dividend(ticker)


def get_latest_operating_cf(ticker: str) -> Optional[float]:
    try:
        cf = yf.Ticker(ticker).cashflow
    except Exception:
        return None

    if cf is None or cf.empty:
        return None

    possible_rows = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]

    for row in possible_rows:
        if row in cf.index:
            values = cf.loc[row].dropna()
            if not values.empty:
                return safe_float(values.iloc[0])

    return None



def get_latest_free_cash_flow(ticker: str) -> Optional[float]:
    try:
        cf = yf.Ticker(ticker).cashflow
    except Exception:
        return None

    if cf is None or cf.empty:
        return None

    possible_fcf_rows = [
        "Free Cash Flow",
        "FreeCashFlow",
    ]

    for row in possible_fcf_rows:
        if row in cf.index:
            values = cf.loc[row].dropna()
            if not values.empty:
                return safe_float(values.iloc[0])

    # FCFが直接ない場合は 営業CF - 設備投資 で近似
    ocf = None
    capex = None

    for row in [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Cash Flow From Continuing Operating Activities",
    ]:
        if row in cf.index:
            values = cf.loc[row].dropna()
            if not values.empty:
                ocf = safe_float(values.iloc[0])
                break

    for row in [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Capital Expenditures",
    ]:
        if row in cf.index:
            values = cf.loc[row].dropna()
            if not values.empty:
                capex = safe_float(values.iloc[0])
                break

    if ocf is None or capex is None:
        return None

    # yfinanceのCapExはマイナス表示が多いので足す
    return ocf + capex


def judge_dividend_no_cut(annual_div: pd.Series) -> Tuple[str, str]:
    """
    会計年度ベースの年間配当で判定。
    小幅減配はNGではなく要注意。
    大幅減配・連続減配はNG。
    """
    if annual_div.empty or len(annual_div) < 3:
        return "ng", "会計年度ベースの配当データ不足"

    values = annual_div.values.astype(float)
    decline_rates = []

    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev > 0:
            decline_rates.append((curr / prev) - 1)

    if not decline_rates:
        return "ng", "減配判定不可"

    big_cut = any(r <= -0.25 for r in decline_rates)
    mid_cut = any(-0.25 < r <= -0.10 for r in decline_rates)
    small_cut = any(-0.10 < r <= -0.05 for r in decline_rates)

    consecutive_cuts = 0
    max_consecutive_cuts = 0

    for r in decline_rates:
        # -2%未満を実質的な減配として扱う
        if r < -0.02:
            consecutive_cuts += 1
            max_consecutive_cuts = max(max_consecutive_cuts, consecutive_cuts)
        else:
            consecutive_cuts = 0

    if big_cut:
        return "ng", "会計年度ベースで25%以上の大幅減配あり"

    if max_consecutive_cuts >= 2:
        return "ng", "会計年度ベースで2期連続の減配あり"

    if mid_cut:
        return "ng", "会計年度ベースで10%以上の減配あり"

    if small_cut:
        return "ng", "会計年度ベースで5%以上の小幅減配あり"

    return "ok", "会計年度ベースで大幅減配・連続減配なし"


def judge_special_dividend(annual_div: pd.Series) -> Tuple[str, str]:
    if annual_div.empty or len(annual_div) < 4:
        return "ng", "会計年度ベースの配当データ不足"

    values = annual_div.values.astype(float)
    median = float(np.median(values))

    if median <= 0:
        return "ng", "一時的高配当の判定不可"

    latest = values[-1]
    max_value = float(np.max(values))

    # 1年だけ大きく跳ねた場合は特別配当・一過性配当の可能性
    if max_value >= median * 1.6 and latest < max_value * 0.8:
        return "ng", "一時的な配当増の可能性"
    return "ok", "分割補正後で特別配当偏重の兆候は小さい"


def judge_payout_ratio(valuation: ValuationData) -> Tuple[str, str]:
    pr = valuation.payout_ratio
    if pr is None:
        return "ng", "配当性向データなし"

    if 0 < pr <= 0.70:
        return "ok", f"配当性向 {format_pct(pr)}"
    if pr <= 0.90:
        return "ng", f"配当性向やや高め {format_pct(pr)}"
    return "ng", f"配当性向が高い {format_pct(pr)}"


def judge_operating_cf(ticker: str) -> Tuple[str, str]:
    ocf = get_latest_operating_cf(ticker)
    fcf = get_latest_free_cash_flow(ticker)

    if ocf is None and fcf is None:
        return "ng", "営業CF/FCFデータなし"

    if ocf is None:
        return "ng", "営業CFデータなし"

    if fcf is None:
        if ocf > 0:
            return "ng", "営業CFは黒字だがFCFデータなし"
        return "ng", "営業CFが赤字"

    if ocf > 0 and fcf > 0:
        return "ok", "営業CF・FCFともに黒字"

    if ocf > 0 and fcf <= 0:
        return "ng", "営業CFは黒字だがFCFが赤字"

    if ocf <= 0 and fcf > 0:
        return "ng", "FCFは黒字だが営業CFが赤字"

    return "ng", "営業CF・FCFともに赤字"


def judge_cyclical_risk(ticker: str) -> Tuple[str, str]:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return "ng", "業種データなし"

    sector = str(info.get("sector", "") or "")
    industry = str(info.get("industry", "") or "")
    text = f"{sector} {industry}".lower()

    cyclical_keywords = [
        "steel", "commodity", "mining", "oil", "gas", "chemical",
        "marine", "shipping", "airlines", "auto", "automobile",
        "semiconductor", "construction", "real estate",
        "bank", "financial"
    ]

    if any(k in text for k in cyclical_keywords):
        return "ng", f"市況影響に注意: {industry or sector or '-'}"

    if sector or industry:
        return "ok", f"市況依存は極端ではなさそう: {industry or sector}"
    return "ng", "市況依存を判定できません"


def judge_weekly_falling_knife(df: pd.DataFrame) -> Tuple[str, str]:
    weekly = df["Close"].resample("W-FRI").last().dropna()

    if len(weekly) < 30:
        return "ng", "週足データ不足"

    ma13 = weekly.rolling(13).mean()
    ma26 = weekly.rolling(26).mean()

    close = float(weekly.iloc[-1])
    w13 = float(ma13.iloc[-1])
    w26 = float(ma26.iloc[-1])

    if close >= w13 and w13 >= w26:
        return "ok", "週足は上向き寄り"
    if close < w13 and w13 < w26:
        return "ng", "週足は落ちるナイフ気味"
    return "ng", "週足は方向感確認中"


def generate_buy_reason(signal: str, result_df: pd.DataFrame, factor_score: FactorScore, box_label: str) -> Tuple[str, str]:
    r180 = result_df[result_df["保有日数"] == 180].iloc[0]
    exp180 = float(r180["期待リターン"])

    if signal == "買い候補":
        reason = f"{box_label}、180日期待{format_pct(exp180)}、品質{factor_score.total}/100"
        return "ok", reason

    if exp180 > 0:
        reason = f"180日期待{format_pct(exp180)}だが総合判定は{signal}"
        return "ng", reason

    return "ng", "買う理由が弱い"



def judge_dividend_yield_min(valuation: ValuationData) -> Tuple[str, str]:
    y = valuation.dividend_yield

    if y is None:
        return "ng", "配当利回りデータなし"

    if y >= 0.035:
        return "ok", f"配当利回り {format_pct(y)}"

    return "ng", f"配当利回りが3.5%未満 {format_pct(y)}"


def build_auto_checklist_items(
    ticker: str,
    df: pd.DataFrame,
    valuation: ValuationData,
    result_df: pd.DataFrame,
    factor_score: FactorScore,
    signal: str,
    box_label: str,
) -> list:
    annual_div = get_dividend_summary(ticker)

    return [
        ("配当利回り3.5%以上", *judge_dividend_yield_min(valuation)),
        ("過去5年で大幅減配・連続減配していないか", *judge_dividend_no_cut(annual_div)),
        ("一時的・市況連動の高配当ではないか", *judge_special_dividend(annual_div)),
        ("配当性向が高すぎないか", *judge_payout_ratio(valuation)),
        ("営業CF・FCFが黒字か", *judge_operating_cf(ticker)),
        ("市況依存が強すぎないか", *judge_cyclical_risk(ticker)),

    ]


def render_sidebar_auto_checklist(
    ticker: str,
    df: pd.DataFrame,
    valuation: ValuationData,
    result_df: pd.DataFrame,
    factor_score: FactorScore,
    signal: str,
    box_label: str,
) -> None:
    st.sidebar.divider()
    st.sidebar.subheader("高配当安定チェックリスト")

    checks = build_auto_checklist_items(
        ticker=ticker,
        df=df,
        valuation=valuation,
        result_df=result_df,
        factor_score=factor_score,
        signal=signal,
        box_label=box_label,
    )

    for idx, (title, status, detail) in enumerate(checks):
        is_ok = status == "ok"
        # ticker/statusをkeyに入れて、前の銘柄のチェック状態が残るのを防ぐ
        st.sidebar.checkbox(
            title,
            value=is_ok,
            disabled=True,
            key=f"auto_check_{ticker}_{idx}_{status}_{title}",
        )

        if not is_ok:
            st.sidebar.markdown(
                f'<div class="side-reason">理由：{detail}</div>',
                unsafe_allow_html=True,
            )


def show_checklist() -> None:
    st.subheader("高配当安定チェックリスト")
    checklist = [
        "過去5年で減配してないか",
        "特別配当込みではないか",
        "配当性向が高すぎないか",
        "営業CFが黒字か",
        "市況依存が強すぎないか",
        "週足が落ちるナイフではないか",
        "買った理由を1行で説明できる",
    ]

    for item in checklist:
        st.checkbox(item, key=f"check_{item}")



def render_judge_ui(
    ticker: str,
    latest_close: float,
    signal: str,
    signal_class: str,
    result_df: pd.DataFrame,
    valuation: ValuationData,
) -> None:
    r60 = result_df[result_df["保有日数"] == 60].iloc[0]
    r180 = result_df[result_df["保有日数"] == 180].iloc[0]
    r360 = result_df[result_df["保有日数"] == 360].iloc[0]

    div_yield = valuation.dividend_yield
    div_amount = latest_close * div_yield if div_yield is not None and latest_close > 0 else None

    if signal in ["危険", "除外候補", "対象外"]:
        signal_color = "#dc2626"
    elif signal == "待ち":
        signal_color = "#b45309"
    else:
        signal_color = "#0f766e"

    html = f"""
    <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #111827;
            background: transparent;
        }}
        .wrap {{
            display: grid;
            grid-template-columns: 390px 1fr;
            gap: 16px;
            align-items: stretch;
            padding: 2px 2px 10px 2px;
            box-sizing: border-box;
        }}
        .card {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
            overflow: hidden;
            min-height: 150px;
            box-sizing: border-box;
        }}
        .summary {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            height: 100%;
        }}
        .expect {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            height: 100%;
        }}
        .cell {{
            padding: 22px 22px;
            box-sizing: border-box;
            border-right: 1px solid #e5e7eb;
        }}
        .cell:last-child {{
            border-right: none;
        }}
        .label {{
            color: #64748b;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.02em;
            margin-bottom: 10px;
        }}
        .value {{
            font-size: 31px;
            font-weight: 900;
            line-height: 1.05;
            letter-spacing: -0.02em;
        }}
        .sub {{
            color: #64748b;
            font-size: 12px;
            margin-top: 10px;
            line-height: 1.35;
        }}
        .expect .value {{
            color: #0f172a;
        }}
    </style>

    <div class="wrap">
        <div class="card">
            <div class="summary">
                <div class="cell">
                    <div class="label">総合判定</div>
                    <div class="value" style="color:{signal_color};">{signal}</div>
                    <div class="sub">{ticker}<br>現在値 {latest_close:,.1f}円</div>
                </div>
                <div class="cell">
                    <div class="label">配当利回り</div>
                    <div class="value">{format_pct(div_yield)}</div>
                    <div class="sub">({format_price(div_amount)}円 / {latest_close:,.1f}円)</div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="expect">
                <div class="cell">
                    <div class="label">60日期待値</div>
                    <div class="value">{format_pct(float(r60["期待リターン"]))}</div>
                    <div class="sub">ライン {format_price(float(r60["期待値ライン"]))}</div>
                </div>
                <div class="cell">
                    <div class="label">180日期待値</div>
                    <div class="value">{format_pct(float(r180["期待リターン"]))}</div>
                    <div class="sub">ライン {format_price(float(r180["期待値ライン"]))}</div>
                </div>
                <div class="cell">
                    <div class="label">360日期待値</div>
                    <div class="value">{format_pct(float(r360["期待リターン"]))}</div>
                    <div class="sub">ライン {format_price(float(r360["期待値ライン"]))}</div>
                </div>
            </div>
        </div>
    </div>
    """

    st.markdown("### 判定サマリー")
    components.html(html, height=178, scrolling=False)


def render_valuation_ui(valuation: ValuationData, factor_score: FactorScore) -> None:
    st.markdown("### 指標・銘柄品質")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("現在PER", format_num(valuation.current_per))
    c2.metric("予想PER", format_num(valuation.forward_per))
    c3.metric("PBR", format_num(valuation.pbr))
    c4.metric("配当利回り", format_pct(valuation.dividend_yield))
    c5.metric("ROE", format_pct(valuation.roe))
    c6.metric("配当性向", format_pct(valuation.payout_ratio))

    c7, c8, c9, c10, c11 = st.columns(5)
    c7.metric("過去PER中央値", format_num(valuation.hist_per_median))
    c8.metric("PER乖離", format_pct(valuation.per_gap))
    c9.metric("バリュー", f"{factor_score.value}/35")
    c10.metric("配当", f"{factor_score.dividend}/25")
    c11.metric("収益性", f"{factor_score.quality}/25")


def main() -> None:
    st.set_page_config(
        page_title="日本株 長期保有判定",
        layout="wide",
    )
    inject_css()
    st.title("日本株 長期保有判定ツール")
    with st.sidebar:
        st.header("入力")

        # main_ver1_2.py の「開く」リンクから渡された銘柄コードを初期表示する。
        # 例: http://localhost:8502/?ticker=9432.T
        default_ticker = "9432.T"
        try:
            query_ticker = st.query_params.get("ticker", default_ticker)
            if isinstance(query_ticker, list):
                query_ticker = query_ticker[0] if query_ticker else default_ticker
            default_ticker = normalize_ticker(str(query_ticker)) or default_ticker
        except Exception:
            pass

        raw_ticker = st.text_input("銘柄コード", value=default_ticker)
        ticker = normalize_ticker(raw_ticker)


    if not ticker:
        st.warning("銘柄コードを入力してください。")
        return

    df = download_price_data(ticker)

    if df.empty or len(df) < 700:
        st.error("株価データを十分に取得できませんでした。銘柄コードを確認してください。")
        return

    valuation = fetch_valuation_data(ticker, df)
    factor_score = calc_factor_score(valuation)
    result_df, match_level = calc_all_horizons(df)
    signal, signal_class = judge_signal(df, result_df, factor_score, valuation)

    latest_close = float(df["Close"].iloc[-1])
    latest_box = str(df["TREND_BOX"].iloc[-1])

    box_label = {
        "strong_up": "濃青：強上昇",
        "weak_up": "薄青：弱上昇",
        "weak_down": "薄赤：弱下落",
        "strong_down": "濃赤：強下落",
        "neutral": "箱なし",
    }.get(latest_box, "箱なし")

    render_judge_ui(
        ticker=ticker,
        latest_close=latest_close,
        signal=signal,
        signal_class=signal_class,
        result_df=result_df,
        valuation=valuation,
    )

    render_sidebar_auto_checklist(
        ticker=ticker,
        df=df,
        valuation=valuation,
        result_df=result_df,
        factor_score=factor_score,
        signal=signal,
        box_label=box_label,
    )

    st.subheader("チャート")

    st.plotly_chart(
        make_chart(
            df=df,
            ticker=ticker,
            result_df=result_df,
            valuation=valuation,
        ),
        use_container_width=True,
    )

    st.subheader("チャート期待値")
    st.dataframe(
        make_display_df(result_df),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        """
        <style>
        .explain-list {
            color: #6b7280;
            font-size: 14px;
            line-height: 1.8;
            margin-top: 10px;
        }
        .explain-row {
            display: grid;
            grid-template-columns: 120px 12px 1fr;
            column-gap: 4px;
        }
        .explain-label {
            font-weight: 700;
            white-space: nowrap;
        }
        .explain-colon {
            font-weight: 700;
            text-align: center;
        }
        </style>

        <div class="explain-list">
            <div class="explain-row"><div class="explain-label">・保有日数</div><div class="explain-colon">：</div><div>何日後の成績を見るか。60日、180日、360日で比較。</div></div>
            <div class="explain-row"><div class="explain-label">・類似局面数</div><div class="explain-colon">：</div><div>現在と似た過去局面を何件使ったか。多いほど統計は安定しやすい。</div></div>
            <div class="explain-row"><div class="explain-label">・期待リターン</div><div class="explain-colon">：</div><div>今回採用している期待値。似た局面の将来リターンの60%水準。</div></div>
            <div class="explain-row"><div class="explain-label">・平均リターン</div><div class="explain-colon">：</div><div>似た局面の平均成績。大きな上昇・下落に引っ張られやすい。</div></div>
            <div class="explain-row"><div class="explain-label">・中央値リターン</div><div class="explain-colon">：</div><div>真ん中の成績。外れ値に強く、現実感を見やすい。</div></div>
            <div class="explain-row"><div class="explain-label">・勝率</div><div class="explain-colon">：</div><div>その保有日数後にプラスで終わった割合。</div></div>
            <div class="explain-row"><div class="explain-label">・平均最大下落率</div><div class="explain-colon">：</div><div>保有中に平均でどれくらい下がったか。含み損耐性の目安。</div></div>
            <div class="explain-row"><div class="explain-label">・最悪最大下落率</div><div class="explain-colon">：</div><div>似た局面の中で最も深かった下落。リスク確認用。</div></div>
            <div class="explain-row"><div class="explain-label">・期待値ライン</div><div class="explain-colon">：</div><div>期待リターンを現在株価に反映した価格目安。</div></div>
            <div class="explain-row"><div class="explain-label">・平均下落ライン</div><div class="explain-colon">：</div><div>平均最大下落率を現在株価に反映した価格目安。</div></div>
            <div class="explain-row"><div class="explain-label">・最悪下落ライン</div><div class="explain-colon">：</div><div>最悪最大下落率を現在株価に反映した価格目安。</div></div>
            <div class="explain-row"><div class="explain-label">・総合判定</div><div class="explain-colon">：</div><div>下落率、下げ止まり、180日/360日期待値を見て判定。</div></div>
            <div class="explain-row"><div class="explain-label">・打診買い候補</div><div class="explain-colon">：</div><div>十分下がり、安値更新が止まりつつあるが、まだ本格反転前の状態。</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )



if __name__ == "__main__":
    main()
