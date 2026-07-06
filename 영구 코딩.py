"""
영구 포트폴리오 (Permanent Portfolio) 전략 구현
------------------------------------------------------
미국 경제학자 해리 브라운(Harry Browne)이 제안한 전략.
어떤 경제 상황에서도 안정적인 성과를 내는 것이 목표.

4가지 자산군을 동일 비중(25%)으로 배분:
    - 주식 25%   -> 경제 성장기에 수익 창출
    - 장기채권 25% -> 경기 침체·금리 하락기에 안정성 제공
    - 금 25%      -> 인플레이션·위기 상황에서 가치 보존
    - 현금 25%    -> 유동성 확보 및 디플레이션 방어

경제 사이클(성장/침체/인플레이션/디플레이션) 각각에 대응하는
자산을 하나씩 배치한 것이 핵심 아이디어. 고정비중이므로 최적화는
필요 없고, 주기적 리밸런싱이 핵심이다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 영구 포트폴리오 목표비중
# ----------------------------------------------------------------------
def permanent_portfolio() -> dict:
    """주식/장기채권/금/현금 각 25%"""
    return {
        "주식": 0.25,
        "장기채권": 0.25,
        "금": 0.25,
        "현금": 0.25,
    }


def map_to_tickers(class_weights: dict, class_to_tickers: dict) -> dict:
    """
    자산군별 목표비중과 실제 보유 티커 매핑을 받아 최종 비중을 산출.
    같은 자산군에 티커가 여러 개면 균등 분배한다.
    """
    ticker_weights = {}
    for asset_class, weight in class_weights.items():
        tickers = class_to_tickers.get(asset_class)
        if not tickers:
            raise ValueError(f"'{asset_class}' 자산군에 매핑된 티커가 없습니다.")
        per_ticker_weight = weight / len(tickers)
        for t in tickers:
            ticker_weights[t] = ticker_weights.get(t, 0) + per_ticker_weight
    return ticker_weights


# ----------------------------------------------------------------------
# 백테스트: 목표비중을 주기적으로 리밸런싱하며 성과 계산
# ----------------------------------------------------------------------
def backtest_static_allocation(returns_df: pd.DataFrame, weights: dict,
                                rebalance: str = "M") -> pd.Series:
    """
    returns_df : 기간별 수익률 DataFrame (컬럼 = 티커)
    weights    : {티커: 목표비중} 딕셔너리 (합계 1.0)
    rebalance  : 리밸런싱 주기 ('M'=월, 'Q'=분기, 'Y'=연, None=리밸런싱 없음)
    """
    tickers = list(weights.keys())
    w0 = np.array([weights[t] for t in tickers])
    r = returns_df[tickers].values
    dates = returns_df.index

    period_map = {"M": "month", "Q": "quarter", "Y": "year"}
    holdings = w0 * 1.0
    values = []
    last_period = None

    for i, date in enumerate(dates):
        if rebalance is not None:
            period = getattr(date, period_map[rebalance])
            if last_period is not None and period != last_period:
                holdings = w0 * holdings.sum()  # 목표비중으로 재조정
            last_period = period

        holdings = holdings * (1 + r[i])
        values.append(holdings.sum())

    return pd.Series(values, index=dates, name="portfolio_value")


def performance_metrics(port_value: pd.Series, periods_per_year: int = 252,
                         risk_free_rate: float = 0.0) -> dict:
    """CAGR, MDD, Sharpe Ratio 계산"""
    returns = port_value.pct_change().dropna()

    n_years = len(port_value) / periods_per_year
    cagr = (port_value.iloc[-1] / port_value.iloc[0]) ** (1 / n_years) - 1

    running_max = port_value.cummax()
    drawdown = port_value / running_max - 1
    mdd = drawdown.min()

    excess_return = returns - risk_free_rate / periods_per_year
    sharpe = np.sqrt(periods_per_year) * excess_return.mean() / returns.std()

    return {"CAGR": cagr, "MDD": mdd, "Sharpe": sharpe}


# ----------------------------------------------------------------------
# 예시 실행
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 실제 보유 자산군 -> 티커 매핑 (예시: 실제 사용시 본인 보유 티커로 교체)
    class_to_tickers = {
        "주식": ["SPY"],
        "장기채권": ["TLT"],
        "금": ["GLD"],
        "현금": ["SHV"],
    }
    tickers = ["SPY", "TLT", "GLD", "SHV"]

    # 자산별 가상 일간 수익률 생성 (실제 사용시엔 실제 가격 데이터의 수익률로 교체)
    n_days = 252 * 10  # 10년치
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mean_map = {"SPY": 0.09, "TLT": 0.03, "GLD": 0.04, "SHV": 0.015}
    vol_map = {"SPY": 0.18, "TLT": 0.13, "GLD": 0.15, "SHV": 0.005}
    returns_df = pd.DataFrame({
        t: np.random.normal(mean_map[t] / 252, vol_map[t] / np.sqrt(252), n_days)
        for t in tickers
    }, index=dates)

    # 영구 포트폴리오 비중 산출
    weights = map_to_tickers(permanent_portfolio(), class_to_tickers)
    print("=" * 50)
    print("영구 포트폴리오 비중")
    print("=" * 50)
    for t, w in weights.items():
        print(f"  {t}: {w:.2%}")

    # 백테스트 (월간 리밸런싱) vs 60/40 비교
    port_value = backtest_static_allocation(returns_df, weights, rebalance="M")
    metrics = performance_metrics(port_value)

    weights_6040 = {"SPY": 0.60, "TLT": 0.40}
    port_value_6040 = backtest_static_allocation(returns_df, weights_6040, rebalance="M")
    metrics_6040 = performance_metrics(port_value_6040)

    print("\n" + "=" * 50)
    print("백테스트 성과 비교 (월간 리밸런싱, 10년 시뮬레이션)")
    print("=" * 50)
    result = pd.DataFrame({
        "영구 포트폴리오": metrics,
        "60/40 (비교용)": metrics_6040,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print("\n(참고: 영구 포트폴리오는 금·현금 비중이 높아 상승장에서는")
    print(" 60/40보다 수익률이 낮게 나오는 경향이 있지만, MDD(최대낙폭)는")
    print(" 더 방어적으로 나오는 것이 일반적입니다.)")