"""
올웨더 (All Weather) 포트폴리오 전략 구현
------------------------------------------------------
세계적 투자자 레이 달리오(Ray Dalio)가 제안한 전략.
경제 사이클(성장/침체/인플레이션/디플레이션)에 관계없이 안정적인
성과를 추구하는 다자산 분산 투자 모델.

구성 (일반적으로 제안되는 비중):
    - 주식 30%      -> 경제 성장기에 수익 창출
    - 장기채권 40%   -> 경기 침체·금리 하락기에 안정성 제공
    - 중기채권 15%   -> 변동성 완화 및 균형 역할
    - 금 7.5%       -> 위기 상황에서 가치 보존
    - 원자재 7.5%    -> 인플레이션 방어

영구 포트폴리오와 개념은 비슷하지만 채권 비중이 훨씬 크고
(장기+중기 합쳐 55%), 자산군이 더 세분화되어 있는 것이 특징.
고정비중 전략이므로 최적화는 필요 없고, 주기적 리밸런싱이 핵심이다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 올웨더 목표비중
# ----------------------------------------------------------------------
def all_weather_portfolio() -> dict:
    """주식30% + 장기채40% + 중기채15% + 금7.5% + 원자재7.5%"""
    return {
        "주식": 0.30,
        "장기채권": 0.40,
        "중기채권": 0.15,
        "금": 0.075,
        "원자재": 0.075,
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
        "중기채권": ["IEF"],
        "금": ["GLD"],
        "원자재": ["DBC"],
    }
    tickers = ["SPY", "TLT", "IEF", "GLD", "DBC"]

    # 자산별 가상 일간 수익률 생성 (실제 사용시엔 실제 가격 데이터의 수익률로 교체)
    n_days = 252 * 10  # 10년치
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mean_map = {"SPY": 0.09, "TLT": 0.03, "IEF": 0.025, "GLD": 0.04, "DBC": 0.03}
    vol_map = {"SPY": 0.18, "TLT": 0.13, "IEF": 0.07, "GLD": 0.15, "DBC": 0.18}
    returns_df = pd.DataFrame({
        t: np.random.normal(mean_map[t] / 252, vol_map[t] / np.sqrt(252), n_days)
        for t in tickers
    }, index=dates)

    # 올웨더 비중 산출
    weights = map_to_tickers(all_weather_portfolio(), class_to_tickers)
    print("=" * 50)
    print("올웨더 포트폴리오 비중")
    print("=" * 50)
    for t, w in weights.items():
        print(f"  {t}: {w:.2%}")

    # 백테스트 (월간 리밸런싱) vs 60/40, 영구 포트폴리오 비교
    port_value = backtest_static_allocation(returns_df, weights, rebalance="M")
    metrics = performance_metrics(port_value)

    weights_6040 = {"SPY": 0.60, "TLT": 0.40}
    port_value_6040 = backtest_static_allocation(returns_df, weights_6040, rebalance="M")
    metrics_6040 = performance_metrics(port_value_6040)

    weights_permanent = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "IEF": 0.25}
    port_value_perm = backtest_static_allocation(returns_df, weights_permanent, rebalance="M")
    metrics_perm = performance_metrics(port_value_perm)

    print("\n" + "=" * 50)
    print("백테스트 성과 비교 (월간 리밸런싱, 10년 시뮬레이션)")
    print("=" * 50)
    result = pd.DataFrame({
        "올웨더": metrics,
        "60/40 (비교용)": metrics_6040,
        "영구 포트폴리오 (비교용)": metrics_perm,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print("\n(참고: 올웨더는 채권 비중(55%)이 매우 높아 금리 급등기에는")
    print(" 성과가 악화될 수 있지만, 자산군이 세분화되어 있어 다양한")
    print(" 경제 국면에 대응력이 좋은 편입니다.)")