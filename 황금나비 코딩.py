"""
황금나비 (Golden Butterfly) 포트폴리오 전략 구현
------------------------------------------------------
미국 투자 커뮤니티에서 제안된 전략으로, 성장성과 안정성을
동시에 추구하는 균형형 자산배분 모델.
영구 포트폴리오의 단순성과 올웨더의 분산 개념을 절충한 형태.

5가지 자산군을 동일 비중(20%)으로 배분:
    - 소형주 20%       -> 성장성 강화
    - 대형주 20%       -> 안정적 성장
    - 장기채권 20%     -> 경기 침체·금리 하락 방어
    - 금 20%          -> 인플레이션·위기 대응
    - 현금/단기채권 20% -> 유동성 확보 및 디플레이션 방어

영구 포트폴리오와 비교하면 주식 비중이 25% -> 40%(소형+대형)로
늘어나 상승장 성과가 개선되는 대신, 하락장 변동성도 커진다.
고정비중 전략이므로 최적화는 필요 없고, 주기적 리밸런싱이 핵심이다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 황금나비 목표비중
# ----------------------------------------------------------------------
def golden_butterfly_portfolio() -> dict:
    """소형주/대형주/장기채/금/현금 각 20%"""
    return {
        "소형주": 0.20,
        "대형주": 0.20,
        "장기채권": 0.20,
        "금": 0.20,
        "현금": 0.20,
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
        "소형주": ["IWM"],
        "대형주": ["SPY"],
        "장기채권": ["TLT"],
        "금": ["GLD"],
        "현금": ["SHV"],
    }
    tickers = ["IWM", "SPY", "TLT", "GLD", "SHV"]

    # 자산별 가상 일간 수익률 생성 (실제 사용시엔 실제 가격 데이터의 수익률로 교체)
    n_days = 252 * 10  # 10년치
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mean_map = {"IWM": 0.08, "SPY": 0.09, "TLT": 0.03, "GLD": 0.04, "SHV": 0.015}
    vol_map = {"IWM": 0.22, "SPY": 0.18, "TLT": 0.13, "GLD": 0.15, "SHV": 0.005}
    returns_df = pd.DataFrame({
        t: np.random.normal(mean_map[t] / 252, vol_map[t] / np.sqrt(252), n_days)
        for t in tickers
    }, index=dates)

    # 황금나비 비중 산출
    weights = map_to_tickers(golden_butterfly_portfolio(), class_to_tickers)
    print("=" * 50)
    print("황금나비 포트폴리오 비중")
    print("=" * 50)
    for t, w in weights.items():
        print(f"  {t}: {w:.2%}")

    # 백테스트 (월간 리밸런싱) vs 영구 포트폴리오 비교
    port_value = backtest_static_allocation(returns_df, weights, rebalance="M")
    metrics = performance_metrics(port_value)

    weights_permanent = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "SHV": 0.25}
    port_value_perm = backtest_static_allocation(returns_df, weights_permanent, rebalance="M")
    metrics_perm = performance_metrics(port_value_perm)

    print("\n" + "=" * 50)
    print("백테스트 성과 비교 (월간 리밸런싱, 10년 시뮬레이션)")
    print("=" * 50)
    result = pd.DataFrame({
        "황금나비": metrics,
        "영구 포트폴리오 (비교용)": metrics_perm,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print("\n(참고: 황금나비는 영구 포트폴리오보다 주식 비중이 높아")
    print(" (25% -> 40%) 상승장에서 성과가 더 좋게 나오는 경향이 있지만,")
    print(" 하락장에서는 변동성이 더 커질 수 있습니다.)")