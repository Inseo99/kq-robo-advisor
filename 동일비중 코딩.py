"""
동일비중 (Equal Weight) 포트폴리오 전략 구현
------------------------------------------------------
가장 단순하면서도 직관적인 자산배분 방식.
모든 자산군에 동일한 비중을 배분한다.

    n개 자산이 있으면 각각 1/n씩 배분

계산이 필요 없고(공분산·모멘텀·최적화 모두 불필요) 누구나 쉽게
적용 가능한 것이 최대 장점. 대신 위험·수익률을 전혀 고려하지 않기
때문에, 변동성이 큰 자산에도 다른 자산과 똑같은 비중이 들어가
비효율이 생길 수 있다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 동일비중 목표비중
# ----------------------------------------------------------------------
def equal_weight_portfolio(asset_classes: list) -> dict:
    """전달받은 자산(군) 리스트를 n등분"""
    n = len(asset_classes)
    return {a: 1 / n for a in asset_classes}


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
        "채권": ["TLT"],
        "금": ["GLD"],
        "원자재": ["DBC"],
        "현금": ["SHV"],
    }
    tickers = ["SPY", "TLT", "GLD", "DBC", "SHV"]

    # 자산별 가상 일간 수익률 생성 (실제 사용시엔 실제 가격 데이터의 수익률로 교체)
    n_days = 252 * 10  # 10년치
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    mean_map = {"SPY": 0.09, "TLT": 0.03, "GLD": 0.04, "DBC": 0.03, "SHV": 0.015}
    vol_map = {"SPY": 0.18, "TLT": 0.13, "GLD": 0.15, "DBC": 0.18, "SHV": 0.005}
    returns_df = pd.DataFrame({
        t: np.random.normal(mean_map[t] / 252, vol_map[t] / np.sqrt(252), n_days)
        for t in tickers
    }, index=dates)

    # 동일비중 비중 산출 (보유 자산군 5개 -> 각 20%)
    weights = map_to_tickers(
        equal_weight_portfolio(list(class_to_tickers.keys())), class_to_tickers
    )
    print("=" * 50)
    print("동일비중 포트폴리오 비중")
    print("=" * 50)
    for t, w in weights.items():
        print(f"  {t}: {w:.2%}")

    # 백테스트 (월간 리밸런싱)
    port_value = backtest_static_allocation(returns_df, weights, rebalance="M")
    metrics = performance_metrics(port_value)

    print("\n" + "=" * 50)
    print("백테스트 성과 (월간 리밸런싱, 10년 시뮬레이션)")
    print("=" * 50)
    print(f"  CAGR   : {metrics['CAGR']:.2%}")
    print(f"  MDD    : {metrics['MDD']:.2%}")
    print(f"  Sharpe : {metrics['Sharpe']:.2f}")

    # 참고: 각 자산이 실제 위험에 얼마나 기여하는지 확인 (변동성 기준 비교)
    print("\n" + "=" * 50)
    print("참고: 자산별 개별 변동성 (동일비중이라도 위험 기여는 다름)")
    print("=" * 50)
    volatility = returns_df.std() * np.sqrt(252)
    for t in tickers:
        print(f"  {t}: {volatility[t]:.2%}  (비중은 동일하게 20%)")

    print("\n(참고: 동일비중은 '금액'만 균등하게 나눌 뿐, 자산별 변동성")
    print(" 차이는 전혀 고려하지 않습니다. 변동성이 큰 자산이 포트폴리오")
    print(" 전체 위험에 더 크게 기여하게 됩니다 - ERC/역변동성과의 차이점.)")