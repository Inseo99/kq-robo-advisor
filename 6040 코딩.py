"""
60/40 포트폴리오 전략 구현
------------------------------------------------------
가장 오래되고 널리 알려진 전통적 자산배분 방식.
    주식 60% + 채권 40%

- 주식: 장기 성장 동력 (기대수익률 높음, 변동성도 높음)
- 채권: 하락장 방어 역할 (변동성 완화)

고정비중 전략이므로 최적화가 필요 없다. 핵심은 주기적으로
목표비중(60/40)으로 되돌리는 '리밸런싱'이다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 60/40 목표비중
# ----------------------------------------------------------------------
def portfolio_60_40(stock_weight: float = 0.60) -> dict:
    """
    주식/채권 비중을 반환. 기본값은 60/40이지만 필요시 다른 비율로 조정 가능
    (예: 보수적으로 40/60을 쓰고 싶으면 stock_weight=0.40)
    """
    return {
        "주식": stock_weight,
        "채권": 1 - stock_weight,
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
        "채권": ["AGG"],
    }

    # 자산별 가상 일간 수익률 생성 (실제 사용시엔 실제 가격 데이터의 수익률로 교체)
    n_days = 252 * 10  # 10년치
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    returns_df = pd.DataFrame({
        "SPY": np.random.normal(0.09 / 252, 0.18 / np.sqrt(252), n_days),
        "AGG": np.random.normal(0.03 / 252, 0.05 / np.sqrt(252), n_days),
    }, index=dates)

    # 60/40 비중 산출
    weights = map_to_tickers(portfolio_60_40(), class_to_tickers)
    print("=" * 50)
    print("60/40 포트폴리오 비중")
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

    # 비교: 리밸런싱 없이 매수 후 보유(buy & hold)
    port_value_bh = backtest_static_allocation(returns_df, weights, rebalance=None)
    metrics_bh = performance_metrics(port_value_bh)

    print("\n" + "=" * 50)
    print("비교: 리밸런싱 없음 (Buy & Hold)")
    print("=" * 50)
    print(f"  CAGR   : {metrics_bh['CAGR']:.2%}")
    print(f"  MDD    : {metrics_bh['MDD']:.2%}")
    print(f"  Sharpe : {metrics_bh['Sharpe']:.2f}")