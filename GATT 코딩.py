"""
GATT (=GTAA, Global Tactical Asset Allocation) 전략 구현
------------------------------------------------------
자산군 간 상관관계나 최적화 대신, 각 자산의 '추세'만 보고
투자 여부를 결정하는 가장 단순한 형태의 동적(전술적) 자산배분.
(Mebane Faber의 "10개월 이동평균 전략"과 동일한 방식)

핵심 로직:
    각 위험자산의 현재 가격이 자신의 lookback개월(보통 10개월)
    이동평균보다 높으면 -> 보유 (risk-on)
    낮으면            -> 그 비중만큼 안전자산(현금/채권)으로 이동 (risk-off)

    각 자산은 완전히 독립적으로 판단하므로, "일부는 보유하고
    일부는 현금으로 이동"하는 부분적 방어가 자연스럽게 일어난다.

특징:
    - 공분산 행렬이나 최적화가 전혀 필요 없음 (계산이 매우 단순)
    - 자산별 독립 판단 -> 시장 전체가 하락해도 상승 추세인 자산은 계속 보유
    - 이동평균 기간(lookback)이 짧을수록 반응은 빠르지만 매매(휩쏘)도 잦아짐

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# GATT/GTAA 비중 계산 (단일 시점)
# ----------------------------------------------------------------------
def gatt_weights(prices: pd.DataFrame, asof_idx: int, risky_assets: list,
                  safe_asset: str, lookback: int = 10) -> dict:
    """
    prices       : 월말 가격 데이터 (index=날짜, columns=자산 티커)
    asof_idx     : prices.index 상의 정수 위치 (판단 시점, lookback 이상이어야 함)
    risky_assets : 추세추종 대상 위험자산 리스트
    safe_asset   : 위험신호 시 이동할 안전자산 (현금성 자산 등)
    lookback     : 이동평균 기간 (개월). Faber 원 논문은 10개월 사용
    """
    n = len(risky_assets)
    base_weight = 1 / n
    weights = {a: 0.0 for a in risky_assets}
    weights[safe_asset] = 0.0

    window = prices[risky_assets].iloc[asof_idx - lookback + 1: asof_idx + 1]
    sma = window.mean()
    current_price = prices[risky_assets].iloc[asof_idx]

    for asset in risky_assets:
        if current_price[asset] > sma[asset]:
            weights[asset] += base_weight  # 추세 상승 -> 보유
        else:
            weights[safe_asset] += base_weight  # 추세 하락 -> 안전자산 이동

    return {k: v for k, v in weights.items() if v > 0}


# ----------------------------------------------------------------------
# 백테스트: 매월 말 신호를 갱신하며 GATT 전략 실행
# ----------------------------------------------------------------------
def backtest_gatt(prices: pd.DataFrame, monthly_returns: pd.DataFrame,
                   risky_assets: list, safe_asset: str,
                   lookback: int = 10) -> pd.Series:
    """
    매월 말 이동평균 신호를 확인해 다음 달 비중을 결정하는 방식으로 백테스트.
    prices, monthly_returns : 같은 index(월말)를 가진 가격/수익률 DataFrame
    """
    all_assets = risky_assets + [safe_asset]
    dates = prices.index
    port_value = [1.0]
    values = []

    for i in range(lookback, len(dates) - 1):
        weights = gatt_weights(prices, i, risky_assets, safe_asset, lookback)
        next_return = sum(weights.get(a, 0) * monthly_returns[a].iloc[i + 1]
                           for a in all_assets)
        port_value.append(port_value[-1] * (1 + next_return))
        values.append(port_value[-1])

    result_dates = dates[lookback + 1: len(dates)]
    return pd.Series(values, index=result_dates, name="portfolio_value")


def performance_metrics(port_value: pd.Series, periods_per_year: int = 12,
                         risk_free_rate: float = 0.0) -> dict:
    """CAGR, MDD, Sharpe Ratio 계산 (월간 데이터 기준)"""
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
    np.random.seed(7)

    # 자산 유니버스: 주식, 선진국주식, 신흥국주식, 금, 원자재 (위험자산) + 현금(안전자산)
    risky_assets = ["SPY", "EFA", "EEM", "GLD", "DBC"]
    safe_asset = "SHY"
    tickers = risky_assets + [safe_asset]

    mean_map = {"SPY": 0.09, "EFA": 0.06, "EEM": 0.07, "GLD": 0.04,
                "DBC": 0.03, "SHY": 0.015}
    vol_map = {"SPY": 0.18, "EFA": 0.19, "EEM": 0.24, "GLD": 0.15,
               "DBC": 0.18, "SHY": 0.02}

    n_months = 15 * 12  # 15년치 월간 데이터
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    monthly_returns = pd.DataFrame(
        {t: np.random.normal(mean_map[t] / 12, vol_map[t] / np.sqrt(12), n_months)
         for t in tickers},
        index=dates,
    )
    prices = (1 + monthly_returns).cumprod() * 100  # 월말 가격 지수화

    # 최근 시점 기준 GATT 비중 확인
    asof_idx = n_months - 1
    current_weights = gatt_weights(prices, asof_idx, risky_assets, safe_asset, lookback=10)

    print("=" * 55)
    print(f"기준 시점: {dates[asof_idx].date()} - GATT 비중")
    print("=" * 55)
    for a, w in current_weights.items():
        print(f"  {a}: {w:.2%}")

    # 백테스트 실행 (10개월 이동평균 기준)
    port_value = backtest_gatt(prices, monthly_returns, risky_assets, safe_asset, lookback=10)
    metrics = performance_metrics(port_value)

    # 비교용: 위험자산 동일비중 매수 후 보유(추세추종 없음)
    buy_hold_returns = monthly_returns[risky_assets].mean(axis=1)
    buy_hold_value = (1 + buy_hold_returns.iloc[10:]).cumprod()
    metrics_bh = performance_metrics(buy_hold_value)

    print("\n" + "=" * 55)
    print("백테스트 성과 비교 (15년, 월간 리밸런싱)")
    print("=" * 55)
    result = pd.DataFrame({
        "GATT (추세추종)": metrics,
        "동일비중 매수후보유 (비교용)": metrics_bh,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print("\n(참고: GATT는 하락 추세에 있는 자산만 선별적으로 현금화하므로")
    print(" 이론상 MDD가 매수후보유보다 낮게 나오는 경향이 있습니다.")
    print(" 다만 이동평균 신호는 후행적이라 급락 초입의 손실은 못 피합니다.)")