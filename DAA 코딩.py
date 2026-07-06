"""
DAA (Defensive Asset Allocation) 전략 구현
------------------------------------------------------
VAA의 "전부 아니면 전무" 식 급격한 on/off 전환을 완화한 버전.
(Keller & Keuning이 VAA를 확장해 제안)

핵심 로직:
    1) 카나리아 자산 n개 중 모멘텀이 마이너스인 자산이 b개라고 하면,
           현금비중(cash_fraction) = min(1, protection_factor * b / n)
       -> VAA처럼 "하나라도 마이너스면 100% 방어"가 아니라,
          "마이너스 비율만큼만 점진적으로 방어" 하는 방식

    2) 위험자산 비중(1 - cash_fraction)은 공격자산 상위 top_n_offensive개에
       동일비중으로 배분

    3) 현금비중은 안전자산 중 모멘텀이 가장 좋은 top_n_safe개에 배분

    protection_factor(보호계수, T)가 클수록 더 보수적으로 반응한다.
    (T=1이면 breadth 비율 그대로, T=2이면 신호에 두 배로 민감하게 반응)

13612W 모멘텀 공식 (최근 1/3/6/12개월 수익률에 차등 가중치):
    score = 12*(P0/P1 - 1) + 4*(P0/P3 - 1) + 2*(P0/P6 - 1) + 1*(P0/P12 - 1)

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 공통: 13612W 모멘텀 스코어
# ----------------------------------------------------------------------
def momentum_13612w(prices: pd.DataFrame, asof_idx: int) -> pd.Series:
    """
    월간 가격 데이터에서 asof_idx 시점 기준 13612W 모멘텀 스코어 계산
    prices   : index=월말 날짜, columns=자산 티커
    asof_idx : 정수 위치 (12 이상 필요)
    """
    p0 = prices.iloc[asof_idx]
    p1 = prices.iloc[asof_idx - 1]
    p3 = prices.iloc[asof_idx - 3]
    p6 = prices.iloc[asof_idx - 6]
    p12 = prices.iloc[asof_idx - 12]

    return (12 * (p0 / p1 - 1) + 4 * (p0 / p3 - 1)
            + 2 * (p0 / p6 - 1) + 1 * (p0 / p12 - 1))


# ----------------------------------------------------------------------
# DAA 비중 계산 (단일 시점)
# ----------------------------------------------------------------------
def daa_weights(prices: pd.DataFrame, asof_idx: int,
                 canary_assets: list, offensive_assets: list, safe_assets: list,
                 top_n_offensive: int = 6, top_n_safe: int = 1,
                 protection_factor: float = 1.0) -> tuple:
    """
    canary_assets     : 위험신호 감지용 카나리아 자산
    offensive_assets  : 공격자산 후보
    safe_assets       : 안전자산 후보
    protection_factor : breadth 신호에 대한 민감도 (T). 1이면 비율 그대로 반영

    반환값: (비중 딕셔너리, 현금비중)
    """
    canary_score = momentum_13612w(prices[canary_assets], asof_idx)
    n = len(canary_assets)
    b = (canary_score < 0).sum()
    cash_fraction = min(1.0, protection_factor * b / n)
    risky_fraction = 1.0 - cash_fraction

    weights = {}

    if risky_fraction > 0:
        off_score = momentum_13612w(prices[offensive_assets], asof_idx)
        selected_off = off_score.nlargest(top_n_offensive).index.tolist()
        w_off = risky_fraction / len(selected_off)
        for a in selected_off:
            weights[a] = weights.get(a, 0) + w_off

    if cash_fraction > 0:
        safe_score = momentum_13612w(prices[safe_assets], asof_idx)
        selected_safe = safe_score.nlargest(top_n_safe).index.tolist()
        w_safe = cash_fraction / len(selected_safe)
        for a in selected_safe:
            weights[a] = weights.get(a, 0) + w_safe

    return weights, cash_fraction


# ----------------------------------------------------------------------
# 백테스트: 매월 신호를 갱신하며 DAA 전략 실행
# ----------------------------------------------------------------------
def backtest_daa(prices: pd.DataFrame, monthly_returns: pd.DataFrame,
                  canary_assets: list, offensive_assets: list, safe_assets: list,
                  top_n_offensive: int = 6, top_n_safe: int = 1,
                  protection_factor: float = 1.0) -> tuple:
    """매월 말 DAA 신호를 계산해 다음 달 수익률에 적용하는 방식으로 백테스트"""
    dates = prices.index
    port_value = [1.0]
    values = []
    cash_fraction_history = []

    for i in range(12, len(dates) - 1):
        weights, cash_fraction = daa_weights(
            prices, i, canary_assets, offensive_assets, safe_assets,
            top_n_offensive, top_n_safe, protection_factor
        )
        next_return = sum(w * monthly_returns[a].iloc[i + 1] for a, w in weights.items())
        port_value.append(port_value[-1] * (1 + next_return))
        values.append(port_value[-1])
        cash_fraction_history.append(cash_fraction)

    result_dates = dates[13: len(dates)]
    pv_series = pd.Series(values, index=result_dates, name="portfolio_value")
    avg_cash_fraction = np.mean(cash_fraction_history)
    return pv_series, avg_cash_fraction


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

    # 카나리아: 대표 주가지수 4개 / 공격자산: 주식+금+원자재 / 안전자산: 중기채+현금
    canary_assets = ["SPY", "EFA", "EEM", "AGG"]
    offensive_assets = ["SPY", "EFA", "EEM", "GLD", "DBC"]
    safe_assets = ["IEF", "SHY"]
    tickers = sorted(set(canary_assets + offensive_assets + safe_assets))

    mean_map = {"SPY": 0.09, "EFA": 0.06, "EEM": 0.07, "AGG": 0.03,
                "GLD": 0.04, "DBC": 0.03, "IEF": 0.025, "SHY": 0.015}
    vol_map = {"SPY": 0.18, "EFA": 0.19, "EEM": 0.24, "AGG": 0.05,
               "GLD": 0.15, "DBC": 0.18, "IEF": 0.07, "SHY": 0.02}

    n_months = 15 * 12  # 15년치 월간 데이터
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    monthly_returns = pd.DataFrame(
        {t: np.random.normal(mean_map[t] / 12, vol_map[t] / np.sqrt(12), n_months)
         for t in tickers},
        index=dates,
    )
    prices = (1 + monthly_returns).cumprod() * 100

    # 최근 시점 기준 DAA 비중 확인
    asof_idx = n_months - 1
    current_weights, current_cash_fraction = daa_weights(
        prices, asof_idx, canary_assets, offensive_assets, safe_assets,
        top_n_offensive=3, top_n_safe=1, protection_factor=1.0
    )

    print("=" * 55)
    print(f"기준 시점: {dates[asof_idx].date()}")
    print(f"현금비중(cash fraction): {current_cash_fraction:.1%}")
    print("=" * 55)
    for a, w in current_weights.items():
        print(f"  {a}: {w:.2%}")

    # 백테스트 실행
    port_value, avg_cash = backtest_daa(
        prices, monthly_returns, canary_assets, offensive_assets, safe_assets,
        top_n_offensive=3, top_n_safe=1, protection_factor=1.0
    )
    metrics = performance_metrics(port_value)

    # 비교용: 공격자산 동일비중 매수 후 보유
    buy_hold_returns = monthly_returns[offensive_assets].mean(axis=1)
    buy_hold_value = (1 + buy_hold_returns.iloc[12:]).cumprod()
    metrics_bh = performance_metrics(buy_hold_value)

    print("\n" + "=" * 55)
    print("백테스트 성과 비교 (15년, 월간 리밸런싱)")
    print("=" * 55)
    result = pd.DataFrame({
        "DAA": metrics,
        "공격자산 동일비중 매수후보유 (비교용)": metrics_bh,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print(f"\n전체 기간 평균 현금비중: {avg_cash:.1%}")
    print("\n(참고: DAA는 VAA와 달리 카나리아 신호 '비율'만큼만 현금화하므로")
    print(" 방어 전환이 점진적입니다. protection_factor를 높이면 VAA처럼")
    print(" 더 급격하게 반응하도록 조절할 수 있습니다.)")