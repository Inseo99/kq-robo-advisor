"""
VAA (Vigilant Asset Allocation) 전략 구현
------------------------------------------------------
"카나리아 자산"의 모멘텀으로 위험신호를 감지해 공격/방어 모드를
전환하는 동적(전술적) 자산배분 전략. (Keller & Keuning이 제안)

핵심 로직:
    1) 카나리아 자산(보통 대표 주가지수 몇 개)의 13612W 모멘텀을 계산
    2) 카나리아 자산 중 단 하나라도 모멘텀이 마이너스면 -> '위험신호'
       -> 즉시 안전자산(채권/현금) 중 모멘텀이 가장 좋은 자산으로 100% 이동
    3) 위험신호가 없으면 -> '평상시'
       -> 공격자산(주식 등) 중 모멘텀 상위 top_n개를 동일비중 편입

13612W 모멘텀 공식 (최근 1/3/6/12개월 수익률에 차등 가중치):
    score = 12*(P0/P1 - 1) + 4*(P0/P3 - 1) + 2*(P0/P6 - 1) + 1*(P0/P12 - 1)

특징: FAA보다 반응이 훨씬 빠르고 극단적이다 (부분적 방어가 아니라
      카나리아 신호 하나에도 전체 포지션이 즉시 전환됨).

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
# VAA 비중 계산 (단일 시점)
# ----------------------------------------------------------------------
def vaa_weights(prices: pd.DataFrame, asof_idx: int,
                 canary_assets: list, offensive_assets: list, safe_assets: list,
                 top_n_offensive: int = 2, top_n_safe: int = 1) -> dict:
    """
    canary_assets    : 위험신호 감지용 카나리아 자산 (보통 3~4개)
    offensive_assets : 평상시 투자할 공격자산 후보
    safe_assets      : 위험신호시 이동할 안전자산 후보
    """
    canary_score = momentum_13612w(prices[canary_assets], asof_idx)
    risk_off = (canary_score < 0).any()  # 카나리아 중 하나라도 마이너스면 위험신호

    if risk_off:
        safe_score = momentum_13612w(prices[safe_assets], asof_idx)
        selected = safe_score.nlargest(top_n_safe).index.tolist()
    else:
        off_score = momentum_13612w(prices[offensive_assets], asof_idx)
        selected = off_score.nlargest(top_n_offensive).index.tolist()

    weight = 1 / len(selected)
    return {a: weight for a in selected}, risk_off


# ----------------------------------------------------------------------
# 백테스트: 매월 신호를 갱신하며 VAA 전략 실행
# ----------------------------------------------------------------------
def backtest_vaa(prices: pd.DataFrame, monthly_returns: pd.DataFrame,
                  canary_assets: list, offensive_assets: list, safe_assets: list,
                  top_n_offensive: int = 2, top_n_safe: int = 1) -> pd.Series:
    """매월 말 VAA 신호를 계산해 다음 달 수익률에 적용하는 방식으로 백테스트"""
    dates = prices.index
    port_value = [1.0]
    values = []
    risk_off_history = []

    for i in range(12, len(dates) - 1):
        weights, risk_off = vaa_weights(prices, i, canary_assets, offensive_assets,
                                         safe_assets, top_n_offensive, top_n_safe)
        next_return = sum(w * monthly_returns[a].iloc[i + 1] for a, w in weights.items())
        port_value.append(port_value[-1] * (1 + next_return))
        values.append(port_value[-1])
        risk_off_history.append(risk_off)

    result_dates = dates[13: len(dates)]
    pv_series = pd.Series(values, index=result_dates, name="portfolio_value")
    risk_off_pct = np.mean(risk_off_history)
    return pv_series, risk_off_pct


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

    # 최근 시점 기준 VAA 비중 확인
    asof_idx = n_months - 1
    current_weights, current_risk_off = vaa_weights(
        prices, asof_idx, canary_assets, offensive_assets, safe_assets,
        top_n_offensive=2, top_n_safe=1
    )

    print("=" * 55)
    print(f"기준 시점: {dates[asof_idx].date()}")
    print(f"위험신호(risk-off): {current_risk_off}")
    print("=" * 55)
    for a, w in current_weights.items():
        print(f"  {a}: {w:.2%}")

    # 백테스트 실행
    port_value, risk_off_pct = backtest_vaa(
        prices, monthly_returns, canary_assets, offensive_assets, safe_assets,
        top_n_offensive=2, top_n_safe=1
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
        "VAA": metrics,
        "공격자산 동일비중 매수후보유 (비교용)": metrics_bh,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print(f"\n전체 기간 중 방어모드(risk-off) 비중: {risk_off_pct:.1%}")
    print("\n(참고: VAA는 카나리아 신호 하나에도 전체 포지션이 즉시")
    print(" 안전자산으로 전환되므로 FAA보다 반응이 빠르고 방어적입니다.")
    print(" 대신 신호가 자주 바뀌면(휩쏘) 거래비용과 기회손실이 커질 수 있습니다.)")