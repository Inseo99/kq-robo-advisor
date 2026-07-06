"""
FAA (Flexible Asset Allocation) 전략 구현
------------------------------------------------------
모멘텀과 위험(변동성·상관관계) 지표를 동시에 고려해
자산 비중을 동적으로 조정하는 전술적 자산배분 전략.
(Wouter Keller가 제안한 모델을 단순화하여 구현)

핵심 로직 - 자산마다 3가지 순위를 매긴 뒤 합산:
    1) 모멘텀(M)   : 최근 momentum_lookback개월 누적수익률 -> 높을수록 좋음
    2) 변동성(V)   : 최근 vol_lookback개월 수익률 표준편차   -> 낮을수록 좋음
    3) 상관관계(C) : 다른 자산들과의 평균 상관계수          -> 낮을수록 좋음

    종합순위 = rank(M, 내림차순) + rank(V, 오름차순) + rank(C, 오름차순)
    종합순위가 가장 좋은(=낮은) 상위 top_n개 자산을 동일비중으로 편입

+ '절대 모멘텀' 필터: 모멘텀이 마이너스인 자산은 아예 후보에서 제외하고
  그 비중만큼 안전자산(현금)으로 이동 (원 FAA 논문의 핵심 방어 장치)

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# FAA 비중 계산 (단일 시점)
# ----------------------------------------------------------------------
def faa_weights(monthly_returns: pd.DataFrame, asof_idx: int,
                 universe: list, safe_asset: str = None,
                 top_n: int = 3, momentum_lookback: int = 12,
                 vol_lookback: int = 12, corr_lookback: int = 12) -> dict:
    """
    monthly_returns : 월간 수익률 DataFrame (index=날짜, columns=자산 티커)
    asof_idx        : monthly_returns.index 상의 정수 위치 (판단 시점)
    universe        : 투자 후보 자산 리스트
    safe_asset      : 절대모멘텀이 마이너스인 자산을 대체할 안전자산 (None이면 필터 미적용)
    top_n           : 최종 선택할 자산 개수
    """
    lookback = max(momentum_lookback, vol_lookback, corr_lookback)
    if asof_idx < lookback:
        raise ValueError("asof_idx가 lookback 기간보다 작아 계산할 수 없습니다.")

    # 1) 모멘텀: 최근 momentum_lookback개월 누적수익률
    ret_window = monthly_returns[universe].iloc[asof_idx - momentum_lookback + 1: asof_idx + 1]
    momentum = (1 + ret_window).prod() - 1

    # 2) 변동성: 최근 vol_lookback개월 수익률 표준편차
    vol_window = monthly_returns[universe].iloc[asof_idx - vol_lookback + 1: asof_idx + 1]
    volatility = vol_window.std()

    # 3) 상관관계: 최근 corr_lookback개월 자산간 평균 상관계수 (자기 자신 제외)
    corr_window = monthly_returns[universe].iloc[asof_idx - corr_lookback + 1: asof_idx + 1]
    corr_matrix = corr_window.corr()
    avg_corr = (corr_matrix.sum() - 1) / (len(universe) - 1)

    # 순위 계산 및 합산 (값이 작을수록 좋은 순위)
    rank_momentum = momentum.rank(ascending=False)   # 모멘텀 높을수록 좋음
    rank_vol = volatility.rank(ascending=True)         # 변동성 낮을수록 좋음
    rank_corr = avg_corr.rank(ascending=True)          # 상관관계 낮을수록 좋음
    combined_rank = rank_momentum + rank_vol + rank_corr

    selected = combined_rank.nsmallest(top_n).index.tolist()

    # 절대모멘텀 필터: 선택된 자산이라도 모멘텀이 마이너스면 안전자산으로 대체
    weights = {}
    weight_each = 1 / top_n
    for asset in selected:
        if safe_asset is not None and momentum[asset] < 0:
            weights[safe_asset] = weights.get(safe_asset, 0) + weight_each
        else:
            weights[asset] = weight_each

    return weights


# ----------------------------------------------------------------------
# 백테스트: 매월 신호를 갱신하며 FAA 전략 실행
# ----------------------------------------------------------------------
def backtest_faa(monthly_returns: pd.DataFrame, universe: list,
                  safe_asset: str = None, top_n: int = 3,
                  momentum_lookback: int = 12, vol_lookback: int = 12,
                  corr_lookback: int = 12) -> pd.Series:
    """매월 말 FAA 신호를 계산해 다음 달 수익률에 적용하는 방식으로 백테스트"""
    lookback = max(momentum_lookback, vol_lookback, corr_lookback)
    all_assets = universe + ([safe_asset] if safe_asset else [])
    dates = monthly_returns.index

    port_value = [1.0]
    values = []

    for i in range(lookback, len(dates) - 1):
        weights = faa_weights(monthly_returns, i, universe, safe_asset,
                               top_n, momentum_lookback, vol_lookback, corr_lookback)
        next_return = sum(w * monthly_returns[a].iloc[i + 1] for a, w in weights.items())
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

    # 투자 후보 유니버스 + 안전자산
    universe = ["SPY", "EFA", "EEM", "TLT", "GLD", "DBC"]
    safe_asset = "SHY"
    tickers = universe + [safe_asset]

    mean_map = {"SPY": 0.09, "EFA": 0.06, "EEM": 0.07, "TLT": 0.03,
                "GLD": 0.04, "DBC": 0.03, "SHY": 0.015}
    vol_map = {"SPY": 0.18, "EFA": 0.19, "EEM": 0.24, "TLT": 0.13,
               "GLD": 0.15, "DBC": 0.18, "SHY": 0.02}

    n_months = 15 * 12  # 15년치 월간 데이터
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    monthly_returns = pd.DataFrame(
        {t: np.random.normal(mean_map[t] / 12, vol_map[t] / np.sqrt(12), n_months)
         for t in tickers},
        index=dates,
    )

    # 최근 시점 기준 FAA 비중 확인
    asof_idx = n_months - 1
    current_weights = faa_weights(monthly_returns, asof_idx, universe, safe_asset, top_n=3)

    print("=" * 55)
    print(f"기준 시점: {dates[asof_idx].date()} - FAA 비중")
    print("=" * 55)
    for a, w in current_weights.items():
        print(f"  {a}: {w:.2%}")

    # 백테스트 실행
    port_value = backtest_faa(monthly_returns, universe, safe_asset, top_n=3)
    metrics = performance_metrics(port_value)

    # 비교용: 유니버스 전체 동일비중 매수 후 보유
    lookback = 12
    buy_hold_returns = monthly_returns[universe].mean(axis=1)
    buy_hold_value = (1 + buy_hold_returns.iloc[lookback:]).cumprod()
    metrics_bh = performance_metrics(buy_hold_value)

    print("\n" + "=" * 55)
    print("백테스트 성과 비교 (15년, 월간 리밸런싱)")
    print("=" * 55)
    result = pd.DataFrame({
        "FAA": metrics,
        "동일비중 매수후보유 (비교용)": metrics_bh,
    }).T
    result["CAGR"] = result["CAGR"].map(lambda x: f"{x:.2%}")
    result["MDD"] = result["MDD"].map(lambda x: f"{x:.2%}")
    result["Sharpe"] = result["Sharpe"].map(lambda x: f"{x:.2f}")
    print(result)

    print("\n(참고: FAA는 모멘텀+변동성+상관관계 3중 필터를 쓰기 때문에")
    print(" GATT보다 정교하지만, 상관관계·변동성 계산 때문에 자산 교체가")
    print(" 더 잦아 거래비용이 늘어날 수 있습니다.)")