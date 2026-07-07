# -*- coding: utf-8 -*-
"""
allocation_strategies.py
=========================
14개 자산배분(Asset Allocation) 전략 알고리즘 모음

[위험 기반 (Risk-based)]
 1. GMV   - Global Minimum Variance      (최소분산)
 2. MDP   - Most Diversified Portfolio   (최대분산효과)
 3. ERC   - Equal Risk Contribution      (위험균등기여, 리스크패리티)
 4. IVOL  - Inverse Volatility           (역변동성)
 5. RB    - Custom Risk Budgeting        (위험기여도 지정배분)

[고정비중 (Static)]
 6. SIXTY_FORTY     - 60/40 주식/채권
 7. PERMANENT       - 영구 포트폴리오 (해리 브라운)
 8. ALL_WEATHER     - 올웨더 (레이 달리오, 리테일 근사버전)
 9. GOLDEN_BUTTERFLY- 황금나비
10. EQUAL_WEIGHT    - 동일비중 (1/N)

[전술적 (Tactical / Momentum)]
11. GTAA - Global Tactical Asset Allocation (Faber, 이동평균 기반)
12. FAA  - Flexible Asset Allocation (Keller & Van Putten)
13. VAA  - Vigilant Asset Allocation (Keller & Keuning)
14. DAA  - Defensive Asset Allocation (Keller & Keuning)

작성자: Claude
전제: 모든 함수는 pandas.DataFrame 형태의 "가격"(prices) 또는 "수익률"(returns)을
      입력받아 자산별 비중(weights, numpy array 또는 dict)을 반환합니다.
      실제 사용시 본인의 가격 데이터를 넣어 사용하세요 (yfinance, pykrx 등으로 수집).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================================================
# 공통 유틸리티
# ============================================================

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """가격 -> 단순 수익률"""
    return prices.pct_change().dropna(how="all")


def annualize_cov(returns: pd.DataFrame, periods_per_year: int = 252) -> np.ndarray:
    """공분산행렬 연율화"""
    return returns.cov().values * periods_per_year


def normalize(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0, None)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


# ============================================================
# 1. GMV : Global Minimum Variance (최소분산)
# ============================================================

def gmv_weights(returns: pd.DataFrame, long_only: bool = True) -> pd.Series:
    """
    목적함수: min  w' Σ w
    제약조건: sum(w) = 1, (long_only 이면 w >= 0)
    """
    cov = annualize_cov(returns)
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def obj(w):
        return w @ cov @ w

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n if long_only else [(-1, 1)] * n

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(normalize(res.x), index=returns.columns, name="GMV")


# ============================================================
# 2. MDP : Most Diversified Portfolio (최대분산효과)
# ============================================================

def mdp_weights(returns: pd.DataFrame) -> pd.Series:
    """
    분산비율(Diversification Ratio) = (w' * sigma) / sqrt(w' Σ w) 를 최대화
    sigma = 개별자산 변동성 벡터
    """
    cov = annualize_cov(returns)
    sigma = np.sqrt(np.diag(cov))
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def neg_div_ratio(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_avg_vol = w @ sigma
        return -weighted_avg_vol / port_vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n

    res = minimize(neg_div_ratio, x0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(normalize(res.x), index=returns.columns, name="MDP")


# ============================================================
# 3. ERC : Equal Risk Contribution (위험균등기여, 리스크패리티)
# ============================================================

def erc_weights(returns: pd.DataFrame) -> pd.Series:
    """
    각 자산의 위험기여도(Risk Contribution)가 동일하도록:
       RC_i = w_i * (Σw)_i / (w'Σw)  ->  모든 i에 대해 RC_i = 1/n
    """
    cov = annualize_cov(returns)
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def risk_contrib(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        return (w * marginal) / port_var  # 각 자산의 위험기여 비율

    def obj(w):
        rc = risk_contrib(w)
        target = np.ones(n) / n
        return np.sum((rc - target) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-6, 1)] * n

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return pd.Series(normalize(res.x), index=returns.columns, name="ERC")


# ============================================================
# 4. IVOL : Inverse Volatility (역변동성)
# ============================================================

def inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """w_i = (1/sigma_i) / sum(1/sigma_j)  (자산간 상관관계는 무시하는 단순버전 리스크패리티)"""
    vol = returns.std() * np.sqrt(252)
    inv = 1 / vol
    w = inv / inv.sum()
    w.name = "InverseVol"
    return w


# ============================================================
# 5. RB : Custom Risk Budgeting (위험기여도 지정배분)
# ============================================================

def risk_budget_weights(returns: pd.DataFrame, budgets: dict) -> pd.Series:
    """
    ERC의 일반화 버전. 각 자산에 원하는 위험기여 비율(budgets, 합=1)을 지정.
    예: budgets = {"주식": 0.5, "채권": 0.3, "금": 0.2}
    """
    cols = list(returns.columns)
    b = np.array([budgets[c] for c in cols])
    b = b / b.sum()
    cov = annualize_cov(returns)
    n = cov.shape[0]
    x0 = np.ones(n) / n

    def obj(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        rc = (w * marginal) / port_var
        return np.sum((rc - b) ** 2)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(1e-6, 1)] * n

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    return pd.Series(normalize(res.x), index=cols, name="RiskBudget")


# ============================================================
# 6. 60/40 (고정비중)
# ============================================================

def sixty_forty_weights(stock_col: str, bond_col: str) -> pd.Series:
    return pd.Series({stock_col: 0.60, bond_col: 0.40}, name="60/40")


# ============================================================
# 7. 영구 포트폴리오 (Permanent Portfolio, Harry Browne)
# ============================================================

def permanent_portfolio_weights(stock_col: str, bond_long_col: str,
                                 cash_col: str, gold_col: str) -> pd.Series:
    """주식 25% / 장기채 25% / 현금(단기채) 25% / 금 25%"""
    return pd.Series({
        stock_col: 0.25, bond_long_col: 0.25,
        cash_col: 0.25, gold_col: 0.25
    }, name="Permanent")


# ============================================================
# 8. 올웨더 (All Weather, Ray Dalio 리테일 근사버전)
# ============================================================

def all_weather_weights(stock_col: str, bond_long_col: str, bond_mid_col: str,
                         gold_col: str, commodity_col: str) -> pd.Series:
    """주식 30% / 장기채 40% / 중기채 15% / 금 7.5% / 원자재 7.5%"""
    return pd.Series({
        stock_col: 0.30, bond_long_col: 0.40, bond_mid_col: 0.15,
        gold_col: 0.075, commodity_col: 0.075
    }, name="AllWeather")


# ============================================================
# 9. 황금나비 (Golden Butterfly)
# ============================================================

def golden_butterfly_weights(total_stock_col: str, small_value_col: str,
                              bond_long_col: str, bond_short_col: str,
                              gold_col: str) -> pd.Series:
    """전체주식20 / 소형가치주20 / 장기채20 / 단기채20 / 금20"""
    return pd.Series({
        total_stock_col: 0.20, small_value_col: 0.20,
        bond_long_col: 0.20, bond_short_col: 0.20, gold_col: 0.20
    }, name="GoldenButterfly")


# ============================================================
# 10. 동일비중 (Equal Weight, 1/N)
# ============================================================

def equal_weight_weights(assets: list) -> pd.Series:
    n = len(assets)
    return pd.Series({a: 1 / n for a in assets}, name="EqualWeight")


# ============================================================
# 11. GTAA : Global Tactical Asset Allocation (Mebane Faber)
# ============================================================

def gtaa_weights(prices: pd.DataFrame, lookback_months: int = 10,
                 cash_col: str = None) -> pd.Series:
    """
    Faber(2007) 방식: 각 자산 가격이 자신의 N개월(기본10) 이동평균선 위에 있으면 보유,
    아래면 현금(cash_col)으로 대체. 편입 자산들은 동일비중.
    prices: 월말 가격 데이터(DatetimeIndex) 필요
    """
    monthly = prices.resample("ME").last()
    sma = monthly.rolling(lookback_months).mean()
    last_price = monthly.iloc[-1]
    last_sma = sma.iloc[-1]

    signal = (last_price > last_sma).astype(float)  # 1: 보유, 0: 회피
    n_assets = len(signal)
    risky_w = signal / n_assets  # 편입자산 균등비중

    cash_alloc = 1 - risky_w.sum()
    w = risky_w.copy()
    if cash_col is not None:
        w[cash_col] = w.get(cash_col, 0) + cash_alloc
    else:
        # 현금 자산이 없으면 편입자산에 재분배
        w = normalize(w.values)
        w = pd.Series(w, index=signal.index)
    w.name = "GTAA"
    return w


# ============================================================
# 12. FAA : Flexible Asset Allocation (Keller & Van Putten, 2012)
# ============================================================

def _momentum_score(prices_monthly: pd.DataFrame, lookback: int = 4) -> pd.Series:
    """단순 모멘텀: (P0 / P_lookback) - 1"""
    return prices_monthly.iloc[-1] / prices_monthly.iloc[-1 - lookback] - 1


def faa_weights(prices: pd.DataFrame, top_n: int = 3, lookback: int = 4,
                cash_col: str = None, vol_lookback: int = 4,
                weights_RVC=(1.0, 0.5, 0.5)) -> pd.Series:
    """
    FAA: 자산을 아래 3개 순위(랭크)의 가중합으로 정렬해 상위 top_n개만 편입
      R(Return)   : 최근 lookback개월 모멘텀 (높을수록 좋음, 랭크 오름차순=낮은순위값이 좋음)
      V(Volatility): 최근 변동성 (낮을수록 좋음)
      C(Correlation): 다른 자산과의 평균상관관계 (낮을수록 좋음, 분산효과)
    편입자산은 동일비중, 모멘텀이 음수인 자산은 강제로 현금 편입(선택)
    """
    monthly = prices.resample("ME").last()
    rets_monthly = monthly.pct_change().dropna()

    mom = _momentum_score(monthly, lookback)
    vol = rets_monthly.iloc[-vol_lookback:].std()
    corr = rets_monthly.iloc[-vol_lookback:].corr().mean()  # 자산별 평균상관계수

    # 랭크: 값이 좋을수록(모멘텀 높음/변동성 낮음/상관 낮음) 순위 1
    rank_R = mom.rank(ascending=False)
    rank_V = vol.rank(ascending=True)
    rank_C = corr.rank(ascending=True)

    wR, wV, wC = weights_RVC
    combined_rank = wR * rank_R + wV * rank_V + wC * rank_C
    combined_rank = combined_rank.sort_values()

    selected = combined_rank.index[:top_n]
    # 모멘텀이 음수인 종목은 제외(현금으로 대체)
    selected = [a for a in selected if mom[a] > 0]

    w = pd.Series(0.0, index=prices.columns)
    if len(selected) > 0:
        for a in selected:
            w[a] = 1 / len(selected) if cash_col is None else (1 / top_n)
    cash_alloc = 1 - w.sum()
    if cash_col is not None:
        w[cash_col] = w.get(cash_col, 0) + cash_alloc
    w.name = "FAA"
    return w


# ============================================================
# 13. VAA : Vigilant Asset Allocation (Keller & Keuning, 2016)
# ============================================================

def _vaa_momentum(monthly: pd.DataFrame) -> pd.Series:
    """VAA 공식 모멘텀 점수: 13612W
       score = 12*(P0/P1 -1) + 4*(P0/P3 -1) + 2*(P0/P6 -1) + 1*(P0/P12 -1)
    """
    p0 = monthly.iloc[-1]
    p1 = monthly.iloc[-2]
    p3 = monthly.iloc[-4]
    p6 = monthly.iloc[-7]
    p12 = monthly.iloc[-13]
    return 12 * (p0 / p1 - 1) + 4 * (p0 / p3 - 1) + 2 * (p0 / p6 - 1) + 1 * (p0 / p12 - 1)


def vaa_weights(prices: pd.DataFrame, offensive_assets: list, canary_assets: list,
               safe_assets: list, top_n: int = 5) -> pd.Series:
    """
    VAA-G12/G4 방식:
      1) canary_assets(카나리아, 보통 신흥국/선진국 주식 등 위험신호 자산) 모멘텀 계산
      2) canary 중 하나라도 모멘텀<=0 이면 "위기신호" -> safe_assets 중 모멘텀 1위 자산에 100% 투자
         (canary 중 음수 개수 비율만큼 안전자산, 나머지는 공격자산 상위 배분 - 여기서는 단순화하여
          "하나라도 음수면 전액 안전자산" 룰 사용. breadth 버전은 아래 daa_weights 참고)
      3) 위기신호 없으면 offensive_assets 중 모멘텀 상위 top_n 자산에 동일비중 투자
    """
    monthly = prices.resample("ME").last()
    mom = _vaa_momentum(monthly)

    canary_mom = mom[canary_assets]
    n_bad = (canary_mom <= 0).sum()

    w = pd.Series(0.0, index=prices.columns)

    if n_bad > 0:
        # 안전자산 중 모멘텀 최상위 1개에 전액
        best_safe = mom[safe_assets].idxmax()
        w[best_safe] = 1.0
    else:
        off_mom = mom[offensive_assets].sort_values(ascending=False)
        selected = off_mom.index[:top_n]
        for a in selected:
            w[a] = 1 / top_n
    w.name = "VAA"
    return w


# ============================================================
# 14. DAA : Defensive Asset Allocation (Keller & Keuning, 2017)
# ============================================================

def daa_weights(prices: pd.DataFrame, offensive_assets: list, canary_assets: list,
                defensive_assets: list, top_n_off: int = 6, top_n_def: int = 1,
                breadth_T: int = None) -> pd.Series:
    """
    DAA(Defensive Asset Allocation): VAA의 확장판으로 "breadth(폭)" 개념 사용.
      canary_assets 중 음수 모멘텀 개수 = n_bad (0 ~ len(canary))
      breadth_T: canary 자산 개수(기본: len(canary_assets))
      안전자산 비중 = n_bad / breadth_T  (나머지는 공격자산 상위 top_n_off개 균등비중)
      안전자산은 defensive_assets 중 모멘텀 상위 top_n_def개에 균등배분
    """
    monthly = prices.resample("ME").last()
    mom = _vaa_momentum(monthly)

    if breadth_T is None:
        breadth_T = len(canary_assets)

    canary_mom = mom[canary_assets]
    n_bad = (canary_mom <= 0).sum()
    def_fraction = n_bad / breadth_T
    off_fraction = 1 - def_fraction

    w = pd.Series(0.0, index=prices.columns)

    # 방어자산 배분
    if def_fraction > 0:
        def_mom = mom[defensive_assets].sort_values(ascending=False)
        def_selected = def_mom.index[:top_n_def]
        for a in def_selected:
            w[a] += def_fraction / len(def_selected)

    # 공격자산 배분
    if off_fraction > 0:
        off_mom = mom[offensive_assets].sort_values(ascending=False)
        off_selected = off_mom.index[:top_n_off]
        # 모멘텀이 음수인 공격자산은 제외하고 남은 비중은 방어자산으로 재배분(단순화 버전)
        off_selected_pos = [a for a in off_selected if mom[a] > 0]
        if len(off_selected_pos) > 0:
            for a in off_selected_pos:
                w[a] += off_fraction / len(off_selected_pos)
        else:
            best_def = mom[defensive_assets].idxmax()
            w[best_def] += off_fraction

    w.name = "DAA"
    return w


# ============================================================
# 데모: 합성 데이터로 14개 전략 전부 실행해보기
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- 합성 가격데이터 생성 (실사용시 실제 가격으로 교체) ----
    n_days = 252 * 6  # 6년치 일간 데이터
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    assets = ["주식", "소형가치주", "장기채", "중기채", "단기채/현금",
              "금", "원자재", "신흥국주식", "선진국주식"]
    mu = np.array([0.0004, 0.00045, 0.0001, 0.00008, 0.00002,
                   0.00015, 0.0001, 0.00035, 0.0003])
    sigma = np.array([0.011, 0.014, 0.006, 0.004, 0.0008,
                      0.009, 0.012, 0.013, 0.011])
    corr = np.eye(len(assets)) * 0 + 0.3
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sigma, sigma) * corr
    daily_rets = np.random.multivariate_normal(mu, cov, size=n_days)
    prices = pd.DataFrame(100 * np.cumprod(1 + daily_rets, axis=0),
                          index=dates, columns=assets)

    returns = to_returns(prices)

    print("=" * 60)
    print("[1] GMV")
    print(gmv_weights(returns).round(3))

    print("\n[2] MDP")
    print(mdp_weights(returns).round(3))

    print("\n[3] ERC (리스크패리티)")
    print(erc_weights(returns).round(3))

    print("\n[4] 역변동성(Inverse Vol)")
    print(inverse_vol_weights(returns).round(3))

    print("\n[5] 위험기여도 지정배분 (주식50/채권30/금20)")
    budgets = {"주식": 0.5, "소형가치주": 0.0, "장기채": 0.15, "중기채": 0.15,
              "단기채/현금": 0.0, "금": 0.2, "원자재": 0.0,
              "신흥국주식": 0.0, "선진국주식": 0.0}
    # 0인 자산은 최소 비중으로 처리하기 위해 제외 후 계산 예시
    rb_assets = ["주식", "장기채", "중기채", "금"]
    rb_budgets = {"주식": 0.5, "장기채": 0.15, "중기채": 0.15, "금": 0.2}
    print(risk_budget_weights(returns[rb_assets], rb_budgets).round(3))

    print("\n[6] 60/40")
    print(sixty_forty_weights("주식", "장기채"))

    print("\n[7] 영구 포트폴리오")
    print(permanent_portfolio_weights("주식", "장기채", "단기채/현금", "금"))

    print("\n[8] 올웨더")
    print(all_weather_weights("주식", "장기채", "중기채", "금", "원자재"))

    print("\n[9] 황금나비")
    print(golden_butterfly_weights("주식", "소형가치주", "장기채", "단기채/현금", "금"))

    print("\n[10] 동일비중")
    print(equal_weight_weights(assets))

    print("\n[11] GTAA")
    print(gtaa_weights(prices, lookback_months=10, cash_col="단기채/현금").round(3))

    print("\n[12] FAA")
    print(faa_weights(prices, top_n=3, cash_col="단기채/현금").round(3))

    print("\n[13] VAA")
    offensive = ["주식", "신흥국주식", "선진국주식", "장기채", "금"]
    canary = ["신흥국주식", "선진국주식"]
    safe = ["장기채", "단기채/현금"]
    print(vaa_weights(prices, offensive, canary, safe, top_n=3).round(3))

    print("\n[14] DAA")
    off = ["주식", "신흥국주식", "선진국주식", "장기채", "금", "원자재"]
    defensiv = ["단기채/현금", "장기채"]
    print(daa_weights(prices, off, canary, defensiv,
                      top_n_off=3, top_n_def=1).round(3))

    print("=" * 60)
    print("완료: 14개 전략 계산 성공")
