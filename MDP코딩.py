"""
MDP (Maximum Diversification Portfolio) 포트폴리오 전략 구현
------------------------------------------------------------
목표: 자산 간 상관관계가 낮을수록 분산효과를 극대화하는 자산 비중을 찾는다.
      "분산비율(Diversification Ratio, DR)"을 최대화하는 문제로 정의된다.

수학적 정의:
    DR(w) = (w^T · σ) / sqrt(w^T · Σ · w)

    w : 자산 비중 벡터
    σ : 자산별 개별 변동성(표준편차) 벡터
    Σ : 공분산 행렬

    분자(w^T·σ)는 "자산을 따로 봤을 때 변동성의 가중평균"이고,
    분모(sqrt(w^T·Σ·w))는 "실제 포트폴리오 변동성"이다.
    상관관계가 낮을수록 분산효과로 분모가 작아져 DR이 커진다.
    -> MDP는 이 DR을 최대화하는 w를 찾는 문제.

닫힌 형태(closed-form) 해가 없어 수치 최적화가 필요하다.
필요 라이브러리: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# MDP 비중 계산 (수치 최적화)
# ----------------------------------------------------------------------
def mdp_weights(cov: np.ndarray, vol: np.ndarray = None,
                 long_only: bool = True, max_weight: float = 1.0) -> np.ndarray:
    """
    분산비율(DR)을 최대화하는 비중을 찾는다.
    (-DR을 최소화하는 방식으로 최적화를 수행)

    cov        : 공분산 행렬
    vol        : 자산별 변동성 벡터. None이면 cov의 대각원소에서 자동 계산
    long_only  : True면 공매도 금지 (비중 >= 0)
    max_weight : 개별 자산 최대 비중 (집중 방지용)
    """
    n = cov.shape[0]
    if vol is None:
        vol = np.sqrt(np.diag(cov))

    def neg_diversification_ratio(w):
        port_vol = np.sqrt(w @ cov @ w)
        weighted_avg_vol = w @ vol
        return -weighted_avg_vol / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    lower = 0.0 if long_only else -1.0
    bounds = [(lower, max_weight)] * n
    w0 = np.ones(n) / n  # 초기값: 동일비중

    result = minimize(neg_diversification_ratio, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-16, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"최적화 실패: {result.message}")
    return result.x


def diversification_ratio(w: np.ndarray, cov: np.ndarray, vol: np.ndarray = None) -> float:
    """주어진 비중 w의 분산비율(DR)을 계산 (결과 검증용)"""
    if vol is None:
        vol = np.sqrt(np.diag(cov))
    port_vol = np.sqrt(w @ cov @ w)
    weighted_avg_vol = w @ vol
    return weighted_avg_vol / port_vol


# ----------------------------------------------------------------------
# 예시 실행
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 예시: 4개 자산 (주식, 채권, 금, 원자재)의 일간 수익률 시뮬레이션
    assets = ["주식", "채권", "금", "원자재"]
    n_assets = len(assets)
    n_days = 500

    # 실제 사용시엔 아래 부분을 실제 자산 수익률 데이터로 교체
    true_vol = np.array([0.18, 0.06, 0.15, 0.20]) / np.sqrt(252)  # 일간 변동성
    corr = np.array([
        [1.00,  -0.20, 0.10, 0.30],
        [-0.20,  1.00, 0.05, -0.10],
        [0.10,   0.05, 1.00, 0.25],
        [0.30,  -0.10, 0.25, 1.00],
    ])
    cov_daily = np.outer(true_vol, true_vol) * corr
    returns = np.random.multivariate_normal(mean=np.zeros(n_assets),
                                              cov=cov_daily, size=n_days)
    returns_df = pd.DataFrame(returns, columns=assets)

    # 연율화 공분산 행렬 및 변동성
    cov_matrix = returns_df.cov().values * 252
    volatility = np.sqrt(np.diag(cov_matrix))

    # MDP 비중 계산
    w_mdp = mdp_weights(cov_matrix, volatility)
    # 비교용: 동일비중
    w_equal = np.ones(n_assets) / n_assets

    result = pd.DataFrame({
        "MDP": w_mdp,
        "동일비중(비교용)": w_equal,
    }, index=assets)

    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print("=" * 55)
    print("MDP 포트폴리오 비중")
    print("=" * 55)
    print(result)

    print("\n" + "=" * 55)
    print("분산비율(Diversification Ratio) 비교")
    print("=" * 55)
    dr_mdp = diversification_ratio(w_mdp, cov_matrix, volatility)
    dr_equal = diversification_ratio(w_equal, cov_matrix, volatility)
    print(f"  MDP:      {dr_mdp:.3f}  (높을수록 분산효과 큼)")
    print(f"  동일비중: {dr_equal:.3f}")

    print("\n" + "=" * 55)
    print("포트폴리오 연간 변동성")
    print("=" * 55)
    for name, w in [("MDP", w_mdp), ("동일비중", w_equal)]:
        port_vol = np.sqrt(w @ cov_matrix @ w)
        print(f"  {name}: {port_vol:.2%}")