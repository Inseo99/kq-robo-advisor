"""
GMV (Global Minimum Variance) 포트폴리오 전략 구현
------------------------------------------------------
목표: 기대수익률은 고려하지 않고, 포트폴리오 전체 분산(변동성)을 최소화하는
      자산 비중을 찾는다.

수학적 공식 (공매도 허용시 해석적 해):
    w_GMV = (Σ^-1 · 1) / (1^T · Σ^-1 · 1)

    Σ    : 자산 수익률의 공분산 행렬
    Σ^-1 : 공분산 행렬의 역행렬
    1    : 모든 원소가 1인 벡터

실전에서는 대부분 공매도를 금지(비중 >= 0)하므로, 제약조건이 있는
수치 최적화(scipy.optimize) 방식도 함께 구현합니다.

필요 라이브러리: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# 방법 1) 해석적 공식 (공매도 허용, closed-form)
# ----------------------------------------------------------------------
def gmv_weights_analytic(cov: np.ndarray) -> np.ndarray:
    """
    w_GMV = (Σ^-1 · 1) / (1^T · Σ^-1 · 1)
    공매도를 허용하는 순수 수학적 해. 계산이 빠르고 항상 유일해를 가짐.
    """
    n = cov.shape[0]
    ones = np.ones(n)
    inv_cov = np.linalg.inv(cov)
    w = inv_cov @ ones / (ones @ inv_cov @ ones)
    return w


# ----------------------------------------------------------------------
# 방법 2) 제약조건부 수치 최적화 (공매도 금지 등 실전 제약 반영)
# ----------------------------------------------------------------------
def gmv_weights_constrained(cov: np.ndarray,
                             long_only: bool = True,
                             max_weight: float = 1.0) -> np.ndarray:
    """
    포트폴리오 분산(w^T Σ w)을 최소화하되,
      - 비중 합 = 1
      - long_only=True 이면 각 비중 >= 0 (공매도 금지)
      - max_weight 로 개별 자산 최대 비중 제한 가능 (집중투자 방지)
    """
    n = cov.shape[0]

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    lower = 0.0 if long_only else -1.0
    bounds = [(lower, max_weight)] * n
    w0 = np.ones(n) / n  # 초기값: 동일비중에서 출발

    result = minimize(objective, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-16, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"최적화 실패: {result.message}")
    return result.x


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

    # 연율화 공분산 행렬
    cov_matrix = returns_df.cov().values * 252

    # 두 가지 방식으로 GMV 비중 계산
    w_analytic = gmv_weights_analytic(cov_matrix)
    w_constrained = gmv_weights_constrained(cov_matrix, long_only=True)

    result = pd.DataFrame({
        "GMV (해석적, 공매도 허용)": w_analytic,
        "GMV (공매도 금지)": w_constrained,
    }, index=assets)

    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print("=" * 55)
    print("GMV 포트폴리오 비중 비교")
    print("=" * 55)
    print(result)

    print("\n" + "=" * 55)
    print("포트폴리오 연간 변동성")
    print("=" * 55)
    for name, w in [("해석적", w_analytic), ("공매도 금지", w_constrained)]:
        port_vol = np.sqrt(w @ cov_matrix @ w)
        print(f"  {name}: {port_vol:.2%}")

    print("\n(참고: 이 예시엔 공매도가 필요한 음수 비중이 나오지 않아")
    print(" 두 결과가 거의 동일합니다. 자산 상관관계가 매우 높으면")
    print(" 해석적 해에서 음수(공매도) 비중이 나타날 수 있습니다.)")