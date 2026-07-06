"""
ERC (Equal Risk Contribution) 포트폴리오 전략 구현
------------------------------------------------------
목표: 모든 자산이 포트폴리오 전체 위험에 '동일하게' 기여하도록 비중을 조정한다.
      (Risk Parity의 가장 기본적인 형태)

수학적 정의:
    자산 i의 위험기여도(Risk Contribution):
        RC_i = w_i · (Σw)_i

    ERC는 모든 자산의 RC_i가 서로 같아지도록(= 포트폴리오 전체 위험을 균등분배)
    비중 w를 최적화하는 문제이다.

    참고) RC_i를 모두 더하면 포트폴리오 분산(w^T Σ w)이 되므로,
          "정규화된 위험기여도" RC_i / (포트폴리오 분산) 의 합은 항상 1이고,
          ERC의 목표는 이 값들을 모두 1/n로 맞추는 것과 같다.

닫힌 형태(closed-form) 해가 없어 수치 최적화가 필요하다.
필요 라이브러리: numpy, pandas, scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ----------------------------------------------------------------------
# ERC 비중 계산 (수치 최적화)
# ----------------------------------------------------------------------
def erc_weights(cov: np.ndarray, long_only: bool = True,
                 max_weight: float = 1.0) -> np.ndarray:
    """
    모든 자산의 정규화된 위험기여도가 1/n로 동일해지도록 비중을 최적화한다.
    목적함수: (자산별 위험기여도 - 목표 1/n)의 제곱합을 최소화

    cov        : 공분산 행렬
    long_only  : True면 공매도 금지 (비중 >= 0)
    max_weight : 개별 자산 최대 비중 (집중 방지용)
    """
    n = cov.shape[0]
    target = np.ones(n) / n  # 목표: 모든 자산이 위험의 1/n씩 기여

    def risk_contributions(w):
        port_var = w @ cov @ w
        marginal_contrib = cov @ w
        rc = w * marginal_contrib
        return rc / port_var  # 정규화된 위험기여도 (합 = 1)

    def objective(w):
        rc = risk_contributions(w)
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    lower = 1e-6 if long_only else -1.0  # 0으로 두면 나눗셈(위험기여도) 불안정 -> 아주 작은 양수
    bounds = [(lower, max_weight)] * n
    w0 = np.ones(n) / n  # 초기값: 동일비중

    result = minimize(objective, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-16, "maxiter": 1000})
    if not result.success:
        raise RuntimeError(f"최적화 실패: {result.message}")
    return result.x


def compute_risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """주어진 비중 w의 자산별 위험기여도(정규화, 합=1)를 계산 (결과 검증용)"""
    port_var = w @ cov @ w
    marginal_contrib = cov @ w
    rc = w * marginal_contrib
    return rc / port_var


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

    # ERC 비중 계산
    w_erc = erc_weights(cov_matrix)
    # 비교용: 동일비중
    w_equal = np.ones(n_assets) / n_assets

    result = pd.DataFrame({
        "ERC": w_erc,
        "동일비중(비교용)": w_equal,
    }, index=assets)

    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print("=" * 55)
    print("ERC 포트폴리오 비중")
    print("=" * 55)
    print(result)

    print("\n" + "=" * 55)
    print("자산별 위험기여도 검증 (ERC는 모두 동일해야 함)")
    print("=" * 55)
    rc_erc = compute_risk_contributions(w_erc, cov_matrix)
    rc_equal = compute_risk_contributions(w_equal, cov_matrix)
    check = pd.DataFrame({
        "ERC 위험기여도": rc_erc,
        "동일비중 위험기여도": rc_equal,
    }, index=assets)
    print(check)

    print("\n" + "=" * 55)
    print("포트폴리오 연간 변동성")
    print("=" * 55)
    for name, w in [("ERC", w_erc), ("동일비중", w_equal)]:
        port_vol = np.sqrt(w @ cov_matrix @ w)
        print(f"  {name}: {port_vol:.2%}")

    print("\n(참고: 동일비중은 '금액'을 균등하게 나눈 것이고,")
    print(" ERC는 '위험'을 균등하게 나눈 것이라 서로 다릅니다.")
    print(" 동일비중 위험기여도를 보면 변동성 높은 자산에 위험이 쏠려있습니다.)")