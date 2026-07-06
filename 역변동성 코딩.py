"""
역변동성 (Inverse Volatility) 포트폴리오 전략 구현
------------------------------------------------------
목표: 변동성이 낮은 자산에는 더 큰 비중을, 변동성이 높은 자산에는
      더 작은 비중을 배분한다. 위험기반 전략 중 계산이 가장 단순하다.

수학적 공식:
    w_i = (1/σ_i) / Σ_j(1/σ_j)

    σ_i : 자산 i의 변동성(표준편차)

특징: ERC/GMV/MDP와 달리 공분산 행렬(자산 간 상관관계)을 전혀 사용하지
      않고, 오직 개별 자산의 변동성만으로 비중을 계산한다.
      -> 최적화 과정이 필요 없어 계산이 즉시 끝나고 이해하기 쉽다.
      -> 대신 자산 간 상관관계가 반영되지 않는다는 한계가 있다.

필요 라이브러리: numpy, pandas
"""

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 역변동성 비중 계산 (해석적 공식, 최적화 불필요)
# ----------------------------------------------------------------------
def inverse_vol_weights(vol: np.ndarray) -> np.ndarray:
    """
    w_i = (1/σ_i) / Σ(1/σ_j)

    vol : 자산별 변동성(표준편차) 벡터
    """
    inv_vol = 1 / vol
    return inv_vol / np.sum(inv_vol)


def inverse_vol_weights_from_returns(returns_df: pd.DataFrame,
                                      annualize_factor: int = 252) -> pd.Series:
    """
    수익률 데이터에서 바로 역변동성 비중을 계산하는 편의 함수.
    returns_df: 컬럼 = 자산, 행 = 기간별 수익률
    """
    vol = returns_df.std().values * np.sqrt(annualize_factor)  # 연율화 변동성
    weights = inverse_vol_weights(vol)
    return pd.Series(weights, index=returns_df.columns)


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

    # 방법 1) 변동성 벡터로 직접 계산
    volatility = returns_df.std().values * np.sqrt(252)  # 연율화
    w_ivol = inverse_vol_weights(volatility)

    # 방법 2) 수익률 데이터에서 바로 계산 (편의 함수)
    w_ivol_series = inverse_vol_weights_from_returns(returns_df)

    result = pd.DataFrame({
        "연간 변동성": volatility,
        "역변동성 비중": w_ivol,
    }, index=assets)

    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print("=" * 55)
    print("역변동성 포트폴리오 비중")
    print("=" * 55)
    print(result)

    print("\n" + "=" * 55)
    print("포트폴리오 연간 변동성 (참고: 상관관계 미고려)")
    print("=" * 55)
    cov_matrix = returns_df.cov().values * 252
    port_vol = np.sqrt(w_ivol @ cov_matrix @ w_ivol)
    print(f"  역변동성 포트폴리오: {port_vol:.2%}")

    print("\n(참고: 변동성이 가장 낮은 채권이 가장 큰 비중을 가져갑니다.")
    print(" ERC/GMV와 방향성은 비슷하지만, 상관관계를 고려하지 않으므로")
    print(" 계산된 비중 값 자체는 다릅니다.)")