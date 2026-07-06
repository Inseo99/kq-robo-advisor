"""
포트폴리오 리스크 기여도(Risk Contribution) 계산
=================================================

개념 설명
---------
포트폴리오 변동성(위험)은 단순히 개별 자산 위험의 합이 아니라,
자산 간 상관관계(공분산)를 고려해서 계산됩니다.

- 포트폴리오 분산:      sigma_p^2 = w' * Sigma * w
- 포트폴리오 변동성:    sigma_p   = sqrt(w' * Sigma * w)

- 한계기여도 (MCTR, Marginal Contribution to Risk):
    자산 i의 비중을 아주 조금 늘렸을 때 포트폴리오 위험이 얼마나 증가하는지
    MCTR_i = (Sigma * w)_i / sigma_p

- 기여도 (CTR, Component Contribution to Risk):
    자산 i가 실제로 포트폴리오 위험에 기여하는 절대적인 양
    CTR_i = w_i * MCTR_i
    (모든 자산의 CTR을 더하면 정확히 sigma_p 가 됩니다 - 오일러 정리)

- 비중기여도 (%CTR, Percentage Contribution to Risk):
    PCTR_i = CTR_i / sigma_p
    (모든 자산의 PCTR 합 = 100%)

필요 라이브러리: numpy, pandas (선택)
"""

import numpy as np
import pandas as pd


def risk_contribution(weights, cov_matrix, asset_names=None, annualize=None):
    """
    포트폴리오 자산별 위험기여도를 계산합니다.

    Parameters
    ----------
    weights : array-like, shape (n,)
        자산별 포트폴리오 비중 (합이 1일 필요는 없지만 보통 1로 정규화)
    cov_matrix : array-like, shape (n, n)
        자산 수익률의 공분산 행렬
    asset_names : list of str, optional
        자산 이름 (결과를 보기 좋게 표시하기 위함)
    annualize : int, optional
        일별 등 단기 데이터를 연율화할 때 곱할 기간 수 (예: 252 거래일)
        None이면 연율화하지 않음

    Returns
    -------
    pandas.DataFrame
        columns = [Weight, MCTR, CTR, PCTR(%)]
        마지막 행은 합계(Total)이며 Total CTR = 포트폴리오 변동성과 동일
    """
    w = np.asarray(weights, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)

    n = len(w)
    if Sigma.shape != (n, n):
        raise ValueError(f"cov_matrix는 {n}x{n} 행렬이어야 합니다. 현재: {Sigma.shape}")

    # 연율화 옵션 (일별 공분산 -> 연간 공분산)
    scale = annualize if annualize else 1

    port_variance = w @ Sigma @ w * scale
    port_vol = np.sqrt(port_variance)

    if port_vol == 0:
        raise ValueError("포트폴리오 변동성이 0입니다. 비중 또는 공분산 행렬을 확인하세요.")

    # 핵심 계산
    marginal = (Sigma @ w) * scale / port_vol          # MCTR
    component = w * marginal                            # CTR
    percent = component / port_vol * 100                 # PCTR (%)

    if asset_names is None:
        asset_names = [f"Asset_{i+1}" for i in range(n)]

    df = pd.DataFrame({
        "Weight": w,
        "MCTR": marginal,
        "CTR": component,
        "PCTR(%)": percent,
    }, index=asset_names)

    # 합계 행 추가 (검증용: CTR 합 = 포트폴리오 변동성, PCTR 합 = 100%)
    total_row = pd.DataFrame({
        "Weight": [w.sum()],
        "MCTR": [np.nan],
        "CTR": [component.sum()],
        "PCTR(%)": [percent.sum()],
    }, index=["Total (Portfolio Vol)"])

    result = pd.concat([df, total_row])
    result.attrs["portfolio_volatility"] = port_vol
    return result


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 예시: 4개 자산으로 구성된 포트폴리오
    # ------------------------------------------------------------
    asset_names = ["국내주식", "해외주식", "채권", "원자재"]

    # 자산별 포트폴리오 비중 (합계 1.0)
    weights = [0.40, 0.30, 0.20, 0.10]

    # 자산별 연간 변동성 (예시 값)
    vol = np.array([0.20, 0.18, 0.05, 0.25])

    # 자산 간 상관계수 행렬 (예시 값)
    corr = np.array([
        [1.00, 0.70, -0.20, 0.30],
        [0.70, 1.00, -0.10, 0.20],
        [-0.20, -0.10, 1.00, 0.00],
        [0.30, 0.20, 0.00, 1.00],
    ])

    # 상관계수 -> 공분산 행렬 변환 (Sigma_ij = corr_ij * vol_i * vol_j)
    cov_matrix = np.outer(vol, vol) * corr

    result = risk_contribution(weights, cov_matrix, asset_names=asset_names)

    print("=" * 55)
    print("포트폴리오 리스크 기여도 분석 결과")
    print("=" * 55)
    print(result.round(4))
    print("-" * 55)
    print(f"포트폴리오 연간 변동성: {result.attrs['portfolio_volatility']:.2%}")
    print("=" * 55)