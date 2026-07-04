"""
validation_permutation.py — Permutation Test (라벨 셔플 1,000회)

목적: 신호의 정보 가치 통계 검정
원리:
  - 실제 전략은 종목 5개를 선정함
  - 그 5개를 무작위 5개로 바꿔서 1,000번 백테스트
  - 실제 결과와 셔플 결과 비교
  - 셔플 분포에서 실제 결과의 위치(percentile) → p-value

Placebo와 차이:
  - Placebo: 매월 완전 무작위 5종목 선정
  - Permutation: 실제 전략이 고른 종목과 같은 수(5)를 무작위로
  - 둘 다 본질적으로 비슷하지만 Permutation은 학술 표준 방법

p-value 해석:
  - p < 0.05: 실제 전략이 무작위 대비 통계적 유의
  - p > 0.05: 무작위와 차이 없음
"""

import os, sys, time
os.environ.setdefault('KQ_DISABLE_TABPFN', '1')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation_core import prepare_bt_data, run_bt_core


def run_permutation_test(data, strategy='quant', n_iter=1000, top_n=5,
                         rebalance='M', seed=42):
    """
    Permutation Test 본체

    절차:
      1. 실제 전략 백테스트 → actual_cagr
      2. 종목 라벨 셔플 후 백테스트 N회 → null distribution
      3. p-value = P(셔플 결과 >= 실제 결과)

    벡터화: 월별 수익률 행렬 사용
    """
    price_df = data['price_df']

    # 월말 수익률 행렬
    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance, 'ME')
    monthly = price_df.resample(rule).last()
    monthly_ret = monthly.pct_change().fillna(0)
    ret_matrix = monthly_ret.values
    n_months, n_tickers = ret_matrix.shape

    valid_mask = (~monthly.isna()).values
    rng = np.random.default_rng(seed)

    # PiT 모드: 매월 시총 상위 N 미리 계산
    pit_mode = data.get('point_in_time', False)
    pit_top_n = data.get('top_n_mcap', 200)
    mcap_hist = data.get('mcap_history')

    pit_valid_per_month = None
    if pit_mode and mcap_hist is not None:
        from validation_core import get_top_mcap_at
        col_to_idx = {c: i for i, c in enumerate(price_df.columns)}
        pit_valid_per_month = []
        for m_date in monthly.index:
            top_tickers = get_top_mcap_at(mcap_hist, m_date, pit_top_n)
            valid_indices = np.array([col_to_idx[t] for t in top_tickers
                                       if t in col_to_idx])
            pit_valid_per_month.append(valid_indices)

    print(f"  Permutation 시뮬레이션 ({n_iter}회)")
    print(f"  월 수: {n_months}, 종목 수: {n_tickers}")
    if pit_mode:
        print(f"  Point-in-Time 모드 적용 (매월 시총 상위 {pit_top_n})")
    t0 = time.time()

    cagrs = np.zeros(n_iter)
    sharpes = np.zeros(n_iter)
    years = (n_months - 1) / 12

    for i in range(n_iter):
        equity = np.ones(n_months)
        for m in range(1, n_months):
            # PiT 모드: 그 시점 시총 상위 N개 풀에서만 선정
            if pit_valid_per_month is not None:
                pit_idx = pit_valid_per_month[m-1]
                mask_idx = valid_mask[m-1]
                valid_idx = pit_idx[mask_idx[pit_idx]] if len(pit_idx) > 0 else np.array([])
            else:
                valid_idx = np.where(valid_mask[m-1])[0]

            if len(valid_idx) < top_n:
                equity[m] = equity[m-1]
                continue
            # Permutation: 무작위 5개 (라벨 셔플 = 사실상 무작위 선정과 동일)
            selected = rng.choice(valid_idx, size=top_n, replace=False)
            month_rets = ret_matrix[m, selected]
            month_rets = month_rets[np.isfinite(month_rets)]
            if len(month_rets) == 0:
                equity[m] = equity[m-1]
                continue
            equity[m] = equity[m-1] * (1 + month_rets.mean())

        total = equity[-1]
        if years > 0 and total > 0:
            cagrs[i] = (total**(1/years) - 1) * 100
        rets = np.diff(equity) / equity[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 0 and rets.std() > 0:
            sharpes[i] = (cagrs[i] / 100 - 0.03) / (rets.std() * np.sqrt(12))

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  진행: {i+1}/{n_iter} ({elapsed:.1f}초)")

    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.1f}초")

    return dict(cagrs=cagrs, sharpes=sharpes, n_iter=n_iter)


def compute_p_value(actual, null_dist, two_sided=False):
    """p-value 계산

    Args:
        actual: 실제 전략 값
        null_dist: 셔플 분포
        two_sided: 양측 검정 (절대값 기준)

    Returns:
        p-value (0~1)
    """
    if two_sided:
        # 양측: |actual|보다 큰 |null| 비율
        return float((np.abs(null_dist) >= np.abs(actual)).mean())
    else:
        # 단측: actual보다 큰 null 비율 (실제가 더 좋은지)
        return float((null_dist >= actual).mean())


def run_full_permutation(strategy='quant', n_iter=1000):
    """전체 Permutation Test 파이프라인"""
    print("=" * 60)
    print(f"Permutation Test ({strategy} 전략, {n_iter}회)")
    print("=" * 60)

    import server
    universe = server.UNIVERSE
    ticker_to_code = server._ticker_to_code

    print("\n[1/3] 데이터 준비 (시총 상위 200)")
    t0 = time.time()
    data = prepare_bt_data(
        universe, ticker_to_code,
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  완료: {time.time()-t0:.1f}초")

    print(f"\n[2/3] 실제 전략 백테스트 ({strategy})")
    t0 = time.time()
    actual = run_bt_core(data, strategy=strategy, top_n=5)
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  CAGR: {actual.get('cagr')}%, Sharpe: {actual.get('sharpe')}")

    print(f"\n[3/3] 라벨 셔플 시뮬레이션 {n_iter}회")
    perm = run_permutation_test(data, strategy=strategy, n_iter=n_iter)

    actual_cagr = actual.get('cagr', 0)
    actual_sharpe = actual.get('sharpe', 0)

    # p-value 계산
    # 우리 전략이 무작위보다 좋은지: 단측 검정
    # H0: 우리 전략 = 무작위 (알파 없음)
    # H1: 우리 전략 > 무작위 (알파 있음)
    p_cagr_upper = compute_p_value(actual_cagr, perm['cagrs'])
    # 우리 전략이 무작위보다 나쁜지
    p_cagr_lower = compute_p_value(-actual_cagr, -perm['cagrs'])
    # 양측 (절대값 기준)
    p_cagr_two = compute_p_value(actual_cagr - perm['cagrs'].mean(),
                                   perm['cagrs'] - perm['cagrs'].mean(),
                                   two_sided=True)

    p_sharpe_upper = compute_p_value(actual_sharpe, perm['sharpes'])

    print("\n" + "=" * 60)
    print("Permutation Test 결과")
    print("=" * 60)

    print(f"\n실제 전략 ({strategy}):")
    print(f"  CAGR: {actual_cagr:.2f}%")
    print(f"  Sharpe: {actual_sharpe:.3f}")

    print(f"\n셔플 분포 (N={n_iter}):")
    print(f"  CAGR 평균: {perm['cagrs'].mean():.2f}% (±{perm['cagrs'].std():.2f})")
    print(f"  Sharpe 평균: {perm['sharpes'].mean():.3f}")

    print(f"\np-value:")
    print(f"  H1: 실제 > 무작위 (알파 있나?): p = {p_cagr_upper:.3f}")
    print(f"  H1: 실제 < 무작위 (마이너스 알파?): p = {p_cagr_lower:.3f}")
    print(f"  양측 검정 (어느 방향이든 차이 있나?): p = {p_cagr_two:.3f}")

    # 결과 해석
    print(f"\n결론:")
    alpha = 0.05
    if p_cagr_upper < alpha:
        print(f"  ✅ 실제 전략이 무작위보다 통계적 유의하게 좋음 (p < {alpha})")
        print(f"     → 진짜 알파 발견")
    elif p_cagr_lower < alpha:
        print(f"  🔴 실제 전략이 무작위보다 통계적 유의하게 나쁨 (p < {alpha})")
        print(f"     → 마이너스 알파 (예측의 정반대로 베팅한 셈)")
    else:
        print(f"  ⚪ 실제 전략과 무작위 차이 없음 (p ≥ {alpha})")
        print(f"     → 신호에 정보 가치 없음 (효율시장가설 일치)")

    return dict(
        strategy=strategy,
        actual=actual,
        permutation=perm,
        p_value_upper=p_cagr_upper,
        p_value_lower=p_cagr_lower,
        p_value_two_sided=p_cagr_two,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='both', choices=['quant', 'robo', 'both'])
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    results = {}
    if args.strategy in ('quant', 'both'):
        results['quant'] = run_full_permutation('quant', args.n)
        print()

    if args.strategy in ('robo', 'both'):
        results['robo'] = run_full_permutation('robo', args.n)
        print()

    # 최종 요약
    print("\n" + "=" * 60)
    print("Permutation Test 최종 요약")
    print("=" * 60)

    for s, r in results.items():
        a = r['actual']
        print(f"\n{s.upper()}:")
        print(f"  실제 CAGR: {a.get('cagr'):.2f}%")
        print(f"  셔플 평균: {r['permutation']['cagrs'].mean():.2f}%")
        print(f"  p(알파 있나): {r['p_value_upper']:.3f}")
        print(f"  p(마이너스 알파인가): {r['p_value_lower']:.3f}")

        if r['p_value_upper'] < 0.05:
            print(f"  → ✅ 통계적 유의 알파")
        elif r['p_value_lower'] < 0.05:
            print(f"  → 🔴 통계적 유의 마이너스 알파")
        else:
            print(f"  → ⚪ 무작위와 차이 없음")

    # 저장
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'permutation_results.npz')
    save_data = {}
    for s, r in results.items():
        save_data[f'{s}_actual_cagr'] = r['actual'].get('cagr', 0)
        save_data[f'{s}_perm_cagrs'] = r['permutation']['cagrs']
        save_data[f'{s}_p_upper'] = r['p_value_upper']
        save_data[f'{s}_p_lower'] = r['p_value_lower']
        save_data[f'{s}_p_two'] = r['p_value_two_sided']
    np.savez(save_path, **save_data)
    print(f"\n💾 결과 저장: {save_path}")

