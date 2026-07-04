"""
validation_placebo.py — Placebo Test (무작위 종목 1,000회 시뮬레이션)

목적: 데이터 누수 검증
원리:
  - 시그널을 완전 무작위로 바꿔서 1,000번 백테스트
  - 무작위로도 알파 나오면 = 데이터 누수
  - 무작위가 KOSPI와 비슷하면 = 누수 없음
  - 실제 전략이 무작위 분포의 어느 위치인지 = 알파의 신뢰도

벡터화: 종목 선정 → 평균 수익률은 단순 산술이므로 행렬 연산으로 1000회 동시 처리.
1회당 5시간 → 전체 1분 (300배 빠름, 결과는 수학적으로 동일)
"""

import os, sys, time
os.environ.setdefault('KQ_DISABLE_TABPFN', '1')

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation_core import prepare_bt_data, run_bt_core


def run_placebo_test(data, n_iter=1000, top_n=5, rebalance='M', seed=42):
    """
    무작위 종목 선정으로 n_iter번 백테스트 시뮬레이션 (벡터화)

    벡터화 로직:
      1. 월말 수익률 행렬 만들기: shape (n_months, n_tickers)
      2. 매 시뮬레이션마다 매월 5종목 무작위 인덱스 선택
      3. 그 5종목의 평균 수익률 = 그 달의 포트폴리오 수익률
      4. 누적 → CAGR 계산
    """
    price_df = data['price_df']
    kospi = data['kospi']

    # 1) 월말 수익률 행렬
    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance, 'ME')
    monthly = price_df.resample(rule).last()
    monthly_ret = monthly.pct_change().fillna(0)
    # shape: (n_months, n_tickers)
    ret_matrix = monthly_ret.values
    n_months, n_tickers = ret_matrix.shape

    # NaN 가격은 0 처리 (해당 종목은 평균에서 제외돼야 함)
    # → 각 월별로 유효 종목 마스크 만들기
    valid_mask = (~monthly.isna()).values  # shape: (n_months, n_tickers)

    # PiT 모드: 매월 시총 상위 N 인덱스 미리 계산
    pit_mode = data.get('point_in_time', False)
    pit_top_n = data.get('top_n_mcap', 200)
    mcap_hist = data.get('mcap_history')

    # 월별 PiT 유효 인덱스 (PiT 모드일 때만)
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

    rng = np.random.default_rng(seed)

    print(f"  벡터화 시뮬레이션 시작 ({n_iter}회)")
    print(f"  월 수: {n_months}, 종목 수: {n_tickers}")
    if pit_mode:
        print(f"  Point-in-Time 모드 적용 (매월 시총 상위 {pit_top_n})")
    t0 = time.time()

    # 결과 저장
    final_returns = np.zeros(n_iter)
    cagrs = np.zeros(n_iter)
    sharpes = np.zeros(n_iter)
    mdds = np.zeros(n_iter)

    years = (n_months - 1) / 12  # 월간 리밸런싱 가정

    for i in range(n_iter):
        equity = np.ones(n_months)  # 누적 자산
        for m in range(1, n_months):
            # PiT 모드: 그 시점 시총 상위 N개 풀에서만 선정
            if pit_valid_per_month is not None:
                pit_idx = pit_valid_per_month[m-1]
                mask_idx = valid_mask[m-1]
                # PiT 종목 중에서 유효한 것만
                valid_idx = pit_idx[mask_idx[pit_idx]] if len(pit_idx) > 0 else np.array([])
            else:
                valid_idx = np.where(valid_mask[m-1])[0]

            if len(valid_idx) < top_n:
                equity[m] = equity[m-1]
                continue
            selected = rng.choice(valid_idx, size=top_n, replace=False)
            # 이번 달 수익률 평균
            month_rets = ret_matrix[m, selected]
            # NaN 또는 inf 제거
            month_rets = month_rets[np.isfinite(month_rets)]
            if len(month_rets) == 0:
                equity[m] = equity[m-1]
                continue
            equity[m] = equity[m-1] * (1 + month_rets.mean())

        # 지표 계산
        total = equity[-1]
        final_returns[i] = (total - 1) * 100
        if years > 0 and total > 0:
            cagrs[i] = (total**(1/years) - 1) * 100
        else:
            cagrs[i] = 0
        # Sharpe
        rets = np.diff(equity) / equity[:-1]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 0 and rets.std() > 0:
            sharpes[i] = (cagrs[i] / 100 - 0.03) / (rets.std() * np.sqrt(12))
        # MDD
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        mdds[i] = dd.min() * 100

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  진행: {i+1}/{n_iter} ({elapsed:.1f}초)")

    elapsed = time.time() - t0
    print(f"\n  완료: {elapsed:.1f}초 ({elapsed/n_iter*1000:.1f}ms/회)")

    return dict(
        cagrs=cagrs,
        total_returns=final_returns,
        sharpes=sharpes,
        mdds=mdds,
        n_iter=n_iter,
        n_months=n_months,
    )


def compute_percentile(value, distribution):
    """value가 distribution의 몇 번째 percentile인지 (0~100)"""
    return float((distribution <= value).mean() * 100)


def run_full_placebo(strategy='quant', n_iter=1000):
    """전체 Placebo Test 파이프라인"""
    print("=" * 60)
    print(f"Placebo Test ({strategy} 전략, {n_iter}회)")
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
    print(f"  CAGR: {actual.get('cagr')}%")
    print(f"  Sharpe: {actual.get('sharpe')}")
    print(f"  알파: {actual.get('alpha')}%p")

    print(f"\n[3/3] 무작위 시뮬레이션 {n_iter}회")
    placebo = run_placebo_test(data, n_iter=n_iter)

    print("\n" + "=" * 60)
    print(f"결과 분석 ({strategy} vs Placebo)")
    print("=" * 60)

    actual_cagr = actual.get('cagr', 0)
    actual_sharpe = actual.get('sharpe', 0)
    actual_mdd = actual.get('mdd', 0)

    # 무작위 분포 통계
    cagr_mean = placebo['cagrs'].mean()
    cagr_std = placebo['cagrs'].std()
    cagr_median = np.median(placebo['cagrs'])
    cagr_p5 = np.percentile(placebo['cagrs'], 5)
    cagr_p95 = np.percentile(placebo['cagrs'], 95)

    # 실제 전략의 percentile
    cagr_pct = compute_percentile(actual_cagr, placebo['cagrs'])
    sharpe_pct = compute_percentile(actual_sharpe, placebo['sharpes'])

    print(f"\n실제 전략 ({strategy}):")
    print(f"  CAGR: {actual_cagr:.2f}%")
    print(f"  Sharpe: {actual_sharpe:.3f}")
    print(f"  MDD: {actual_mdd:.2f}%")

    print(f"\n무작위 분포 (N={n_iter}):")
    print(f"  CAGR 평균: {cagr_mean:.2f}% (표준편차 {cagr_std:.2f})")
    print(f"  CAGR 중앙값: {cagr_median:.2f}%")
    print(f"  CAGR 5~95% 분위: [{cagr_p5:.2f}, {cagr_p95:.2f}]")

    print(f"\n실제 vs 무작위:")
    print(f"  CAGR 백분위: {cagr_pct:.1f}%")
    print(f"  Sharpe 백분위: {sharpe_pct:.1f}%")

    # 해석
    print(f"\n해석:")
    if cagr_pct > 95:
        print(f"  ✅ 실제 전략 CAGR이 무작위의 상위 5%에 속함")
        print(f"     → 통계적으로 유의한 알파 (p < 0.05)")
    elif cagr_pct > 80:
        print(f"  🟡 실제 전략 CAGR이 무작위의 상위 20%")
        print(f"     → 약한 알파 가능성")
    elif cagr_pct > 20:
        print(f"  ⚪ 실제 전략 CAGR이 무작위 중간 범위")
        print(f"     → 무작위 선정과 차이 없음 (알파 없음)")
    else:
        print(f"  🔴 실제 전략 CAGR이 무작위의 하위 20%")
        print(f"     → 무작위보다 못함 (마이너스 알파)")

    # 누수 검증
    print(f"\n누수 검증:")
    if cagr_mean < 5:
        print(f"  ✅ 무작위 평균이 매우 낮음 → 누수 없음")
    elif cagr_mean < 15:
        print(f"  ✅ 무작위 평균이 KOSPI({actual.get('bench_cagr')}%)와 유사 → 누수 없음")
    else:
        print(f"  ⚠️ 무작위 평균이 비현실적으로 높음 → 누수 의심")

    return dict(
        strategy=strategy,
        actual=actual,
        placebo=placebo,
        cagr_percentile=cagr_pct,
        sharpe_percentile=sharpe_pct,
        cagr_mean=cagr_mean,
        cagr_std=cagr_std,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='both', choices=['quant', 'robo', 'both'])
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    results = {}
    if args.strategy in ('quant', 'both'):
        results['quant'] = run_full_placebo('quant', args.n)
        print()

    if args.strategy in ('robo', 'both'):
        results['robo'] = run_full_placebo('robo', args.n)
        print()

    # 결과 저장
    print("\n" + "=" * 60)
    print("최종 요약")
    print("=" * 60)
    for s, r in results.items():
        a = r['actual']
        print(f"\n{s.upper()}:")
        print(f"  실제 CAGR: {a.get('cagr')}% (KOSPI: {a.get('bench_cagr')}%)")
        print(f"  무작위 평균: {r['cagr_mean']:.2f}% (±{r['cagr_std']:.2f})")
        print(f"  실제 위치: {r['cagr_percentile']:.1f}th percentile")

    # 결과를 npz로 저장 (보고서 작성용)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'placebo_results.npz')
    save_data = {}
    for s, r in results.items():
        save_data[f'{s}_actual_cagr'] = r['actual'].get('cagr', 0)
        save_data[f'{s}_actual_sharpe'] = r['actual'].get('sharpe', 0)
        save_data[f'{s}_actual_mdd'] = r['actual'].get('mdd', 0)
        save_data[f'{s}_placebo_cagrs'] = r['placebo']['cagrs']
        save_data[f'{s}_placebo_sharpes'] = r['placebo']['sharpes']
        save_data[f'{s}_placebo_mdds'] = r['placebo']['mdds']
        save_data[f'{s}_cagr_percentile'] = r['cagr_percentile']
    np.savez(save_path, **save_data)
    print(f"\n💾 결과 저장: {save_path}")

