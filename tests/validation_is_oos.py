"""
validation_is_oos.py — In-Sample vs Out-of-Sample 분리 검증

목적: 과적합 + 시간 안정성 검증
원리:
  - IS (학습 구간):  2014 ~ 2020 (7년)
  - OOS (검증 구간): 2021 ~ 2026 (5년)
  - 두 구간 성과 비교

해석:
  - IS > OOS 큰 격차: 과적합 / 누수 의심
  - IS ≈ OOS: 시간 안정성 검증 (진짜 결과)
  - 둘 다 동시에 측정해야 의미 있음
"""

import os, sys, time
os.environ.setdefault('KQ_DISABLE_TABPFN', '1')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation_core import prepare_bt_data, run_bt_core


def run_is_oos_split(strategy='quant', top_n=5,
                     is_start='2014-01-01', is_end='2020-12-31',
                     oos_start='2021-01-01', oos_end='2026-12-31'):
    """IS / OOS 분리 백테스트"""

    print("=" * 60)
    print(f"IS/OOS Split Test ({strategy} 전략)")
    print(f"  IS:  {is_start} ~ {is_end}")
    print(f"  OOS: {oos_start} ~ {oos_end}")
    print("=" * 60)

    import server
    universe = server.UNIVERSE
    ticker_to_code = server._ticker_to_code

    # === IS (In-Sample) ===
    print(f"\n[1/3] IS 데이터 준비 ({is_start} ~ {is_end}) — 시총 상위 200")
    t0 = time.time()
    data_is = prepare_bt_data(
        universe, ticker_to_code,
        start_date=pd.Timestamp(is_start),
        end_date=pd.Timestamp(is_end),
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  price_df: {data_is['price_df'].shape}")

    print(f"\n[2/3] OOS 데이터 준비 ({oos_start} ~ {oos_end}) — 시총 상위 200")
    t0 = time.time()
    data_oos = prepare_bt_data(
        universe, ticker_to_code,
        start_date=pd.Timestamp(oos_start),
        end_date=pd.Timestamp(oos_end),
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  price_df: {data_oos['price_df'].shape}")

    print(f"\n[3/3] 백테스트 실행")
    r_is = run_bt_core(data_is, strategy=strategy, top_n=top_n)
    r_oos = run_bt_core(data_oos, strategy=strategy, top_n=top_n)

    # === 결과 비교 ===
    print("\n" + "=" * 60)
    print(f"{strategy.upper()} — IS vs OOS 비교")
    print("=" * 60)

    print(f"\n{'지표':<15} {'IS (학습)':<15} {'OOS (검증)':<15} {'차이':<10}")
    print("-" * 55)

    is_cagr = r_is.get('cagr', 0)
    oos_cagr = r_oos.get('cagr', 0)
    is_sharpe = r_is.get('sharpe', 0)
    oos_sharpe = r_oos.get('sharpe', 0)
    is_mdd = r_is.get('mdd', 0)
    oos_mdd = r_oos.get('mdd', 0)
    is_bench = r_is.get('bench_cagr', 0)
    oos_bench = r_oos.get('bench_cagr', 0)
    is_alpha = r_is.get('alpha', 0)
    oos_alpha = r_oos.get('alpha', 0)

    print(f"{'CAGR':<15} {is_cagr:>8.2f}%      {oos_cagr:>8.2f}%      {oos_cagr-is_cagr:+.2f}%p")
    print(f"{'Sharpe':<15} {is_sharpe:>8.3f}       {oos_sharpe:>8.3f}       {oos_sharpe-is_sharpe:+.3f}")
    print(f"{'MDD':<15} {is_mdd:>8.2f}%      {oos_mdd:>8.2f}%      {oos_mdd-is_mdd:+.2f}%p")
    print(f"{'KOSPI CAGR':<15} {is_bench:>8.2f}%      {oos_bench:>8.2f}%      {oos_bench-is_bench:+.2f}%p")
    print(f"{'알파':<15} {is_alpha:>8.2f}%p     {oos_alpha:>8.2f}%p     {oos_alpha-is_alpha:+.2f}%p")
    print(f"{'리밸런싱 횟수':<15} {r_is.get('n_rebalance'):<15} {r_oos.get('n_rebalance'):<15}")

    # === 해석 ===
    print("\n" + "=" * 60)
    print("해석")
    print("=" * 60)

    cagr_gap = abs(is_cagr - oos_cagr)
    sharpe_gap = abs(is_sharpe - oos_sharpe)

    print(f"\nCAGR 격차: {cagr_gap:.2f}%p")
    if cagr_gap < 5:
        print("  ✅ IS/OOS CAGR 일관성 매우 높음 → 과적합 없음")
    elif cagr_gap < 15:
        print("  🟡 IS/OOS CAGR 적당한 차이 → 약한 과적합 가능성")
    else:
        print("  🔴 IS/OOS CAGR 큰 격차 → 과적합 또는 데이터 누수 의심")

    print(f"\nSharpe 격차: {sharpe_gap:.3f}")
    if sharpe_gap < 0.3:
        print("  ✅ Sharpe 일관성 → 위험조정수익률 안정")
    else:
        print("  🟡 Sharpe 차이 → 변동성 변화 가능성")

    print(f"\n알파 일관성:")
    if (is_alpha < 0 and oos_alpha < 0) or (is_alpha > 0 and oos_alpha > 0):
        print(f"  ✅ IS/OOS 알파 부호 일치 ({is_alpha:+.2f} → {oos_alpha:+.2f})")
        print(f"     → 시간에 따라 결과 일관됨")
    else:
        print(f"  🟡 IS/OOS 알파 부호 불일치 ({is_alpha:+.2f} → {oos_alpha:+.2f})")
        print(f"     → 시기에 따라 결과 변동 가능")

    return dict(
        strategy=strategy,
        is_result=r_is,
        oos_result=r_oos,
        is_period=(is_start, is_end),
        oos_period=(oos_start, oos_end),
        cagr_gap=cagr_gap,
        sharpe_gap=sharpe_gap,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', default='both', choices=['quant', 'robo', 'both'])
    args = parser.parse_args()

    results = {}
    if args.strategy in ('quant', 'both'):
        results['quant'] = run_is_oos_split('quant')
        print()

    if args.strategy in ('robo', 'both'):
        results['robo'] = run_is_oos_split('robo')
        print()

    # === 종합 요약 ===
    print("\n" + "=" * 60)
    print("최종 요약 (IS/OOS Split Test)")
    print("=" * 60)

    for s, r in results.items():
        is_r = r['is_result']
        oos_r = r['oos_result']
        print(f"\n{s.upper()}:")
        print(f"  IS  CAGR: {is_r.get('cagr'):>7.2f}% | KOSPI: {is_r.get('bench_cagr'):>6.2f}% | 알파: {is_r.get('alpha'):+.2f}%p")
        print(f"  OOS CAGR: {oos_r.get('cagr'):>7.2f}% | KOSPI: {oos_r.get('bench_cagr'):>6.2f}% | 알파: {oos_r.get('alpha'):+.2f}%p")
        print(f"  격차: CAGR {r['cagr_gap']:.2f}%p, Sharpe {r['sharpe_gap']:.3f}")

    # 결과 저장
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'is_oos_results.npz')
    save_data = {}
    for s, r in results.items():
        save_data[f'{s}_is_cagr'] = r['is_result'].get('cagr', 0)
        save_data[f'{s}_oos_cagr'] = r['oos_result'].get('cagr', 0)
        save_data[f'{s}_is_alpha'] = r['is_result'].get('alpha', 0)
        save_data[f'{s}_oos_alpha'] = r['oos_result'].get('alpha', 0)
        save_data[f'{s}_is_sharpe'] = r['is_result'].get('sharpe', 0)
        save_data[f'{s}_oos_sharpe'] = r['oos_result'].get('sharpe', 0)
        save_data[f'{s}_cagr_gap'] = r['cagr_gap']
    np.savez(save_path, **save_data)
    print(f"\n💾 결과 저장: {save_path}")

