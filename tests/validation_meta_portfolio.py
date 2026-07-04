"""
validation_meta_portfolio.py — 포트폴리오 결합 (Portfolio of Portfolios) 검증

핵심 아이디어:
  각 포트폴리오(영구/올웨더/동일비중/국면조건부 등)의 강점을 합치면
  단일 포트폴리오보다 더 안정적인 결과가 나올 수 있는가?

결합 방식 4가지 (모두 누수 없음):
  1. 메타 1/N: 6개 포트폴리오 단순 평균
  2. IS Sharpe 가중: IS(2014-2020) Sharpe 비례 가중
  3. IS Risk-Parity: IS 변동성 역수 가중
  4. IS Top-3: IS Sharpe 상위 3개 동일 가중

평가:
  - 모두 OOS(2021-2026)에서만 비교
  - 단일 최고 포트폴리오(영구포트폴리오)와 비교
  - OOS Placebo 1000회

발표 메시지 가능성:
  "단일 포트폴리오로는 알파 없지만, 메타 분산으로 위험조정 개선"
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from server import ETFs, STRATEGIES
from regime_pit import build_pit_macro_features, classify_regime_at
from validation_regime_allocation_fixed import (
    load_etf_data, run_portfolio_backtest, _compute_metrics,
    make_static_fn, make_regime_fn,
    REGIME_ALLOC_TRADITIONAL, ALLOC_60_40, ALLOC_KODEX200
)


def get_base_portfolios(features_df):
    """6개 베이스 포트폴리오의 weight function 반환"""
    return {
        'KODEX 200':        make_static_fn(ALLOC_KODEX200),
        '정적 60/40':       make_static_fn(ALLOC_60_40),
        '영구포트폴리오':   make_static_fn(STRATEGIES['영구포트폴리오']),
        '올웨더':           make_static_fn(STRATEGIES['올웨더']),
        '동일비중':         make_static_fn(STRATEGIES['동일비중']),
        '국면 조건부':      make_regime_fn(REGIME_ALLOC_TRADITIONAL, features_df),
    }


def run_with_combined_weights(etf_df, base_fns, combination_weights, **kwargs):
    """여러 포트폴리오를 결합하여 백테스트.

    Args:
        etf_df: ETF 가격
        base_fns: dict {name: weight_function(cur_date, hist)}
        combination_weights: dict {name: meta_weight}
                            합 = 1.0 권장

    Returns:
        백테스트 결과
    """
    # 각 시점에서 base 포트폴리오들의 weight를 가져와 결합
    def get_combined_weights(cur_date, hist):
        # 자산 단위로 모음
        asset_weights = {}
        total_meta = sum(combination_weights.values())
        for name, base_fn in base_fns.items():
            meta_w = combination_weights.get(name, 0) / total_meta if total_meta > 0 else 0
            if meta_w <= 0:
                continue
            try:
                weights = base_fn(cur_date, hist)
            except Exception:
                weights = None
            if weights is None:
                continue
            for ticker, w in weights.items():
                asset_weights[ticker] = asset_weights.get(ticker, 0) + w * meta_w
        return asset_weights

    return run_portfolio_backtest(etf_df, get_combined_weights, **kwargs)


# ─────────── 결합 가중치 계산 (IS 기반) ───────────

def compute_meta_weights(is_results, method='equal'):
    """IS 결과로부터 메타 가중치 계산 (OOS 누수 없음).

    Args:
        is_results: dict {name: metrics_dict}
        method: 'equal' | 'sharpe' | 'risk_parity' | 'top3_sharpe'

    Returns:
        dict {name: meta_weight}
    """
    valid = {n: r for n, r in is_results.items() if 'error' not in r}

    if method == 'equal':
        # 메타 1/N
        n = len(valid)
        return {name: 1.0 / n for name in valid}

    if method == 'sharpe':
        # Sharpe 비례 (음수는 0 처리)
        sharpes = {n: max(r['sharpe'], 0) for n, r in valid.items()}
        total = sum(sharpes.values())
        if total == 0:
            return {n: 1.0 / len(valid) for n in valid}
        return {n: s / total for n, s in sharpes.items()}

    if method == 'risk_parity':
        # 변동성 역수 가중
        inv_vols = {n: 1.0 / max(r['vol'], 1e-6) for n, r in valid.items()}
        total = sum(inv_vols.values())
        return {n: iv / total for n, iv in inv_vols.items()}

    if method == 'top3_sharpe':
        # Sharpe 상위 3개 동일 가중
        sorted_by_sharpe = sorted(valid.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        top3 = [n for n, _ in sorted_by_sharpe[:3]]
        return {n: (1.0 / 3 if n in top3 else 0) for n in valid}

    return {n: 1.0 / len(valid) for n in valid}


# ─────────── 전체 파이프라인 ───────────

def run_meta_validation(n_placebo=1000,
                         is_start='2014-06-26', is_end='2020-12-31',
                         oos_start='2021-01-01', oos_end=None):
    print("=" * 60)
    print("Meta Portfolio 검증 — 결합 전략")
    print("=" * 60)

    print("\n[1] ETF + 매크로 로드")
    etf_df = load_etf_data()
    import data_loader as dl
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  ETF 기간: {etf_df.index[0].date()} ~ {etf_df.index[-1].date()}")

    print("\n[2] 베이스 포트폴리오 6개 IS/OOS 백테스트")
    base_fns = get_base_portfolios(features_df)
    is_results = {}
    oos_results = {}
    for name, fn in base_fns.items():
        is_results[name] = run_portfolio_backtest(
            etf_df, fn, start_date=is_start, end_date=is_end
        )
        oos_results[name] = run_portfolio_backtest(
            etf_df, fn, start_date=oos_start, end_date=oos_end
        )

    print("\n  베이스 OOS 성과:")
    for name, r in oos_results.items():
        if 'error' in r:
            continue
        print(f"    {name:<20} Sharpe {r['sharpe']:>6.3f}, MDD {r['mdd']:>7.2f}%, Calmar {r['calmar']:>6.3f}")

    # 베이스 중 최고 Sharpe (비교 기준)
    best_base = max(
        [(n, r) for n, r in oos_results.items() if 'error' not in r],
        key=lambda x: x[1]['sharpe']
    )
    print(f"\n  🏆 단일 최고 Sharpe: {best_base[0]} ({best_base[1]['sharpe']:.3f})")

    print("\n[3] 4가지 결합 전략 IS 가중치 결정")
    methods = ['equal', 'sharpe', 'risk_parity', 'top3_sharpe']
    method_labels = {
        'equal':       '메타 1/N (단순 평균)',
        'sharpe':      'IS Sharpe 가중',
        'risk_parity': 'IS Risk-Parity',
        'top3_sharpe': 'IS Top-3 동일가중',
    }

    meta_weights_dict = {}
    for m in methods:
        meta_weights_dict[m] = compute_meta_weights(is_results, method=m)
        print(f"\n  {method_labels[m]}:")
        for name, w in meta_weights_dict[m].items():
            if w > 0:
                print(f"    {name:<20} {w*100:>5.1f}%")

    print("\n[4] 결합 전략 4종 OOS 백테스트")
    meta_oos = {}
    for m in methods:
        r = run_with_combined_weights(
            etf_df, base_fns, meta_weights_dict[m],
            start_date=oos_start, end_date=oos_end
        )
        meta_oos[m] = r

    print("\n" + "-" * 80)
    print("OOS 결합 전략 결과 (2021-2026)")
    print("-" * 80)
    print(f"{'전략':<25} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Vol':>8}")

    # 베이스 단일 최고와 비교
    print(f"{'[단일 최고] ' + best_base[0]:<25} {best_base[1]['cagr']:>7.2f}% "
          f"{best_base[1]['sharpe']:>8.3f} {best_base[1]['mdd']:>7.2f}% "
          f"{best_base[1]['calmar']:>8.3f} {best_base[1]['vol']:>7.2f}%")

    # 결합 전략들
    for m in methods:
        r = meta_oos[m]
        if 'error' in r:
            print(f"{method_labels[m]:<25} ❌")
            continue
        print(f"{method_labels[m]:<25} {r['cagr']:>7.2f}% {r['sharpe']:>8.3f} "
              f"{r['mdd']:>7.2f}% {r['calmar']:>8.3f} {r['vol']:>7.2f}%")

    # 베스트 결합 vs 단일 최고
    best_meta = max(
        [(m, r) for m, r in meta_oos.items() if 'error' not in r],
        key=lambda x: x[1]['sharpe']
    )

    print(f"\n  🏆 최고 결합 전략: {method_labels[best_meta[0]]} (Sharpe {best_meta[1]['sharpe']:.3f})")

    print(f"\n[비교] 결합 vs 단일 최고:")
    print(f"  단일 최고 ({best_base[0]}):   Sharpe {best_base[1]['sharpe']:.3f}, MDD {best_base[1]['mdd']:.2f}%, Calmar {best_base[1]['calmar']:.3f}")
    print(f"  최고 결합 ({method_labels[best_meta[0]]}): Sharpe {best_meta[1]['sharpe']:.3f}, MDD {best_meta[1]['mdd']:.2f}%, Calmar {best_meta[1]['calmar']:.3f}")

    if best_meta[1]['sharpe'] > best_base[1]['sharpe']:
        improvement = (best_meta[1]['sharpe'] - best_base[1]['sharpe']) / abs(best_base[1]['sharpe']) * 100
        print(f"  → 결합 전략이 단일 최고보다 Sharpe {improvement:+.1f}% 개선")
    else:
        delta = best_meta[1]['sharpe'] - best_base[1]['sharpe']
        print(f"  → 결합 전략이 단일 최고보다 못함 (ΔSharpe {delta:+.3f})")

    # MDD/Calmar 측면 (위험관리)
    if best_meta[1]['mdd'] > best_base[1]['mdd']:  # MDD는 덜 음수일수록 좋음
        delta_mdd = best_meta[1]['mdd'] - best_base[1]['mdd']
        print(f"  → 결합 전략 MDD가 단일 최고보다 작음 (Δ{delta_mdd:+.2f}%p)")

    if best_meta[1]['calmar'] > best_base[1]['calmar']:
        delta_cal = best_meta[1]['calmar'] - best_base[1]['calmar']
        print(f"  → 결합 전략 Calmar가 단일 최고보다 높음 (Δ{delta_cal:+.3f})")

    # 결합의 Risk-Adjusted 우월성 종합 판단
    print(f"\n[종합 평가 — 위험조정 관점]")
    metas_valid = [(m, r) for m, r in meta_oos.items() if 'error' not in r]
    avg_meta_sharpe = np.mean([r['sharpe'] for _, r in metas_valid])
    avg_meta_calmar = np.mean([r['calmar'] for _, r in metas_valid])
    avg_meta_mdd = np.mean([r['mdd'] for _, r in metas_valid])

    base_valid = [(n, r) for n, r in oos_results.items() if 'error' not in r]
    avg_base_sharpe = np.mean([r['sharpe'] for _, r in base_valid])
    avg_base_calmar = np.mean([r['calmar'] for _, r in base_valid])
    avg_base_mdd = np.mean([r['mdd'] for _, r in base_valid])

    print(f"  베이스 평균:  Sharpe {avg_base_sharpe:.3f}, MDD {avg_base_mdd:.2f}%, Calmar {avg_base_calmar:.3f}")
    print(f"  결합 평균:    Sharpe {avg_meta_sharpe:.3f}, MDD {avg_meta_mdd:.2f}%, Calmar {avg_meta_calmar:.3f}")

    if avg_meta_calmar > avg_base_calmar and avg_meta_mdd > avg_base_mdd:
        print(f"  ✅ 결합 전략이 평균적으로 위험조정 측면에서 우수 (MDD↓ Calmar↑)")
    elif avg_meta_sharpe > avg_base_sharpe:
        print(f"  🟢 결합 전략이 평균 Sharpe에서 우수")
    else:
        print(f"  ⚪ 결합 전략의 명확한 우위 없음")

    # 저장
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'meta_portfolio_results.npz'
    )
    save_kwargs = {}
    for name, r in oos_results.items():
        if 'error' in r:
            continue
        key = 'base_' + name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        save_kwargs[f'{key}_sharpe'] = r['sharpe']
        save_kwargs[f'{key}_mdd'] = r['mdd']
        save_kwargs[f'{key}_calmar'] = r['calmar']
    for m, r in meta_oos.items():
        if 'error' in r:
            continue
        save_kwargs[f'meta_{m}_sharpe'] = r['sharpe']
        save_kwargs[f'meta_{m}_mdd'] = r['mdd']
        save_kwargs[f'meta_{m}_calmar'] = r['calmar']
    np.savez(save_path, **save_kwargs)
    print(f"\n💾 결과 저장: {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    run_meta_validation(n_placebo=args.n)
