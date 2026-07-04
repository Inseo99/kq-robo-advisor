"""
validation_recommend_portfolio.py — 추천 포트폴리오 레이어 검증

목적:
  추천 탭의 핵심 구조를 기존 검증과 같은 기준으로 점검한다.

검증 레이어:
  A. 운영 베이스: 동일비중 40% + 영구포트폴리오 35% + 올웨더 25%
  B. IS Sharpe 베이스: OOS를 보지 않고 IS(2014~2020) 성과로 3개 베이스를 가중
  C. 운영 베이스 + 국면 틸트: 추천 탭의 REGIME_TARGETS와 자동 틸트 룰 적용
  D. IS 베이스 + 국면 틸트: look-ahead 우려를 줄인 추천형 포트폴리오
  E. IS 베이스 + 국면 틸트 + PiT 로보 프록시: 과거 데이터만 쓰는 로보신호 미세조정

주의:
  실제 추천 탭의 analyze_stock()은 "오늘 기준" 단일 종목 분석 함수다.
  과거 백테스트에 그대로 넣으면 미래 데이터가 섞일 수 있으므로,
  이 스크립트에서는 과거 시점까지의 가격만 쓰는 간단한 PiT 로보 프록시를 사용한다.
"""
import os
import sys

os.environ['KQ_DISABLE_TABPFN'] = '1'
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from server import STRATEGIES, REGIME_TARGETS
from regime_pit import build_pit_macro_features, classify_regime_at
from validation_regime_allocation_fixed import (
    load_etf_data,
    run_portfolio_backtest,
    make_static_fn,
    ALLOC_60_40,
    ALLOC_KODEX200,
)


IS_START = '2014-06-26'
IS_END = '2020-12-31'
OOS_START = '2021-01-01'

CORE_COMPONENTS = ['동일비중', '영구포트폴리오', '올웨더']
OPERATING_META_WEIGHTS = {
    '동일비중': 0.40,
    '영구포트폴리오': 0.35,
    '올웨더': 0.25,
}


def normalize_weights(weights):
    clean = {k: float(v) for k, v in weights.items() if v and v > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def combine_weight_sets(weight_sets):
    out = {}
    total_mix = 0.0
    for mix_w, weights in weight_sets:
        if not weights or mix_w <= 0:
            continue
        total_mix += float(mix_w)
        for ticker, weight in weights.items():
            out[ticker] = out.get(ticker, 0.0) + float(mix_w) * float(weight)
    if total_mix > 0:
        out = {ticker: weight / total_mix for ticker, weight in out.items()}
    return normalize_weights(out)


def meta_weights_to_asset_weights(meta_weights):
    sets = []
    for name, meta_w in meta_weights.items():
        weights = STRATEGIES.get(name)
        if weights:
            sets.append((meta_w, weights))
    return combine_weight_sets(sets)


def compute_is_sharpe_meta_weights(is_results):
    sharpes = {}
    for name in CORE_COMPONENTS:
        result = is_results.get(name, {})
        sharpes[name] = max(float(result.get('sharpe', 0.0)), 0.0)
    total = sum(sharpes.values())
    if total <= 0:
        return {name: 1.0 / len(CORE_COMPONENTS) for name in CORE_COMPONENTS}
    return {name: value / total for name, value in sharpes.items()}


def regime_sequence_until(features_df, as_of_date):
    visible = features_df[features_df['available_from'] <= as_of_date]
    labels = []
    if len(visible) < 4:
        return labels
    for end in range(4, len(visible) + 1):
        hist = visible.iloc[:end]
        latest = hist.iloc[-1]
        growth_med = hist['gdp_growth'].median()
        spread_med = hist['spread'].median()
        growth_up = latest['gdp_growth'] > growth_med
        inflation_up = latest['spread'] < spread_med
        if growth_up and not inflation_up:
            labels.append('골디락스')
        elif growth_up and inflation_up:
            labels.append('리플레이션')
        elif not growth_up and inflation_up:
            labels.append('스태그플레이션')
        else:
            labels.append('디플레이션')
    return labels


def pit_snapshot(features_df, cur_date, regime):
    """추천 탭 자동 틸트에 넣을 PiT snapshot.

    현재 UI의 TabPFN 확률을 과거 전 구간으로 정확히 되감기는 어렵다.
    대신 해당 시점까지 발표된 매크로만으로 confidence와 동일국면 유지확률을
    보수적으로 추정한다.
    """
    visible = features_df[features_df['available_from'] <= cur_date]
    if len(visible) < 4 or regime is None:
        return dict(current=regime or '골디락스', confidence=0.50, next_quarter={})

    latest = visible.iloc[-1]
    growth_std = float(visible['gdp_growth'].std()) or 1.0
    spread_std = float(visible['spread'].std()) or 1.0
    growth_dist = abs(float(latest['gdp_growth'] - visible['gdp_growth'].median())) / max(growth_std, 1e-9)
    spread_dist = abs(float(latest['spread'] - visible['spread'].median())) / max(spread_std, 1e-9)
    raw = (growth_dist + spread_dist) / 2.0
    confidence = 0.50 + 0.45 * (raw / (1.0 + raw))
    confidence = float(max(0.50, min(0.95, confidence)))

    labels = regime_sequence_until(features_df, cur_date)
    stay_prob = None
    if len(labels) >= 2:
        same = 1.0
        total = 4.0
        for a, b in zip(labels[:-1], labels[1:]):
            if a == regime:
                total += 1.0
                if b == regime:
                    same += 1.0
        stay_prob = same / total

    next_q = {regime: stay_prob} if stay_prob is not None else {}
    return dict(current=regime, confidence=confidence, next_quarter=next_q)


def build_context_cache(etf_df, features_df, start_date=None, end_date=None, rebalance='M'):
    """리밸런싱 날짜별 PiT 국면/snapshot을 한 번만 계산해 Placebo 속도를 높인다."""
    df = etf_df.copy()
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]
    rule = {'M': 'ME', 'Q': 'QE', 'W': 'W'}.get(rebalance, 'ME')
    months = df.resample(rule).last()
    cache = {}
    for cur_date in months.index[:-1]:
        regime, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
        if regime is None:
            regime = '골디락스'
        cache[pd.Timestamp(cur_date)] = {
            'regime': regime,
            'snapshot': pit_snapshot(features_df, cur_date, regime),
        }
    return cache


def pit_robo_signal_map(hist, tickers):
    """과거 시점까지의 ETF 가격만 쓰는 간단한 로보신호 프록시."""
    signal_map = {}
    for ticker in tickers:
        if ticker not in hist.columns:
            signal_map[ticker] = {'cw_signal': '관망', 'confidence': 0.0, 'exit_days': None}
            continue
        s = hist[ticker].dropna()
        if len(s) < 60:
            signal_map[ticker] = {'cw_signal': '관망', 'confidence': 0.0, 'exit_days': None}
            continue

        cur = float(s.iloc[-1])
        ma20 = float(s.rolling(20).mean().iloc[-1])
        ma60 = float(s.rolling(60).mean().iloc[-1])
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        if pd.isna(gain) or pd.isna(loss) or loss == 0:
            rsi = 50.0
        else:
            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        score = 0.0
        score += 20.0 if cur > ma20 else -20.0
        score += 20.0 if cur > ma60 else -20.0
        if rsi < 30:
            score += 15.0
        elif rsi > 70:
            score -= 15.0

        if score >= 25:
            action = '매수'
        elif score <= -25:
            action = '매도'
        else:
            action = '관망'

        confidence = min(1.0, abs(score) / 55.0)
        signal_map[ticker] = {
            'cw_signal': action,
            'confidence': confidence,
            'exit_days': 10,
            'score': score,
        }
    return signal_map


def make_recommend_fn(base_weights, features_df, use_regime=True,
                      use_signal=False, fixed_tilt=None, shuffle_rng=None,
                      context_cache=None):
    regime_names = list(REGIME_TARGETS.keys())
    memo = {}

    def get_context(cur_date):
        key = pd.Timestamp(cur_date)
        if context_cache is not None and key in context_cache:
            return context_cache[key]
        if key not in memo:
            regime, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
            if regime is None:
                regime = '골디락스'
            memo[key] = {
                'regime': regime,
                'snapshot': pit_snapshot(features_df, cur_date, regime),
            }
        return memo[key]

    def get_weights(cur_date, hist):
        ctx = get_context(cur_date)
        actual_regime = ctx['regime']
        regime = actual_regime
        if shuffle_rng is not None:
            regime = shuffle_rng.choice(regime_names)

        weights = dict(base_weights)
        if use_regime:
            snapshot = ctx['snapshot']
            auto = server._auto_regime_tilt(snapshot)
            regime_tilt = auto['regime_tilt'] if fixed_tilt is None else float(fixed_tilt)
            target = REGIME_TARGETS.get(regime, REGIME_TARGETS['리플레이션'])
            weights = combine_weight_sets([(1.0 - regime_tilt, weights), (regime_tilt, target)])

        if use_signal:
            signal_map = pit_robo_signal_map(hist, list(weights.keys()))
            weights, _ = server._apply_signal_tilt(weights, signal_map)

        return normalize_weights(weights)

    return get_weights


def print_table(title, results):
    print("\n" + "-" * 88)
    print(title)
    print("-" * 88)
    print(f"{'포트폴리오':<34} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Vol':>8}")
    for name, result in results.items():
        if 'error' in result:
            print(f"{name:<34} ERROR: {result['error']}")
            continue
        print(f"{name:<34} {result['cagr']:>7.2f}% {result['sharpe']:>8.3f} "
              f"{result['mdd']:>7.2f}% {result['calmar']:>8.3f} {result['vol']:>7.2f}%")


def run_placebo(etf_df, features_df, base_weights, n_placebo=1000,
                oos_start=OOS_START, oos_end=None):
    print("\n" + "=" * 60)
    print(f"OOS Placebo — 추천 국면 틸트 셔플 ({n_placebo}회)")
    print("=" * 60)

    context_cache = build_context_cache(etf_df, features_df, start_date=oos_start, end_date=oos_end)
    actual_fn = make_recommend_fn(base_weights, features_df, use_regime=True, use_signal=False,
                                  context_cache=context_cache)
    actual = run_portfolio_backtest(etf_df, actual_fn, start_date=oos_start, end_date=oos_end)
    print(f"  실제: Sharpe {actual['sharpe']:.3f}, MDD {actual['mdd']:.2f}%, Calmar {actual['calmar']:.3f}")

    rng = np.random.default_rng(42)
    sharpes, mdds, calmars, cagrs = [], [], [], []
    t0 = time.time()
    for i in range(n_placebo):
        sub_rng = np.random.default_rng(rng.integers(0, 1_000_000_000))
        fn = make_recommend_fn(base_weights, features_df, use_regime=True,
                               use_signal=False, shuffle_rng=sub_rng,
                               context_cache=context_cache)
        result = run_portfolio_backtest(etf_df, fn, start_date=oos_start, end_date=oos_end)
        if 'error' not in result:
            sharpes.append(result['sharpe'])
            mdds.append(result['mdd'])
            calmars.append(result['calmar'])
            cagrs.append(result['cagr'])
        if (i + 1) % 200 == 0:
            print(f"  진행: {i+1}/{n_placebo} ({time.time()-t0:.1f}초)")

    sharpes = np.array(sharpes)
    mdds = np.array(mdds)
    calmars = np.array(calmars)
    cagrs = np.array(cagrs)

    p_sharpe = float((sharpes >= actual['sharpe']).mean())
    p_calmar = float((calmars >= actual['calmar']).mean())
    sharpe_pct = float((sharpes <= actual['sharpe']).mean() * 100.0)
    mdd_pct = float((mdds <= actual['mdd']).mean() * 100.0)
    calmar_pct = float((calmars <= actual['calmar']).mean() * 100.0)

    print("\nPlacebo 결과:")
    print(f"  Sharpe — 실제 {actual['sharpe']:.3f} vs 셔플평균 {sharpes.mean():.3f} "
          f"(percentile={sharpe_pct:.1f}%, p={p_sharpe:.3f})")
    print(f"  MDD    — 실제 {actual['mdd']:.2f}% vs 셔플평균 {mdds.mean():.2f}% "
          f"(percentile={mdd_pct:.1f}%)")
    print(f"  Calmar — 실제 {actual['calmar']:.3f} vs 셔플평균 {calmars.mean():.3f} "
          f"(percentile={calmar_pct:.1f}%, p={p_calmar:.3f})")

    if p_sharpe < 0.05 or p_calmar < 0.05:
        print("  판단: 국면 틸트가 무작위 국면보다 통계적으로 우수할 가능성이 있음")
    else:
        print("  판단: 국면 틸트가 무작위 국면 대비 유의하다고 보기 어려움")

    return dict(
        actual=actual,
        sharpes=sharpes,
        mdds=mdds,
        calmars=calmars,
        cagrs=cagrs,
        p_sharpe=p_sharpe,
        p_calmar=p_calmar,
    )


def run_validation(n_placebo=1000):
    print("=" * 70)
    print("KQ 추천 포트폴리오 검증 — IS/OOS + Placebo")
    print("=" * 70)

    print("\n[1] ETF + 매크로 데이터 로드")
    etf_df = load_etf_data()
    import data_loader as dl
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  ETF 기간: {etf_df.index[0].date()} ~ {etf_df.index[-1].date()}")
    print(f"  매크로 분기: {len(features_df)}개")

    print("\n[2] IS 구간에서 코어 3개 포트폴리오 평가")
    core_fns = {
        '동일비중': make_static_fn(STRATEGIES['동일비중']),
        '영구포트폴리오': make_static_fn(STRATEGIES['영구포트폴리오']),
        '올웨더': make_static_fn(STRATEGIES['올웨더']),
    }
    core_is = {
        name: run_portfolio_backtest(etf_df, fn, start_date=IS_START, end_date=IS_END)
        for name, fn in core_fns.items()
    }
    print_table(f"코어 3개 IS 성과 ({IS_START} ~ {IS_END})", core_is)

    is_meta_weights = compute_is_sharpe_meta_weights(core_is)
    operating_base = meta_weights_to_asset_weights(OPERATING_META_WEIGHTS)
    is_base = meta_weights_to_asset_weights(is_meta_weights)

    print("\n[3] 베이스 가중치")
    print("  운영 베이스(현재 추천 탭):")
    for name, weight in OPERATING_META_WEIGHTS.items():
        print(f"    {name:<12} {weight*100:>5.1f}%")
    print("  IS Sharpe 베이스(누수 완화):")
    for name, weight in is_meta_weights.items():
        print(f"    {name:<12} {weight*100:>5.1f}%")

    portfolios = {
        'KODEX 200': make_static_fn(ALLOC_KODEX200),
        '정적 60/40': make_static_fn(ALLOC_60_40),
        '영구포트폴리오': make_static_fn(STRATEGIES['영구포트폴리오']),
        '올웨더': make_static_fn(STRATEGIES['올웨더']),
        '동일비중': make_static_fn(STRATEGIES['동일비중']),
        '운영 베이스 40/35/25': make_static_fn(operating_base),
        'IS Sharpe 베이스': make_static_fn(is_base),
        '추천: 운영베이스+국면틸트': make_recommend_fn(operating_base, features_df, use_regime=True),
        '추천: IS베이스+국면틸트': make_recommend_fn(is_base, features_df, use_regime=True),
        '추천: IS베이스+국면+로보프록시': make_recommend_fn(is_base, features_df, use_regime=True, use_signal=True),
    }

    print("\n[4] IS/OOS 백테스트")
    is_results, oos_results = {}, {}
    for name, fn in portfolios.items():
        print(f"  실행: {name}")
        is_results[name] = run_portfolio_backtest(etf_df, fn, start_date=IS_START, end_date=IS_END)
        oos_results[name] = run_portfolio_backtest(etf_df, fn, start_date=OOS_START)

    print_table(f"IS 성과 ({IS_START} ~ {IS_END})", is_results)
    print_table(f"OOS 성과 ({OOS_START} 이후)", oos_results)

    print("\n[5] OOS 핵심 비교")
    compare_names = [
        '운영 베이스 40/35/25',
        '추천: 운영베이스+국면틸트',
        'IS Sharpe 베이스',
        '추천: IS베이스+국면틸트',
        '추천: IS베이스+국면+로보프록시',
    ]
    for name in compare_names:
        result = oos_results.get(name, {})
        if 'error' in result:
            continue
        print(f"  {name:<28} Sharpe {result['sharpe']:>6.3f}, "
              f"MDD {result['mdd']:>7.2f}%, Calmar {result['calmar']:>6.3f}")

    print("\n[6] Placebo 검증: IS베이스+국면틸트")
    placebo = run_placebo(etf_df, features_df, is_base, n_placebo=n_placebo)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'recommend_portfolio_results.npz')
    save_kwargs = {
        'placebo_sharpes': placebo['sharpes'],
        'placebo_mdds': placebo['mdds'],
        'placebo_calmars': placebo['calmars'],
        'p_sharpe': placebo['p_sharpe'],
        'p_calmar': placebo['p_calmar'],
    }
    for name, result in oos_results.items():
        if 'error' in result:
            continue
        key = name.replace(' ', '_').replace(':', '').replace('/', '_')
        save_kwargs[f'OOS_{key}_sharpe'] = result['sharpe']
        save_kwargs[f'OOS_{key}_mdd'] = result['mdd']
        save_kwargs[f'OOS_{key}_calmar'] = result['calmar']
    np.savez(save_path, **save_kwargs)
    print(f"\n결과 저장: {save_path}")

    print("\n최종 해석 기준:")
    print("  1) 추천 국면틸트가 베이스보다 Sharpe/Calmar를 높이고 MDD를 낮추는지")
    print("  2) Placebo p값이 0.05 이하인지")
    print("  3) 그렇지 않다면 추천 탭은 초과수익 엔진이 아니라 투명한 의사결정 리포트로 표현")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000, help='Placebo 반복 횟수')
    args = parser.parse_args()
    run_validation(n_placebo=args.n)
