"""
validation_regime_conditional_fast.py — 국면 조건부 전략 검증 (벡터화 버전)

원본 validation_regime_conditional.py와 동일한 검증 로직이지만 1000회 셔플 속도 최적화.

핵심 최적화:
  원본: 1000회 × 144개월 × (저변동성 200종목 std 계산 + 모멘텀 + 가치프록시)
        = 약 30분 소요

  Fast: 한 번만 매월 3가지 전략(모멘텀/저변동성/가치프록시)의 그 달 수익률을
        미리 계산해서 저장 → 셔플은 lookup만 → 약 1~2분 소요

기댓값:
  - 실제 결과는 원본과 100% 동일 (수식 같음)
  - 1000회 셔플이 ~30분 → ~1분으로 단축
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation_core import prepare_bt_data, get_top_mcap_at
from regime_pit import build_pit_macro_features, classify_regime_at
from validation_regime_conditional import (
    select_low_vol, select_momentum, select_value_proxy
)


# 전략 이름과 함수 매핑
STRATEGY_FNS = {
    'momentum': select_momentum,
    'low_vol': select_low_vol,
    'value': select_value_proxy,
}

REGIME_TO_STRATEGY = {
    '골디락스': 'momentum',
    '리플레이션': 'value',
    '스태그플레이션': 'low_vol',
    '디플레이션': 'low_vol',
}


def precompute_monthly_strategy_returns(data, top_n=5, rebalance='M'):
    """매월 리밸런싱 시점마다 3가지 전략(모멘텀/저변동성/가치프록시)의
    다음 달 수익률을 미리 계산.

    Returns:
        DataFrame, index=리밸런싱 날짜, columns=['momentum', 'low_vol', 'value']
        각 셀: 그 전략으로 5종목 선정 → 다음 달 동일가중 수익률
    """
    price_df = data['price_df']
    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance, 'ME')
    months = price_df.resample(rule).last()

    # PiT 시총 필터
    pit_mode = data.get('point_in_time', False)
    pit_top_n = data.get('top_n_mcap', 200)
    mcap_hist = data.get('mcap_history')

    rows = []
    for i in range(len(months) - 1):
        cur_date = months.index[i]
        next_date = months.index[i+1]

        hist_full = price_df.loc[:cur_date]
        if len(hist_full) < 60:
            continue

        # PiT 시총 필터링
        if pit_mode and mcap_hist is not None:
            top_tickers_pit = get_top_mcap_at(mcap_hist, cur_date, pit_top_n)
            available = [t for t in top_tickers_pit if t in hist_full.columns]
            if len(available) < top_n:
                continue
            hist = hist_full[available]
        else:
            hist = hist_full

        # 다음 기간 수익률 데이터
        period_px = price_df.loc[cur_date:next_date]
        if len(period_px) < 2:
            continue

        # 3가지 전략별 종목 선정 + 그 달 수익률
        row_data = {'date': cur_date}
        for strat_name, strat_fn in STRATEGY_FNS.items():
            try:
                selected = strat_fn(hist, top_n)
            except Exception:
                selected = []
            if not selected:
                row_data[strat_name] = np.nan
                continue

            w = 1.0 / len(selected)
            period_ret = 0.0
            valid_count = 0
            for t in selected:
                if t not in period_px.columns:
                    continue
                col = period_px[t].dropna()
                if len(col) < 2:
                    continue
                p0, p1 = float(col.iloc[0]), float(col.iloc[-1])
                if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
                    continue
                r = (p1 - p0) / p0
                period_ret += w * r
                valid_count += 1

            if valid_count == 0:
                row_data[strat_name] = np.nan
            else:
                if valid_count < len(selected):
                    period_ret = period_ret * len(selected) / valid_count
                row_data[strat_name] = period_ret

        rows.append(row_data)

    df = pd.DataFrame(rows).set_index('date')
    return df


def compute_metrics_from_returns(monthly_rets, freq=12):
    """월 수익률 시리즈로부터 CAGR/Sharpe/MDD 계산"""
    rets = pd.Series(monthly_rets).dropna()
    if len(rets) < 3:
        return dict(cagr=0, sharpe=0, mdd=0, total=0)
    equity = (1 + rets).cumprod()
    n = len(rets)
    years = n / freq
    total = float(equity.iloc[-1])
    cagr = (total ** (1/years) - 1) if years > 0 else 0
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0
    peak = equity.cummax()
    mdd = float(((equity - peak) / peak).min())
    return dict(
        cagr=round(cagr * 100, 2),
        sharpe=round(sharpe, 3),
        mdd=round(mdd * 100, 2),
        total=round((total - 1) * 100, 2),
    )


def run_full_validation_fast(n_placebo=1000):
    """벡터화된 전체 검증 파이프라인"""
    import server
    import data_loader as dl

    universe = server.UNIVERSE
    ticker_to_code = server._ticker_to_code

    print("=" * 60)
    print("Task 5: 국면 조건부 포트폴리오 검증 (Fast)")
    print("=" * 60)

    print("\n[1/4] 데이터 준비 (시총 상위 200, PiT)")
    t0 = time.time()
    data = prepare_bt_data(
        universe, ticker_to_code,
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  완료: {time.time()-t0:.1f}초")

    print("\n[2/4] 매크로 + PiT 국면 특징 계산")
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  매크로 분기 데이터: {len(features_df)}개 분기")

    print("\n[3/4] 매월 3가지 전략 수익률 사전계산 (1회만)")
    t0 = time.time()
    monthly_strategy_rets = precompute_monthly_strategy_returns(data, top_n=5)
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  사전계산 결과: {monthly_strategy_rets.shape}")
    print(f"  전략별 평균 월 수익률:")
    for col in monthly_strategy_rets.columns:
        v = monthly_strategy_rets[col].dropna()
        print(f"    {col}: {v.mean()*100:.2f}% (n={len(v)})")

    # === 실제 전략: PiT 국면 판단 + 해당 전략 lookup ===
    print(f"\n[4a/4] 실제 PiT 국면 전략 (lookup 기반)")
    actual_rets = []
    regime_log = []
    for cur_date in monthly_strategy_rets.index:
        regime, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
        if regime is None:
            regime = '골디락스'  # 초기 구간 fallback
        strat_name = REGIME_TO_STRATEGY.get(regime, 'momentum')
        ret = monthly_strategy_rets.loc[cur_date, strat_name]
        actual_rets.append(ret)
        regime_log.append((cur_date, regime, strat_name))

    actual_metrics = compute_metrics_from_returns(actual_rets)
    print(f"  CAGR: {actual_metrics['cagr']}%")
    print(f"  Sharpe: {actual_metrics['sharpe']}")
    print(f"  MDD: {actual_metrics['mdd']}%")

    # 국면 분포
    regime_counts = pd.Series([r for _, r, _ in regime_log]).value_counts()
    print(f"\n  국면 분포 (총 {len(regime_log)}회 리밸런싱):")
    for r, c in regime_counts.items():
        print(f"    {r}: {c}회 ({c/len(regime_log)*100:.1f}%)")

    # === KOSPI 벤치마크 ===
    kospi = data['kospi']
    if kospi is not None and len(kospi) > 30:
        # 월말 수익률
        kospi_monthly = kospi.resample('ME').last().pct_change().dropna()
        kospi_aligned = kospi_monthly.reindex(monthly_strategy_rets.index, method='ffill').dropna()
        kospi_metrics = compute_metrics_from_returns(kospi_aligned)
        bench_cagr = kospi_metrics['cagr']
        alpha = round(actual_metrics['cagr'] - bench_cagr, 2)
        print(f"\n  KOSPI CAGR: {bench_cagr}%, 알파: {alpha}%p")
    else:
        bench_cagr = None
        alpha = None

    # === Placebo: 국면 셔플 1000회 ===
    print(f"\n[4b/4] Placebo Test — 국면 셔플 {n_placebo}회 (벡터화)")
    t0 = time.time()
    rng = np.random.default_rng(42)
    strategies = ['momentum', 'low_vol', 'value']
    n_months = len(monthly_strategy_rets)

    # 행렬 형태로 변환 (lookup 빠르게)
    strat_matrix = monthly_strategy_rets[strategies].values  # (n_months, 3)
    strat_to_idx = {s: i for i, s in enumerate(strategies)}

    shuffled_cagrs = np.zeros(n_placebo)
    shuffled_sharpes = np.zeros(n_placebo)
    shuffled_mdds = np.zeros(n_placebo)
    regime_to_strat_idx = np.array([
        strat_to_idx[REGIME_TO_STRATEGY[r]]
        for r in ['골디락스', '리플레이션', '스태그플레이션', '디플레이션']
    ])

    for i in range(n_placebo):
        # 매월 무작위 국면 → 그에 대응하는 전략 인덱스 → 해당 수익률
        random_regimes = rng.integers(0, 4, size=n_months)
        random_strat_idx = regime_to_strat_idx[random_regimes]
        sim_rets = strat_matrix[np.arange(n_months), random_strat_idx]
        m = compute_metrics_from_returns(sim_rets)
        shuffled_cagrs[i] = m['cagr']
        shuffled_sharpes[i] = m['sharpe']
        shuffled_mdds[i] = m['mdd']

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  진행: {i+1}/{n_placebo} ({elapsed:.1f}초)")

    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.1f}초 ({elapsed/n_placebo*1000:.2f}ms/회)")

    # === 결과 분석 ===
    actual_cagr = actual_metrics['cagr']
    actual_sharpe = actual_metrics['sharpe']
    cagr_pct = float((shuffled_cagrs <= actual_cagr).mean() * 100)
    sharpe_pct = float((shuffled_sharpes <= actual_sharpe).mean() * 100)
    p_better = float((shuffled_cagrs >= actual_cagr).mean())

    print("\n" + "=" * 60)
    print("결과 분석 (국면 조건부 전략 vs 국면 셔플)")
    print("=" * 60)

    print(f"\n실제 전략 (PiT 국면 조건부):")
    print(f"  CAGR: {actual_cagr:.2f}%")
    print(f"  Sharpe: {actual_sharpe:.3f}")
    print(f"  MDD: {actual_metrics['mdd']}%")
    if bench_cagr is not None:
        print(f"  KOSPI: {bench_cagr}%, 알파: {alpha}%p")

    print(f"\n국면 셔플 분포 (N={n_placebo}):")
    print(f"  CAGR 평균: {shuffled_cagrs.mean():.2f}% (표준편차 {shuffled_cagrs.std():.2f})")
    print(f"  CAGR 중앙값: {np.median(shuffled_cagrs):.2f}%")
    print(f"  CAGR 5~95% 분위: [{np.percentile(shuffled_cagrs, 5):.2f}, "
          f"{np.percentile(shuffled_cagrs, 95):.2f}]")

    print(f"\n실제 vs 셔플:")
    print(f"  CAGR 백분위: {cagr_pct:.1f}%")
    print(f"  Sharpe 백분위: {sharpe_pct:.1f}%")
    print(f"  p-value (국면판단이 셔플보다 나은가): {p_better:.3f}")

    print(f"\n해석:")
    if p_better < 0.05:
        print(f"  ✅ 실제 PiT 국면 판단이 무작위 국면 배정보다 통계적으로 유의하게 좋음 (p={p_better:.3f})")
        print(f"     → 매크로 국면 인식 자체가 가치를 더한다는 증거")
    elif cagr_pct > 70:
        print(f"  🟡 실제 국면 판단이 셔플 분포 상위권 (p={p_better:.3f}, 약하지만 유의하지 않음)")
        print(f"     → 추가 검증(다른 시드, 다른 기간) 필요")
    else:
        print(f"  ⚪ 실제 국면 판단과 무작위 국면 배정 차이 없음 (p={p_better:.3f})")
        print(f"     → 국면별 종목군 전환 자체로는 알파 생성 안 됨")
        print(f"     → 본 결과 자체가 의미 있는 학술적 발견(EMH와 일치)")

    return dict(
        actual_metrics=actual_metrics,
        actual_cagr=actual_cagr,
        actual_sharpe=actual_sharpe,
        actual_mdd=actual_metrics['mdd'],
        bench_cagr=bench_cagr,
        alpha=alpha,
        shuffled_cagrs=shuffled_cagrs,
        shuffled_sharpes=shuffled_sharpes,
        shuffled_mdds=shuffled_mdds,
        cagr_percentile=cagr_pct,
        sharpe_percentile=sharpe_pct,
        p_better=p_better,
        regime_log=regime_log,
        monthly_strategy_rets=monthly_strategy_rets,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    result = run_full_validation_fast(n_placebo=args.n)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'regime_conditional_results.npz')
    save_kwargs = dict(
        actual_cagr=result['actual_cagr'],
        actual_sharpe=result['actual_sharpe'],
        actual_mdd=result['actual_mdd'],
        shuffled_cagrs=result['shuffled_cagrs'],
        shuffled_sharpes=result['shuffled_sharpes'],
        shuffled_mdds=result['shuffled_mdds'],
        cagr_percentile=result['cagr_percentile'],
        p_better=result['p_better'],
    )
    if result['bench_cagr'] is not None:
        save_kwargs['bench_cagr'] = result['bench_cagr']
        save_kwargs['alpha'] = result['alpha']
    np.savez(save_path, **save_kwargs)
    print(f"\n💾 결과 저장: {save_path}")
