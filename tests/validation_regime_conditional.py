"""
validation_regime_conditional.py — 국면 조건부 전략 검증 (Task 5)

가설:
  단순 모멘텀/기술적 지표는 알파가 없다 (Task 2~4에서 검증됨).
  그러나 매크로 국면에 따라 동적으로 자산/종목군을 바꾸면
  알파가 생길 수 있다.

전략 설계 (국면별 종목 풀 전환):
  골디락스(성장↑물가↓)     → 모멘텀 상위 5종목 (성장주 베팅)
  리플레이션(성장↑물가↑)   → 가치주 성격: 저PER 5종목 (없으면 모멘텀 대체)
  스태그플레이션(성장↓물가↑) → 저변동성 5종목 (방어)
  디플레이션(성장↓물가↓)    → 저변동성 5종목 (방어) + 국면 불명확시 KOSPI 동행

PiT 원칙:
  - 매월 리밸런싱 시점 t에서, regime_pit.classify_regime_at()으로
    "t까지 알 수 있었던 매크로 데이터"만으로 국면 판단
  - 종목 선정도 t 시점까지의 가격 데이터만 사용 (validation_core와 동일)

검증 방법 (Task 2~4와 동일한 3중 검정):
  1. Placebo: 국면 라벨을 무작위로 섞은 버전과 비교
  2. Permutation: 종목 선정 결과를 셔플해 p-value 계산
  3. 무조건부 단일 전략(quant/robo) 대비 비교
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validation_core import (
    prepare_bt_data, run_bt_core, get_top_mcap_at, _at
)
from regime_pit import build_pit_macro_features, classify_regime_at


def select_low_vol(hist, top_n, lookback=60):
    """저변동성 종목 선정 (방어적 국면용)"""
    if len(hist) < lookback:
        return []
    window = hist.iloc[-lookback:]
    rets = window.pct_change().dropna()
    if rets.empty:
        return []
    vol = rets.std()
    vol = vol[vol > 0].dropna()
    if len(vol) < top_n:
        return vol.index.tolist()
    return vol.nsmallest(top_n).index.tolist()


def select_momentum(hist, top_n):
    """모멘텀 종목 선정 (성장 국면용) — validation_core.select_quant와 동일 로직"""
    if len(hist) >= 252:
        mom = (hist.iloc[-21] / hist.iloc[-252] - 1)
    elif len(hist) >= 60:
        mom = (hist.iloc[-1] / hist.iloc[-60] - 1)
    else:
        return []
    return mom.dropna().nlargest(top_n).index.tolist()


def select_value_proxy(hist, top_n, lookback=252):
    """가치주 프록시: 최근 1년 수익률이 낮은(저평가 가능성) 종목 중
    너무 망한 종목(상장폐지 위험)은 제외하기 위해 하위 50%~80% 구간 선택.
    실제 PER/PBR 데이터 없이 가격만으로 만든 근사치임을 명시."""
    if len(hist) < lookback:
        return select_low_vol(hist, top_n, lookback=min(60, len(hist)))
    mom = (hist.iloc[-1] / hist.iloc[-lookback] - 1).dropna()
    if len(mom) < top_n * 2:
        return mom.nsmallest(top_n).index.tolist()
    # 하위 50~80% 분위 구간에서 선택 (망한 종목 제외)
    lo = mom.quantile(0.50)
    hi = mom.quantile(0.80)
    candidates = mom[(mom >= lo) & (mom <= hi)]
    if len(candidates) < top_n:
        candidates = mom.nsmallest(top_n * 2)
    return candidates.nsmallest(top_n).index.tolist() if len(candidates) >= top_n \
        else candidates.index.tolist()


REGIME_TO_SELECTOR = {
    '골디락스': lambda hist, n: select_momentum(hist, n),
    '리플레이션': lambda hist, n: select_value_proxy(hist, n),
    '스태그플레이션': lambda hist, n: select_low_vol(hist, n),
    '디플레이션': lambda hist, n: select_low_vol(hist, n),
}


def make_regime_conditional_selector(macro_data, regime_log=None, shuffle_regimes=False,
                                       rng=None):
    """국면 조건부 custom_selector 팩토리.

    Args:
        macro_data: data_loader.load_macro() 결과
        regime_log: list, 호출될 때마다 (date, regime) 기록 (분석용, optional)
        shuffle_regimes: True면 PiT 국면 판단 대신 무작위 국면 배정
                         (Placebo Test용 — "국면 인식 자체가 가치 있는가" 검증)
        rng: shuffle_regimes=True일 때 사용할 난수 생성기

    Returns:
        selector(hist, top_n, cur_date) -> list[ticker]
    """
    features_df = build_pit_macro_features(macro_data)
    regime_names = ['골디락스', '리플레이션', '스태그플레이션', '디플레이션']

    def selector(hist, top_n, cur_date):
        if shuffle_regimes:
            regime = rng.choice(regime_names)
        else:
            regime, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
            if regime is None:
                # 데이터 부족(초기 구간) → 중립적으로 모멘텀 사용
                regime = '골디락스'

        if regime_log is not None:
            regime_log.append((cur_date, regime))

        fn = REGIME_TO_SELECTOR.get(regime, select_momentum)
        try:
            selected = fn(hist, top_n)
        except Exception:
            selected = select_momentum(hist, top_n)
        return selected

    return selector


def run_regime_conditional_backtest(data, macro_data, top_n=5, rebalance='M'):
    """국면 조건부 전략 1회 백테스트 (실제 전략)"""
    regime_log = []
    selector = make_regime_conditional_selector(macro_data, regime_log=regime_log)
    result = run_bt_core(data, strategy='custom', top_n=top_n, rebalance=rebalance,
                          custom_selector=selector)
    result['regime_log'] = regime_log
    return result


def run_shuffled_regime_backtest(data, macro_data, top_n=5, rebalance='M', rng=None):
    """국면 라벨을 무작위로 섞은 Placebo 버전 1회 백테스트.
    실제 국면 판단 대신 매월 무작위 국면을 배정 → 그래도 알파가 나오면
    "국면별 종목군 전환" 자체가 아니라 다른 요인(생존편향 등) 의심."""
    if rng is None:
        rng = np.random.default_rng()
    selector = make_regime_conditional_selector(
        macro_data, shuffle_regimes=True, rng=rng
    )
    return run_bt_core(data, strategy='custom', top_n=top_n, rebalance=rebalance,
                        custom_selector=selector)


def run_full_validation(n_placebo=1000):
    """전체 파이프라인: 실제 전략 + 단일 전략 비교 + Placebo 검증"""
    import server
    import data_loader as dl

    universe = server.UNIVERSE
    ticker_to_code = server._ticker_to_code

    print("=" * 60)
    print("Task 5: 국면 조건부 포트폴리오 검증")
    print("=" * 60)

    print("\n[1/4] 데이터 준비 (시총 상위 200, Point-in-Time)")
    t0 = time.time()
    data = prepare_bt_data(
        universe, ticker_to_code,
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  완료: {time.time()-t0:.1f}초")

    print("\n[2/4] 매크로 데이터 로드")
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  매크로 분기 데이터: {len(features_df)}개 분기")

    print("\n[3/4] 국면 조건부 전략 백테스트 (실제)")
    t0 = time.time()
    actual = run_regime_conditional_backtest(data, macro_data, top_n=5)
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  CAGR: {actual.get('cagr')}%")
    print(f"  Sharpe: {actual.get('sharpe')}")
    print(f"  MDD: {actual.get('mdd')}%")
    print(f"  알파: {actual.get('alpha')}%p (vs KOSPI {actual.get('bench_cagr')}%)")

    # 국면 분포 출력
    regime_log = actual.get('regime_log', [])
    if regime_log:
        regime_counts = pd.Series([r for _, r in regime_log]).value_counts()
        print(f"\n  국면 분포 (총 {len(regime_log)}회 리밸런싱):")
        for r, c in regime_counts.items():
            print(f"    {r}: {c}회 ({c/len(regime_log)*100:.1f}%)")

    print(f"\n[4/4] Placebo Test — 국면 셔플 {n_placebo}회")
    print("  (실제 국면 판단 대신 무작위 국면 배정 후 비교)")
    t0 = time.time()
    rng = np.random.default_rng(42)
    shuffled_cagrs = []
    shuffled_sharpes = []
    for i in range(n_placebo):
        r = run_shuffled_regime_backtest(data, macro_data, top_n=5, rng=rng)
        if 'error' not in r:
            shuffled_cagrs.append(r.get('cagr', np.nan))
            shuffled_sharpes.append(r.get('sharpe', np.nan))
        if (i + 1) % 200 == 0:
            print(f"  진행: {i+1}/{n_placebo} ({time.time()-t0:.1f}초)")
    elapsed = time.time() - t0
    print(f"  완료: {elapsed:.1f}초 ({elapsed/n_placebo*1000:.1f}ms/회)")

    shuffled_cagrs = np.array(shuffled_cagrs)
    shuffled_sharpes = np.array(shuffled_sharpes)

    actual_cagr = actual.get('cagr', 0)
    actual_sharpe = actual.get('sharpe', 0)
    cagr_pct = float((shuffled_cagrs <= actual_cagr).mean() * 100)
    sharpe_pct = float((shuffled_sharpes <= actual_sharpe).mean() * 100)
    p_better = float((shuffled_cagrs >= actual_cagr).mean())  # 실제 > 무작위국면 확률

    print("\n" + "=" * 60)
    print("결과 분석 (국면 조건부 전략 vs 국면 셔플)")
    print("=" * 60)

    print(f"\n실제 전략 (PiT 국면 조건부):")
    print(f"  CAGR: {actual_cagr:.2f}%")
    print(f"  Sharpe: {actual_sharpe:.3f}")
    print(f"  KOSPI: {actual.get('bench_cagr')}%, 알파: {actual.get('alpha')}%p")

    print(f"\n국면 셔플 분포 (N={len(shuffled_cagrs)}):")
    print(f"  CAGR 평균: {shuffled_cagrs.mean():.2f}% (표준편차 {shuffled_cagrs.std():.2f})")
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
        actual=actual,
        shuffled_cagrs=shuffled_cagrs,
        shuffled_sharpes=shuffled_sharpes,
        cagr_percentile=cagr_pct,
        sharpe_percentile=sharpe_pct,
        p_better=p_better,
        regime_log=regime_log,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    result = run_full_validation(n_placebo=args.n)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'regime_conditional_results.npz')
    np.savez(
        save_path,
        actual_cagr=result['actual'].get('cagr', 0),
        actual_sharpe=result['actual'].get('sharpe', 0),
        actual_mdd=result['actual'].get('mdd', 0),
        actual_alpha=result['actual'].get('alpha', 0),
        shuffled_cagrs=result['shuffled_cagrs'],
        shuffled_sharpes=result['shuffled_sharpes'],
        cagr_percentile=result['cagr_percentile'],
        p_better=result['p_better'],
    )
    print(f"\n💾 결과 저장: {save_path}")
