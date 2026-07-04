"""
validation_regime_allocation_fixed.py — 사전 고정 룰 기반 국면 자산배분 검증

v2 대비 개선:
  ✅ Phase 2 winner 탐색 제거 (look-ahead bias 차단)
  ✅ 국면별 비중을 사전에 고정 (책 표준 / 한국 룰)
  ✅ IS/OOS 분리 (2014-2020 학습 vs 2021-2026 검증)
  ✅ OOS만으로 Placebo 검증 (실제 평가는 OOS만)

비교 포트폴리오 (6개):
  1. KODEX 200 (벤치마크) - 069500.KS 100%
  2. 정적 60/40 - 주식 60% + 채권 40%
  3. 영구포트폴리오 (본인 STRATEGIES) - 25/25/25/25
  4. 올웨더 (본인 STRATEGIES)
  5. 동일비중 (본인 STRATEGIES)
  6. 국면 조건부 (사전 고정 룰) - 메인 무기

자산:
  ETFs 7종 (본인 server.py와 동일)

PiT 원칙: 매월 t 시점까지 매크로 데이터로만 국면 판단
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server
from server import _gmv_weights, _mdp_weights, _erc_weights, ETFs, STRATEGIES
from regime_pit import build_pit_macro_features, classify_regime_at


# ─────────── 사전 고정 룰 (데이터 보기 전 결정) ───────────

# 자산 → 대표 ETF 매핑
ASSET_TO_ETF = {
    'stock':  '069500.KS',  # KODEX 200
    'bond':   '148070.KS',  # KOSEF 국고채10년
    'gold':   '132030.KS',  # KODEX 골드선물
    'cash':   '153130.KS',  # KODEX 단기채권
}


def assets_to_weights(asset_weights):
    """{'stock':0.7, 'bond':0.2, ...} → {ticker: weight}"""
    return {ASSET_TO_ETF[a]: w for a, w in asset_weights.items() if a in ASSET_TO_ETF}


# 국면별 사전 고정 룰 (학술 표준)
REGIME_ALLOC_TRADITIONAL = {
    '골디락스':       assets_to_weights({'stock': 0.70, 'bond': 0.20, 'gold': 0.05, 'cash': 0.05}),
    '리플레이션':     assets_to_weights({'stock': 0.50, 'bond': 0.10, 'gold': 0.30, 'cash': 0.10}),
    '스태그플레이션': assets_to_weights({'stock': 0.20, 'bond': 0.10, 'gold': 0.50, 'cash': 0.20}),
    '디플레이션':     assets_to_weights({'stock': 0.20, 'bond': 0.60, 'gold': 0.00, 'cash': 0.20}),
}

# 정적 60/40 (전통 분산)
ALLOC_60_40 = assets_to_weights({'stock': 0.60, 'bond': 0.40, 'gold': 0.00, 'cash': 0.00})

# KOSPI 단독 (벤치마크) - 사실 KODEX200이지만 명명상 KOSPI로 표기
ALLOC_KODEX200 = {'069500.KS': 1.0}


# 캐시
CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'cache', 'etf_assets_v2.parquet'
)


def load_etf_data(force_refresh=False):
    cache_path = os.path.abspath(CACHE_FILE)
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.exists(cache_path) and not force_refresh:
        print(f"  ETF 캐시 로드: {cache_path}")
        df = pd.read_parquet(cache_path)
        df.index = pd.to_datetime(df.index)
        return df
    print(f"  yfinance에서 ETF 다운로드...")
    import yfinance as yf
    closes = {}
    for ticker in ETFs:
        try:
            ydf = yf.download(ticker, period='12y', progress=False, auto_adjust=False)
            if isinstance(ydf.columns, pd.MultiIndex):
                ydf.columns = ydf.columns.get_level_values(0)
            closes[ticker] = ydf['Close'].dropna()
        except Exception:
            pass
    df = pd.DataFrame(closes).dropna()
    df.to_parquet(cache_path)
    return df


# ─────────── 백테스트 코어 ───────────

def run_portfolio_backtest(etf_df, get_weights_fn, rebalance='M',
                            start_date=None, end_date=None):
    df = etf_df.copy()
    if start_date is not None:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df.index <= pd.Timestamp(end_date)]
    if df.empty:
        return {'error': '데이터 없음'}

    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance, 'ME')
    months = df.resample(rule).last()

    equity = [100.0]
    eq_dates = [df.index[0]]
    monthly_rets = []
    weights_log = []

    for i in range(len(months) - 1):
        cur_date = months.index[i]
        next_date = months.index[i+1]
        hist = df.loc[:cur_date]
        period_px = df.loc[cur_date:next_date]
        if len(period_px) < 2:
            continue

        try:
            weights = get_weights_fn(cur_date, hist)
        except Exception:
            weights = None
        if weights is None or sum(weights.values()) == 0:
            continue

        port_ret = 0.0
        total_w = 0.0
        for t, w in weights.items():
            if w <= 0 or t not in period_px.columns:
                continue
            col = period_px[t].dropna()
            if len(col) < 2:
                continue
            p0, p1 = float(col.iloc[0]), float(col.iloc[-1])
            if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
                continue
            r = (p1 - p0) / p0
            port_ret += w * r
            total_w += w

        if total_w == 0:
            continue
        if abs(total_w - 1.0) > 0.01:
            port_ret = port_ret / total_w

        equity.append(equity[-1] * (1 + port_ret))
        eq_dates.append(next_date)
        monthly_rets.append(port_ret)
        weights_log.append({'date': cur_date, **weights})

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))
    return _compute_metrics(eq_series, monthly_rets, weights_log)


def _compute_metrics(eq_series, monthly_rets=None, weights_log=None, freq=12):
    n = len(eq_series) - 1
    if n < 3:
        return {'error': '데이터 부족', 'equity': eq_series}
    years = n / freq
    total = float(eq_series.iloc[-1] / eq_series.iloc[0])
    cagr = (total ** (1/years) - 1) if years > 0 else 0
    rets = eq_series.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    mdd = float(dd.min())
    calmar = (cagr / abs(mdd)) if mdd != 0 else 0
    return dict(
        equity=eq_series,
        cagr=round(cagr * 100, 2),
        vol=round(vol * 100, 2),
        sharpe=round(sharpe, 3),
        mdd=round(mdd * 100, 2),
        calmar=round(calmar, 3),
        n_periods=n,
        monthly_rets=monthly_rets or [],
        weights_log=weights_log or [],
    )


# ─────────── 6개 포트폴리오 weight function ───────────

def make_static_fn(weights):
    def get_weights(cur_date, hist):
        return weights
    return get_weights


def make_regime_fn(regime_alloc, features_df, log=None, shuffle_rng=None):
    """국면 조건부 자산배분 (사전 고정 룰)"""
    regime_names = list(regime_alloc.keys())

    def get_weights(cur_date, hist):
        if shuffle_rng is not None:
            regime = shuffle_rng.choice(regime_names)
        else:
            r, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
            regime = r if r else '골디락스'
        if log is not None:
            log.append({'date': cur_date, 'regime': regime})
        return regime_alloc.get(regime)

    return get_weights


# ─────────── 모든 포트폴리오 6개 ───────────

def get_all_portfolios(features_df):
    """6개 포트폴리오의 weight function 반환"""
    return {
        'KODEX 200 (벤치마크)': make_static_fn(ALLOC_KODEX200),
        '정적 60/40':           make_static_fn(ALLOC_60_40),
        '영구포트폴리오':       make_static_fn(STRATEGIES['영구포트폴리오']),
        '올웨더':               make_static_fn(STRATEGIES['올웨더']),
        '동일비중':             make_static_fn(STRATEGIES['동일비중']),
        '국면 조건부 (전통룰)': make_regime_fn(REGIME_ALLOC_TRADITIONAL, features_df),
    }


# ─────────── IS/OOS 분리 검증 ───────────

def run_is_oos_validation(etf_df, features_df,
                          is_start='2014-06-26', is_end='2020-12-31',
                          oos_start='2021-01-01', oos_end=None):
    """IS/OOS 분리 검증"""
    print("=" * 60)
    print(f"IS/OOS 분리 검증")
    print(f"  IS:  {is_start} ~ {is_end}")
    print(f"  OOS: {oos_start} ~ {oos_end or '데이터 끝'}")
    print("=" * 60)

    portfolios = get_all_portfolios(features_df)

    is_results = {}
    oos_results = {}

    for name, fn in portfolios.items():
        print(f"  실행: {name}")
        # IS
        r_is = run_portfolio_backtest(etf_df, fn, start_date=is_start, end_date=is_end)
        is_results[name] = r_is
        # OOS
        r_oos = run_portfolio_backtest(etf_df, fn, start_date=oos_start, end_date=oos_end)
        oos_results[name] = r_oos

    # 결과 표
    print("\n" + "-" * 80)
    print(f"IS 성과 ({is_start} ~ {is_end})")
    print("-" * 80)
    print(f"{'포트폴리오':<25} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Vol':>8}")
    for name, r in is_results.items():
        if 'error' in r:
            print(f"{name:<25} ❌")
            continue
        print(f"{name:<25} {r['cagr']:>7.2f}% {r['sharpe']:>8.3f} "
              f"{r['mdd']:>7.2f}% {r['calmar']:>8.3f} {r['vol']:>7.2f}%")

    print("\n" + "-" * 80)
    print(f"OOS 성과 ({oos_start} 이후) — 실제 평가 구간")
    print("-" * 80)
    print(f"{'포트폴리오':<25} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Vol':>8}")
    for name, r in oos_results.items():
        if 'error' in r:
            print(f"{name:<25} ❌")
            continue
        print(f"{name:<25} {r['cagr']:>7.2f}% {r['sharpe']:>8.3f} "
              f"{r['mdd']:>7.2f}% {r['calmar']:>8.3f} {r['vol']:>7.2f}%")

    # OOS 순위 (이게 진짜 평가)
    print("\nOOS 기준 순위:")
    valid = [(n, r) for n, r in oos_results.items() if 'error' not in r]
    if valid:
        print(f"  🏆 Sharpe 최고: {max(valid, key=lambda x: x[1]['sharpe'])[0]}")
        print(f"  🏆 MDD 최저:    {max(valid, key=lambda x: x[1]['mdd'])[0]}")
        print(f"  🏆 Calmar 최고: {max(valid, key=lambda x: x[1]['calmar'])[0]}")

    # IS → OOS 격차
    print("\nIS/OOS 격차 (과적합 검증):")
    for name in is_results.keys():
        if 'error' in is_results[name] or 'error' in oos_results[name]:
            continue
        is_s = is_results[name]['sharpe']
        oos_s = oos_results[name]['sharpe']
        gap = oos_s - is_s
        marker = "🟢" if abs(gap) < 0.3 else "🟡" if abs(gap) < 0.8 else "🔴"
        print(f"  {name:<25} IS Sharpe {is_s:>6.3f} → OOS {oos_s:>6.3f} (Δ{gap:+.3f}) {marker}")

    return is_results, oos_results


# ─────────── OOS Placebo Test ───────────

def run_oos_placebo(etf_df, features_df, n_placebo=1000,
                    oos_start='2021-01-01', oos_end=None):
    """OOS 구간에서만 국면 셔플 Placebo 검증"""
    print("\n" + "=" * 60)
    print(f"OOS Placebo Test ({n_placebo}회) — 국면 셔플")
    print("=" * 60)

    # 실제 OOS
    regime_log = []
    actual_fn = make_regime_fn(REGIME_ALLOC_TRADITIONAL, features_df, log=regime_log)
    r_actual = run_portfolio_backtest(
        etf_df, actual_fn, start_date=oos_start, end_date=oos_end
    )
    if 'error' in r_actual:
        print(f"  ❌ {r_actual['error']}")
        return None

    print(f"\n실제 PiT 국면 (OOS):")
    print(f"  CAGR: {r_actual['cagr']:.2f}%, Sharpe: {r_actual['sharpe']:.3f}, "
          f"MDD: {r_actual['mdd']:.2f}%, Calmar: {r_actual['calmar']:.3f}")

    # OOS 국면 분포
    regime_counts = pd.Series([r['regime'] for r in regime_log]).value_counts()
    print(f"\nOOS 국면 분포 ({len(regime_log)}회):")
    for r, c in regime_counts.items():
        print(f"  {r}: {c}회 ({c/len(regime_log)*100:.1f}%)")

    # 셔플 1000회
    print(f"\n국면 셔플 {n_placebo}회 시작...")
    t0 = time.time()
    rng = np.random.default_rng(42)
    sharpes, mdds, calmars, cagrs = [], [], [], []

    for i in range(n_placebo):
        sub_rng = np.random.default_rng(rng.integers(0, 1e9))
        shuf_fn = make_regime_fn(REGIME_ALLOC_TRADITIONAL, features_df, shuffle_rng=sub_rng)
        r = run_portfolio_backtest(
            etf_df, shuf_fn, start_date=oos_start, end_date=oos_end
        )
        if 'error' not in r:
            cagrs.append(r['cagr'])
            sharpes.append(r['sharpe'])
            mdds.append(r['mdd'])
            calmars.append(r['calmar'])
        if (i+1) % 200 == 0:
            print(f"  진행: {i+1}/{n_placebo} ({time.time()-t0:.1f}초)")

    sharpes = np.array(sharpes)
    mdds = np.array(mdds)
    calmars = np.array(calmars)

    print(f"  완료: {time.time()-t0:.1f}초")

    # 분석
    p_sharpe = float((sharpes >= r_actual['sharpe']).mean())
    p_calmar = float((calmars >= r_actual['calmar']).mean())
    sharpe_pct = float((sharpes <= r_actual['sharpe']).mean() * 100)
    mdd_pct = float((mdds <= r_actual['mdd']).mean() * 100)
    calmar_pct = float((calmars <= r_actual['calmar']).mean() * 100)

    print("\n" + "=" * 60)
    print("OOS Placebo 결과")
    print("=" * 60)
    print(f"\n  Sharpe — 실제 {r_actual['sharpe']:.3f} vs 셔플평균 {sharpes.mean():.3f} "
          f"(percentile={sharpe_pct:.1f}%, p={p_sharpe:.3f})")
    print(f"  MDD    — 실제 {r_actual['mdd']:.2f}% vs 셔플평균 {mdds.mean():.2f}% "
          f"(percentile={mdd_pct:.1f}%)")
    print(f"  Calmar — 실제 {r_actual['calmar']:.3f} vs 셔플평균 {calmars.mean():.3f} "
          f"(percentile={calmar_pct:.1f}%, p={p_calmar:.3f})")

    print(f"\n해석:")
    if p_sharpe < 0.05:
        print(f"  ✅ Sharpe — PiT 국면 판단이 OOS에서 통계적 유의 (p={p_sharpe:.3f})")
    elif sharpe_pct > 90:
        print(f"  🟡 Sharpe — 상위 10% (약하지만 유의 미달)")
    else:
        print(f"  ⚪ Sharpe — 무작위 국면과 차이 없음")
    if p_calmar < 0.05:
        print(f"  ✅ Calmar — PiT 국면 판단이 OOS에서 통계적 유의 (p={p_calmar:.3f})")
    elif calmar_pct > 90:
        print(f"  🟡 Calmar — 상위 10%")
    else:
        print(f"  ⚪ Calmar — 무작위 국면과 차이 없음")

    return dict(
        actual=r_actual,
        sharpes=sharpes,
        mdds=mdds,
        calmars=calmars,
        p_sharpe=p_sharpe,
        p_calmar=p_calmar,
    )


# ─────────── 전체 파이프라인 ───────────

def run_full_validation(n_placebo=1000):
    print("=" * 60)
    print("KQ Quant Tool — 사전 고정 룰 + IS/OOS 검증")
    print("=" * 60)

    print("\n[1] ETF 데이터 로드")
    etf_df = load_etf_data()
    print(f"  기간: {etf_df.index[0].date()} ~ {etf_df.index[-1].date()}")

    print("\n[2] PiT 매크로 국면")
    import data_loader as dl
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  매크로 분기: {len(features_df)}개")

    print("\n[3] IS/OOS 분리 백테스트")
    is_results, oos_results = run_is_oos_validation(etf_df, features_df)

    print("\n[4] OOS Placebo 검증")
    placebo = run_oos_placebo(etf_df, features_df, n_placebo=n_placebo)

    # 종합 해석
    print("\n" + "=" * 60)
    print("최종 종합 해석")
    print("=" * 60)

    regime_name = '국면 조건부 (전통룰)'
    if regime_name in oos_results and 'error' not in oos_results[regime_name]:
        r_regime = oos_results[regime_name]
        # OOS에서 정적 60/40과 비교
        r_6040 = oos_results.get('정적 60/40')
        r_kospi = oos_results.get('KODEX 200 (벤치마크)')

        print(f"\n[OOS 2021-2026] 국면 조건부 vs 정적 비교:")
        print(f"  {regime_name:<25} Sharpe {r_regime['sharpe']:>6.3f}, MDD {r_regime['mdd']:>7.2f}%, Calmar {r_regime['calmar']:>6.3f}")
        if r_6040 and 'error' not in r_6040:
            print(f"  {'정적 60/40':<25} Sharpe {r_6040['sharpe']:>6.3f}, MDD {r_6040['mdd']:>7.2f}%, Calmar {r_6040['calmar']:>6.3f}")
        if r_kospi and 'error' not in r_kospi:
            print(f"  {'KODEX 200':<25} Sharpe {r_kospi['sharpe']:>6.3f}, MDD {r_kospi['mdd']:>7.2f}%, Calmar {r_kospi['calmar']:>6.3f}")

    # 저장
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'asset_allocation_fixed_results.npz'
    )
    save_kwargs = {}
    for name, r in is_results.items():
        if 'error' in r:
            continue
        key = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        save_kwargs[f'IS_{key}_sharpe'] = r['sharpe']
        save_kwargs[f'IS_{key}_mdd'] = r['mdd']
        save_kwargs[f'IS_{key}_calmar'] = r['calmar']
    for name, r in oos_results.items():
        if 'error' in r:
            continue
        key = name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        save_kwargs[f'OOS_{key}_sharpe'] = r['sharpe']
        save_kwargs[f'OOS_{key}_mdd'] = r['mdd']
        save_kwargs[f'OOS_{key}_calmar'] = r['calmar']
    if placebo:
        save_kwargs['placebo_sharpes'] = placebo['sharpes']
        save_kwargs['placebo_mdds'] = placebo['mdds']
        save_kwargs['placebo_calmars'] = placebo['calmars']
        save_kwargs['p_sharpe'] = placebo['p_sharpe']
        save_kwargs['p_calmar'] = placebo['p_calmar']
    np.savez(save_path, **save_kwargs)
    print(f"\n💾 결과 저장: {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    run_full_validation(n_placebo=args.n)
