"""
validation_asset_allocation_v2.py — 본인 시스템의 10개 포트폴리오 12년 검증

핵심:
  본인 server.py의 10개 포트폴리오 + 위험기반 함수를 그대로 재사용해서
  12년 장기 백테스트 + 국면 조건부 스위칭 검증

설계:
  Phase 1: 10개 포트폴리오 12년 정적 백테스트 (Sharpe/MDD/Calmar)
  Phase 2: 매크로 국면별 성과 분석 (어떤 게 어떤 국면에 강한가)
  Phase 3: 국면 조건부 스위칭 전략 검증
  Phase 4: Placebo 1,000회 — 국면 셔플 vs 실제 PiT

비교 대상:
  - KOSPI (벤치마크)
  - 10개 포트폴리오 (영구포트폴리오/황금나비/올웨더/정적 60/40/동일비중/GTAA/역변동성/GMV/MDP/ERC)
  - 국면 조건부 스위칭 (메인 무기)

자산:
  - 069500.KS (KODEX 200) - 한국 주식
  - 229200.KS (KODEX 코스닥150)
  - 130680.KS (TIGER 미국S&P500)
  - 132030.KS (KODEX 골드선물)
  - 114260.KS (TIGER 단기통안채)
  - 148070.KS (KOSEF 국고채10년)
  - 153130.KS (KODEX 단기채권)
"""
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 상위 폴더 (kq_tool/) 도 path에 추가 - server.py import 위함
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'src'))

# server.py의 핵심 함수/객체 활용
import server
from server import _gmv_weights, _mdp_weights, _erc_weights, ETFs, STRATEGIES

from regime_pit import build_pit_macro_features, classify_regime_at
from kq_tool.data.gateway import get_kospi_benchmark, load_close_panel


# 캐시 경로
CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'data', 'cache', 'etf_assets_v2.parquet'
)


def _ticker_to_code(ticker):
    return str(ticker).split('.', 1)[0]


def load_etf_data(force_refresh=False):
    """ETFs 7종 종가 데이터 - 단일 게이트웨이 수정주가 패널."""
    panel = load_close_panel()
    closes = {}
    for ticker in ETFs:
        code = _ticker_to_code(ticker)
        if code in panel.columns:
            closes[ticker] = pd.to_numeric(panel[code], errors='coerce').dropna()
            print(f"    {ticker}: {len(closes[ticker])}행 (gateway)")
        else:
            print(f"    {ticker}: close.csv에 없음")
    df = pd.DataFrame(closes).dropna(how='all').dropna()
    if df.empty:
        raise RuntimeError('ETF 수정주가 패널 로드 실패')
    return df


def load_kospi_data():
    """KOSPI 지수 데이터 (단일 게이트웨이 ECOS 수집본)."""
    return get_kospi_benchmark()


def compute_dynamic_weights(price_df, strategy_name, lookback_days=252):
    """동적 포트폴리오 비중 계산 (특정 시점 기준)

    Args:
        price_df: 과거 가격 데이터 (해당 시점까지)
        strategy_name: 'GMV', 'MDP', 'ERC', '역변동성', 'GTAA'
        lookback_days: 공분산 계산 윈도우

    Returns:
        dict {ticker: weight}
    """
    if len(price_df) < lookback_days:
        return None

    # 최근 lookback 기간 수익률
    window = price_df.iloc[-lookback_days:]
    rets = window.pct_change().dropna()
    if rets.empty or len(rets.columns) < 2:
        return None

    cov = rets.cov().values * 252  # 연환산 공분산
    vols = np.sqrt(np.diag(cov))
    tickers = list(rets.columns)

    weights = None
    try:
        if strategy_name == 'GMV':
            w = _gmv_weights(cov, cap=0.5)
        elif strategy_name == 'MDP':
            w = _mdp_weights(cov, vols, cap=0.5)
        elif strategy_name == 'ERC':
            w = _erc_weights(cov, cap=0.5)
        elif strategy_name == '역변동성':
            inv_vol = 1.0 / vols
            w = inv_vol / inv_vol.sum()
        elif strategy_name == 'GTAA':
            # GTAA: 12개월 모멘텀 양수인 자산만 동일 비중
            mom = (price_df.iloc[-1] / price_df.iloc[-252] - 1) if len(price_df) >= 252 else None
            if mom is None:
                return None
            positive = mom[mom > 0]
            if len(positive) == 0:
                # 모두 음수면 현금 100%
                w = np.zeros(len(tickers))
                cash_idx = tickers.index('153130.KS') if '153130.KS' in tickers else -1
                if cash_idx >= 0:
                    w[cash_idx] = 1.0
                else:
                    return None
            else:
                w = np.zeros(len(tickers))
                for t in positive.index:
                    if t in tickers:
                        w[tickers.index(t)] = 1.0 / len(positive)
        else:
            return None

        weights = {t: float(w[i]) for i, t in enumerate(tickers)}
    except Exception as e:
        print(f"  ⚠️ {strategy_name} 비중 계산 실패: {e}")
        return None

    return weights


def run_portfolio_backtest(etf_df, get_weights_fn, rebalance='M',
                            start_date=None, end_date=None):
    """포트폴리오 백테스트 코어

    Args:
        etf_df: load_etf_data() 결과
        get_weights_fn: callable(cur_date, hist_df) -> dict {ticker: weight}
        rebalance: 'M'/'Q'
        start_date, end_date: 백테스트 기간

    Returns:
        dict (cagr, sharpe, mdd, calmar, vol, equity, monthly_rets, dates)
    """
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

        # 자산 비중 결정
        try:
            weights = get_weights_fn(cur_date, hist)
        except Exception as e:
            weights = None

        if weights is None or sum(weights.values()) == 0:
            continue

        # 포트폴리오 수익률
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
    """성과 지표 계산"""
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
        total_return=round((total - 1) * 100, 2),
        vol=round(vol * 100, 2),
        sharpe=round(sharpe, 3),
        mdd=round(mdd * 100, 2),
        calmar=round(calmar, 3),
        n_periods=n,
        monthly_rets=monthly_rets or [],
        weights_log=weights_log or [],
    )


# ─────────── 10가지 포트폴리오 weight function 생성기 ───────────

def make_static_weights_fn(strategy_name):
    """고정 비중 포트폴리오: 영구포트폴리오/황금나비/올웨더/정적 60/40/동일비중"""
    fixed_weights = STRATEGIES.get(strategy_name)
    if fixed_weights is None:
        return None

    def get_weights(cur_date, hist):
        return fixed_weights

    return get_weights


def make_dynamic_weights_fn(strategy_name, lookback_days=252):
    """동적 비중 포트폴리오: GMV/MDP/ERC/역변동성/GTAA"""
    def get_weights(cur_date, hist):
        return compute_dynamic_weights(hist, strategy_name, lookback_days)
    return get_weights


# ─────────── Phase 1: 10개 포트폴리오 12년 백테스트 ───────────

def run_phase1_all_portfolios(etf_df, start_date='2014-06-26', end_date=None):
    """10개 포트폴리오 + KOSPI 12년 백테스트"""
    print("\n" + "=" * 60)
    print("Phase 1: 10개 포트폴리오 12년 정적 백테스트")
    print("=" * 60)

    results = {}

    # KOSPI (벤치마크) — 주식 100%
    print("\n실행 중: KOSPI 벤치마크")
    r = run_portfolio_backtest(
        etf_df, lambda d, h: {'069500.KS': 1.0},
        start_date=start_date, end_date=end_date
    )
    results['KOSPI (벤치마크)'] = r

    # 고정 비중 5개
    for name in ['영구포트폴리오', '황금나비', '올웨더', '정적 60/40', '동일비중']:
        print(f"실행 중: {name}")
        fn = make_static_weights_fn(name)
        if fn is None:
            continue
        r = run_portfolio_backtest(etf_df, fn, start_date=start_date, end_date=end_date)
        results[name] = r

    # 동적 5개
    for name in ['역변동성', 'GMV', 'MDP', 'ERC', 'GTAA']:
        print(f"실행 중: {name}")
        fn = make_dynamic_weights_fn(name)
        r = run_portfolio_backtest(etf_df, fn, start_date=start_date, end_date=end_date)
        results[name] = r

    # 결과 표
    print("\n" + "-" * 70)
    print(f"{'포트폴리오':<20} {'CAGR':>8} {'Sharpe':>8} {'MDD':>8} {'Calmar':>8} {'Vol':>8}")
    print("-" * 70)
    for name, r in results.items():
        if 'error' in r:
            print(f"{name:<20} ❌ {r['error']}")
            continue
        print(f"{name:<20} {r['cagr']:>7.2f}% {r['sharpe']:>8.3f} "
              f"{r['mdd']:>7.2f}% {r['calmar']:>8.3f} {r['vol']:>7.2f}%")
    print("-" * 70)

    # 순위
    valid = [(n, r) for n, r in results.items() if 'error' not in r]
    if valid:
        print(f"\n🏆 최고 Sharpe: {max(valid, key=lambda x: x[1]['sharpe'])[0]}")
        print(f"🏆 최저 MDD:    {max(valid, key=lambda x: x[1]['mdd'])[0]}")
        print(f"🏆 최고 Calmar: {max(valid, key=lambda x: x[1]['calmar'])[0]}")

    return results


# ─────────── Phase 2: 매크로 국면별 성과 ───────────

def run_phase2_regime_analysis(results, features_df):
    """각 포트폴리오의 월간 수익률을 매크로 국면별로 묶어서 분석"""
    print("\n" + "=" * 60)
    print("Phase 2: 매크로 국면별 성과 분석")
    print("=" * 60)

    regime_names = ['골디락스', '리플레이션', '스태그플레이션', '디플레이션']
    # 각 포트폴리오 × 국면별 평균 월수익률
    by_regime = {}
    for name, r in results.items():
        if 'error' in r or not r.get('weights_log'):
            continue
        # 각 월의 국면 판단
        dates = [w['date'] for w in r['weights_log']]
        monthly_rets = r['monthly_rets']
        if len(monthly_rets) != len(dates):
            continue

        regime_rets = {rg: [] for rg in regime_names}
        for d, ret in zip(dates, monthly_rets):
            regime, _ = classify_regime_at(features_df, d, min_quarters=4)
            if regime in regime_rets:
                regime_rets[regime].append(ret)

        by_regime[name] = {
            rg: dict(
                n=len(rets),
                mean=np.mean(rets) * 12 * 100 if rets else 0,  # 연환산 %
                std=np.std(rets) * np.sqrt(12) * 100 if rets else 0,
            )
            for rg, rets in regime_rets.items()
        }

    # 표 출력
    print(f"\n각 포트폴리오의 국면별 연환산 평균 수익률 (%):")
    print(f"{'포트폴리오':<20} {'골디락스':>10} {'리플레이션':>12} {'스태그':>10} {'디플레이션':>12}")
    print("-" * 70)
    for name, regime_perf in by_regime.items():
        row = f"{name:<20}"
        for rg in regime_names:
            v = regime_perf[rg]['mean']
            row += f" {v:>10.2f}"
        print(row)

    # 각 국면별 최고 포트폴리오
    print(f"\n각 국면에서 최고 성과 포트폴리오:")
    best_by_regime = {}
    for rg in regime_names:
        ranked = sorted(
            [(name, regime_perf[rg]['mean']) for name, regime_perf in by_regime.items()],
            key=lambda x: x[1],
            reverse=True
        )
        if ranked:
            best_by_regime[rg] = ranked[0]
            print(f"  {rg}: {ranked[0][0]} ({ranked[0][1]:+.2f}%)")

    return by_regime, best_by_regime


# ─────────── Phase 3: 국면 조건부 스위칭 ───────────

def run_phase3_regime_switching(etf_df, features_df, best_by_regime,
                                  start_date='2014-06-26', end_date=None):
    """매월 PiT 국면 판단 → 그 국면에 가장 좋은 포트폴리오 사용"""
    print("\n" + "=" * 60)
    print("Phase 3: 국면 조건부 스위칭 전략")
    print("=" * 60)

    print("\n스위칭 룰 (Phase 2 결과 기반):")
    for rg, (best, ret) in best_by_regime.items():
        print(f"  {rg} → {best} (월평균 {ret/12:.2f}%, 연환산 {ret:.2f}%)")

    # 각 포트폴리오의 weight 계산 함수 미리 만들어두기
    weight_fns = {}
    weight_fns['KOSPI (벤치마크)'] = lambda d, h: {'069500.KS': 1.0}
    for name in ['영구포트폴리오', '황금나비', '올웨더', '정적 60/40', '동일비중']:
        weight_fns[name] = make_static_weights_fn(name)
    for name in ['역변동성', 'GMV', 'MDP', 'ERC', 'GTAA']:
        weight_fns[name] = make_dynamic_weights_fn(name)

    # 국면 → 포트폴리오 매핑
    regime_to_portfolio = {rg: best for rg, (best, _) in best_by_regime.items()}

    # 스위칭 weight function
    switching_log = []

    def get_switching_weights(cur_date, hist):
        regime, _ = classify_regime_at(features_df, cur_date, min_quarters=4)
        if regime is None:
            regime = '골디락스'
        portfolio = regime_to_portfolio.get(regime, '동일비중')
        switching_log.append({'date': cur_date, 'regime': regime, 'portfolio': portfolio})
        fn = weight_fns.get(portfolio)
        if fn is None:
            return None
        return fn(cur_date, hist)

    r_switching = run_portfolio_backtest(
        etf_df, get_switching_weights,
        start_date=start_date, end_date=end_date
    )

    if 'error' not in r_switching:
        print(f"\n스위칭 전략 성과:")
        print(f"  CAGR:    {r_switching['cagr']:.2f}%")
        print(f"  Sharpe:  {r_switching['sharpe']:.3f}")
        print(f"  MDD:     {r_switching['mdd']:.2f}%")
        print(f"  Calmar:  {r_switching['calmar']:.3f}")
        print(f"  Vol:     {r_switching['vol']:.2f}%")
        print(f"  Periods: {r_switching['n_periods']}")

    return r_switching, switching_log


# ─────────── Phase 4: Placebo 검증 ───────────

def run_phase4_placebo(etf_df, features_df, best_by_regime, n_placebo=1000,
                       start_date='2014-06-26', end_date=None):
    """국면 셔플 1000회 — 실제 PiT 국면 vs 무작위 국면"""
    print("\n" + "=" * 60)
    print(f"Phase 4: Placebo Test (국면 셔플 {n_placebo}회)")
    print("=" * 60)

    regime_names = list(best_by_regime.keys())
    regime_to_portfolio = {rg: best for rg, (best, _) in best_by_regime.items()}

    # 각 포트폴리오 weight function
    weight_fns = {}
    weight_fns['KOSPI (벤치마크)'] = lambda d, h: {'069500.KS': 1.0}
    for name in ['영구포트폴리오', '황금나비', '올웨더', '정적 60/40', '동일비중']:
        weight_fns[name] = make_static_weights_fn(name)
    for name in ['역변동성', 'GMV', 'MDP', 'ERC', 'GTAA']:
        weight_fns[name] = make_dynamic_weights_fn(name)

    placebo_sharpes = []
    placebo_mdds = []
    placebo_calmars = []
    placebo_cagrs = []

    rng = np.random.default_rng(42)
    t0 = time.time()

    for i in range(n_placebo):
        def get_shuffled_weights(cur_date, hist, _rng=rng):
            regime = _rng.choice(regime_names)
            portfolio = regime_to_portfolio.get(regime, '동일비중')
            fn = weight_fns.get(portfolio)
            if fn is None:
                return None
            return fn(cur_date, hist)

        r = run_portfolio_backtest(
            etf_df, get_shuffled_weights,
            start_date=start_date, end_date=end_date
        )
        if 'error' not in r:
            placebo_cagrs.append(r['cagr'])
            placebo_sharpes.append(r['sharpe'])
            placebo_mdds.append(r['mdd'])
            placebo_calmars.append(r['calmar'])

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  진행: {i+1}/{n_placebo} ({elapsed:.1f}초)")

    print(f"  완료: {time.time()-t0:.1f}초")

    return dict(
        cagrs=np.array(placebo_cagrs),
        sharpes=np.array(placebo_sharpes),
        mdds=np.array(placebo_mdds),
        calmars=np.array(placebo_calmars),
    )


# ─────────── 전체 파이프라인 ───────────

def run_full_validation(n_placebo=1000):
    print("=" * 60)
    print("KQ Quant Tool — 자산배분 검증 v2")
    print("=" * 60)

    # 1. ETF 데이터
    print("\n[1] ETF 데이터 로드")
    t0 = time.time()
    etf_df = load_etf_data()
    print(f"  완료: {time.time()-t0:.1f}초")
    print(f"  기간: {etf_df.index[0].date()} ~ {etf_df.index[-1].date()}")
    print(f"  ETF: {list(etf_df.columns)}")

    # 2. 매크로 + PiT 국면
    print("\n[2] 매크로 데이터 + PiT 국면")
    import data_loader as dl
    macro_data = dl.load_macro()
    features_df = build_pit_macro_features(macro_data)
    print(f"  매크로 분기: {len(features_df)}개")

    bt_start = max(etf_df.index[0], pd.Timestamp('2014-06-26'))
    bt_end = etf_df.index[-1]

    # Phase 1: 9개 백테스트
    results = run_phase1_all_portfolios(etf_df, start_date=bt_start, end_date=bt_end)

    # Phase 2: 국면별 분석
    by_regime, best_by_regime = run_phase2_regime_analysis(results, features_df)

    # Phase 3: 국면 스위칭
    r_switching, switching_log = run_phase3_regime_switching(
        etf_df, features_df, best_by_regime,
        start_date=bt_start, end_date=bt_end
    )

    # Phase 4: Placebo
    placebo = run_phase4_placebo(
        etf_df, features_df, best_by_regime, n_placebo=n_placebo,
        start_date=bt_start, end_date=bt_end
    )

    # 종합 결과
    print("\n" + "=" * 60)
    print("최종 종합 결과")
    print("=" * 60)

    # 스위칭 vs 단일 포트폴리오
    if 'error' not in r_switching:
        sw_sharpe = r_switching['sharpe']
        sw_mdd = r_switching['mdd']
        sw_calmar = r_switching['calmar']

        print(f"\n[스위칭] CAGR {r_switching['cagr']:.2f}% / "
              f"Sharpe {sw_sharpe:.3f} / MDD {sw_mdd:.2f}% / Calmar {sw_calmar:.3f}")

        # Placebo 비교
        sharpe_pct = float((placebo['sharpes'] <= sw_sharpe).mean() * 100)
        mdd_pct = float((placebo['mdds'] <= sw_mdd).mean() * 100)
        calmar_pct = float((placebo['calmars'] <= sw_calmar).mean() * 100)
        p_sharpe = float((placebo['sharpes'] >= sw_sharpe).mean())
        p_calmar = float((placebo['calmars'] >= sw_calmar).mean())

        print(f"\n셔플 분포 (N={len(placebo['sharpes'])}):")
        print(f"  Sharpe: 실제 {sw_sharpe:.3f} vs 셔플평균 {placebo['sharpes'].mean():.3f} "
              f"(percentile={sharpe_pct:.1f}%, p={p_sharpe:.3f})")
        print(f"  MDD:    실제 {sw_mdd:.2f}% vs 셔플평균 {placebo['mdds'].mean():.2f}% "
              f"(percentile={mdd_pct:.1f}%)")
        print(f"  Calmar: 실제 {sw_calmar:.3f} vs 셔플평균 {placebo['calmars'].mean():.3f} "
              f"(percentile={calmar_pct:.1f}%, p={p_calmar:.3f})")

        print(f"\n해석:")
        if p_sharpe < 0.05:
            print(f"  ✅ PiT 국면 스위칭이 무작위 국면 대비 Sharpe 통계적 유의 우수 (p={p_sharpe:.3f})")
        elif sharpe_pct > 70:
            print(f"  🟡 PiT 국면 스위칭 Sharpe 상위권 (percentile {sharpe_pct:.1f}%)")
        else:
            print(f"  ⚪ PiT 국면 스위칭과 무작위 셔플 차이 미미")

        if p_calmar < 0.05:
            print(f"  ✅ Calmar에서 통계적 유의 우수 (p={p_calmar:.3f})")

    # 저장
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'asset_allocation_v2_results.npz'
    )
    save_kwargs = {}
    for name, r in results.items():
        if 'error' in r:
            continue
        key = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        save_kwargs[f'{key}_cagr'] = r['cagr']
        save_kwargs[f'{key}_sharpe'] = r['sharpe']
        save_kwargs[f'{key}_mdd'] = r['mdd']
        save_kwargs[f'{key}_calmar'] = r['calmar']
    if 'error' not in r_switching:
        save_kwargs['switching_cagr'] = r_switching['cagr']
        save_kwargs['switching_sharpe'] = r_switching['sharpe']
        save_kwargs['switching_mdd'] = r_switching['mdd']
        save_kwargs['switching_calmar'] = r_switching['calmar']
    save_kwargs['placebo_cagrs'] = placebo['cagrs']
    save_kwargs['placebo_sharpes'] = placebo['sharpes']
    save_kwargs['placebo_mdds'] = placebo['mdds']
    save_kwargs['placebo_calmars'] = placebo['calmars']
    np.savez(save_path, **save_kwargs)
    print(f"\n💾 결과 저장: {save_path}")

    return dict(
        portfolios=results,
        by_regime=by_regime,
        best_by_regime=best_by_regime,
        switching=r_switching,
        switching_log=switching_log,
        placebo=placebo,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--refresh', action='store_true', help='ETF 캐시 강제 갱신')
    args = parser.parse_args()

    if args.refresh:
        load_etf_data(force_refresh=True)

    run_full_validation(n_placebo=args.n)

