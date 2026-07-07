"""
validation_core.py — 백테스트 검증용 코어 모듈

데이터 누수 + 통계적 유의성 검증을 위해 백테스트 로직을 재사용 가능한 형태로 추출.
server.py의 _run_strategy_backtest와 동일한 로직이지만 다음을 분리:

  1. _prepare_bt_data()  — 데이터 준비 (1회만)
  2. _run_bt_core()      — 종목 선정 + 백테스트 실행 (1000회 반복 가능)

핵심: 사전계산(precomputed)을 한 번만 하고, 종목 선정 로직만 바꿔가며 검증.
"""

import os, sys, time
os.environ.setdefault('KQ_DISABLE_TABPFN', '1')
import numpy as np
import pandas as pd

# server.py에서 필요한 객체 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_loader as _dl_mod


def _at(df_, date):
    """date 이하의 가장 최근 영업일 row 반환 (pandas 3.0 .asof() 우회)"""
    idx = df_.index[df_.index <= date]
    if len(idx) == 0:
        return pd.Series(np.nan, index=df_.columns)
    return df_.loc[idx[-1]]


def build_mcap_history(universe_dict, ticker_to_code_fn, fin_data,
                       fin_tickers_cache=None):
    """모든 종목의 분기별 시총을 DataFrame으로 구성
    Returns: DataFrame, shape (날짜, 종목), value=시총
    """
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from kq_tool.data.financial_pit import build_mcap_history_pit

    return build_mcap_history_pit(
        universe_dict, ticker_to_code_fn, fin_data, fin_tickers_cache
    )

    mcap_key = '시가총액(티커-상장예정주식수 포함)(백만원)'
    mcap_series_dict = {}

    for yt in universe_dict:
        code = ticker_to_code_fn(yt)
        try:
            if fin_tickers_cache is not None and code not in fin_tickers_cache:
                continue
            sub = fin_data.loc[code]
            if mcap_key not in sub.index:
                continue
            rows = sub.loc[mcap_key]
            if isinstance(rows, pd.Series):
                rows = rows.to_frame().T
            # date, value 추출
            df = rows[['date', 'value']].dropna()
            if df.empty:
                continue
            df = df.set_index(pd.to_datetime(df['date']))['value']
            # value를 숫자로 강제 변환 (문자열/object 섞임 방지)
            # 변환 실패 값(NaN)은 이후 dropna/ffill에서 자연히 제거됨
            df = pd.to_numeric(df, errors='coerce')
            mcap_series_dict[yt] = df
        except Exception:
            continue

    if not mcap_series_dict:
        return pd.DataFrame()

    # DataFrame 결합 — 인덱스 = 날짜, 컬럼 = 종목
    mcap_df = pd.DataFrame(mcap_series_dict)
    mcap_df = mcap_df.sort_index()
    # forward-fill (분기 사이 빈 날짜 채우기)
    mcap_df = mcap_df.ffill()
    return mcap_df


def get_top_mcap_at(mcap_history, date, n=200):
    """date 시점에서 시총 상위 n개 종목 반환 (Point-in-Time)

    Args:
        mcap_history: build_mcap_history() 결과
        date: 기준 시점
        n: 상위 종목 수

    Returns:
        list of tickers (상위 n개)
    """
    if mcap_history.empty:
        return []
    # date 이전의 가장 최근 시총 row
    available = mcap_history.index[mcap_history.index <= date]
    if len(available) == 0:
        return []
    latest_row = mcap_history.loc[available[-1]]
    # NaN 제거 후 상위 n
    valid = latest_row.dropna()
    if len(valid) == 0:
        return []
    # dtype이 object로 섞여 들어온 경우 방어적으로 숫자 변환
    # (원래는 build_mcap_history()에서 이미 숫자로 변환되지만,
    #  혹시 다른 경로로 mcap_history가 만들어질 경우를 대비)
    if valid.dtype == object:
        valid = pd.to_numeric(valid, errors='coerce').dropna()
        if len(valid) == 0:
            return []
    return valid.nlargest(min(n, len(valid))).index.tolist()


def prepare_bt_data(universe_dict, ticker_to_code_fn,
                    start_date=pd.Timestamp('2014-01-01'),
                    end_date=None,
                    use_top_mcap=True,
                    top_n_mcap=200,
                    fin_data=None,
                    fin_tickers_cache=None,
                    point_in_time=False):
    """백테스트 데이터 준비 (한 번만 호출)

    Args:
        use_top_mcap: 시가총액 상위 N개로 제한 (학술적 표준)
        top_n_mcap: 상위 몇 개 (기본 200)
        point_in_time: True면 매월 리밸런싱 시 그 시점 시총으로 동적 선정
                       False면 가장 최근 시총으로 1회 선정 (누수 의심 가능성)

    Returns:
        dict: {price_df, kospi, precomputed_robo, mcap_history, ...}
    """
    # 0) 시총 이력 만들기 (PiT 모드든 정적 모드든 활용)
    mcap_history = None
    if fin_data is not None:
        mcap_history = build_mcap_history(
            universe_dict, ticker_to_code_fn, fin_data, fin_tickers_cache
        )

    # 0-1) 정적 시총 상위 N 필터 (PiT 모드가 아닐 때만)
    target_universe = universe_dict
    if use_top_mcap and not point_in_time and mcap_history is not None and not mcap_history.empty:
        # 최근 시총 기준 상위 N개 (이전 동작과 동일 - 누수 의심)
        latest_row = mcap_history.iloc[-1].dropna()
        if latest_row.dtype == object:
            latest_row = pd.to_numeric(latest_row, errors='coerce').dropna()
        top_tickers = latest_row.nlargest(top_n_mcap).index.tolist()
        target_universe = {t: universe_dict[t] for t in top_tickers if t in universe_dict}
        print(f"  시총 상위 {len(target_universe)}개로 제한 (정적, 최근 기준)")
    elif use_top_mcap and point_in_time:
        # PiT 모드: 전체 유니버스 사용, 매월 동적 필터
        print(f"  Point-in-Time 모드: 매월 시총 상위 {top_n_mcap}개 동적 선정")
    elif not use_top_mcap:
        print(f"  전체 유니버스 사용 ({len(target_universe)}개)")

    # 1) 수정주가 패널 로딩 (검증 관문 통과본) — 구 parquet 캐시 대체
    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from kq_tool.data.app_data import load_close_panel

    _panel = load_close_panel()
    close_groups = {code: _panel[code].dropna() for code in _panel.columns}

    if not close_groups:
        raise RuntimeError('종가 데이터 로드 실패')

    # 2) UNIVERSE 종목 시리즈
    px = {}
    for t in target_universe:
        code = ticker_to_code_fn(t)
        s = close_groups.get(code)
        if s is not None and len(s) >= 20:
            px[t] = s

    # 3) DataFrame 구성 + 기간 필터
    price_df = pd.DataFrame(px)
    price_df = price_df[price_df.index >= start_date]
    if end_date is not None:
        price_df = price_df[price_df.index <= end_date]
    valid_cols = [c for c in price_df.columns if price_df[c].notna().sum() >= 60]
    price_df = price_df[valid_cols]
    price_df = price_df.ffill().dropna(how='all')

    # 4) KOSPI 벤치마크 (단일 게이트웨이, ECOS 수집본)
    from kq_tool.data.gateway import get_kospi_benchmark
    kospi = get_kospi_benchmark()
    kospi = kospi[(kospi.index >= start_date) &
                   (kospi.index <= (end_date or kospi.index[-1]))]

    # 5) 로보 사전계산 (RSI, MACD, MA)
    rsi_dict, macd_bull_dict, ma20_dict, ma60_dict = {}, {}, {}, {}
    for col in price_df.columns:
        s = price_df[col].dropna()
        if len(s) < 60:
            continue
        # RSI (14일)
        delta = s.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        ag = gain.rolling(14, min_periods=14).mean()
        al = loss.rolling(14, min_periods=14).mean()
        rs = ag / al.replace(0, np.nan)
        rsi_dict[col] = 100 - (100 / (1 + rs))
        # MACD (12, 26, 9)
        ema12 = s.ewm(span=12, adjust=False).mean()
        ema26 = s.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bull_dict[col] = (macd_line > signal_line).astype(int)
        # MA20, MA60
        ma20_dict[col] = s.rolling(20).mean()
        ma60_dict[col] = s.rolling(60).mean()

    precomputed = dict(
        rsi=pd.DataFrame(rsi_dict),
        macd_bull=pd.DataFrame(macd_bull_dict),
        ma20=pd.DataFrame(ma20_dict),
        ma60=pd.DataFrame(ma60_dict),
    )

    return dict(
        price_df=price_df,
        kospi=kospi,
        precomputed=precomputed,
        mcap_history=mcap_history,
        point_in_time=point_in_time,
        top_n_mcap=top_n_mcap,
        use_top_mcap=use_top_mcap,
    )


def select_quant(hist, top_n):
    """모멘텀 종목 선정 (12-1개월)"""
    if len(hist) >= 252:
        mom = (hist.iloc[-21] / hist.iloc[-252] - 1)
    elif len(hist) >= 60:
        mom = (hist.iloc[-1] / hist.iloc[-60] - 1)
    else:
        return []
    return mom.dropna().nlargest(top_n).index.tolist()


def select_robo(hist, top_n, precomputed, cur_date):
    """로보 신호 종목 선정 (RSI+MACD+MA)"""
    if len(hist) < 60:
        return []

    rsi_last = _at(precomputed['rsi'], cur_date)
    macd_bull = _at(precomputed['macd_bull'], cur_date) == 1
    ma20 = _at(precomputed['ma20'], cur_date)
    ma60 = _at(precomputed['ma60'], cur_date)
    cur = hist.iloc[-1]

    rsi_score = np.where(rsi_last < 30, 20,
                np.where(rsi_last > 70, -20, 0))
    macd_score = np.where(macd_bull, 25, -25)
    ma20_score = np.where(cur > ma20, 15, -15)
    ma60_score = np.where(cur > ma60, 20, -20)
    total_score = pd.Series(
        rsi_score + macd_score + ma20_score + ma60_score,
        index=hist.columns
    )
    valid_mask = rsi_last.notna() & ma20.notna() & ma60.notna() & cur.notna()
    total_score = total_score[valid_mask]

    positive = total_score[total_score > 0]
    if len(positive) >= top_n:
        return positive.nlargest(top_n).index.tolist()
    else:
        return total_score.nlargest(top_n).index.tolist()


def select_random(hist, top_n, rng):
    """무작위 종목 선정 (Placebo Test용) — 벡터화"""
    if len(hist) < 60:
        return []
    # NaN 아닌 컬럼만 (벡터 연산)
    last_row = hist.iloc[-1]
    valid_cols = last_row.dropna().index.tolist()
    if len(valid_cols) == 0:
        return []
    n = min(top_n, len(valid_cols))
    return rng.choice(valid_cols, size=n, replace=False).tolist()


def run_bt_core(data, strategy='quant', top_n=5, rebalance='M',
                custom_selector=None, rng=None):
    """백테스트 코어 실행 (재사용 가능)

    Args:
        data: prepare_bt_data() 결과
        strategy: 'quant' | 'robo' | 'random'
        custom_selector: 사용자 정의 종목 선정 함수 (override)
        rng: 무작위 선정용 (Placebo Test)

    Returns:
        dict: {equity, dates, metrics, kospi_equity, ...}
    """
    price_df = data['price_df']
    kospi = data['kospi']
    precomputed = data['precomputed']

    if price_df.empty:
        return {'error': '데이터 없음'}

    # 리밸런싱 날짜
    rule = {'M':'ME','Q':'QE','W':'W'}.get(rebalance, 'ME')
    months = price_df.resample(rule).last()

    equity = [100.0]
    eq_dates = [price_df.index[0]]
    holdings_log = []

    for i in range(len(months) - 1):
        cur_date = months.index[i]
        next_date = months.index[i+1]
        hist = price_df.loc[:cur_date]
        if len(hist) < 60:
            continue

        # Point-in-Time 시총 필터 (PiT 모드)
        # 매월 그 시점 시총 상위 N개로 종목 풀 동적 제한 → 누수 방지
        if data.get('point_in_time') and data.get('mcap_history') is not None:
            mcap_hist = data['mcap_history']
            top_tickers_pit = get_top_mcap_at(mcap_hist, cur_date, data.get('top_n_mcap', 200))
            # hist에서 그 종목들만 사용
            available = [t for t in top_tickers_pit if t in hist.columns]
            if len(available) < top_n:
                continue
            hist = hist[available]

        # 종목 선정
        if custom_selector is not None:
            selected = custom_selector(hist, top_n, cur_date)
        elif strategy == 'quant':
            selected = select_quant(hist, top_n)
        elif strategy == 'robo':
            # PiT 모드면 precomputed 컬럼이 더 많을 수 있음 — 매칭만 사용
            if data.get('point_in_time') and precomputed is not None:
                # precomputed의 컬럼은 전체 종목, 우리는 hist 컬럼만 사용
                pre_filtered = {
                    k: v[hist.columns] if isinstance(v, pd.DataFrame) and
                       set(hist.columns).issubset(v.columns) else v
                    for k, v in precomputed.items()
                }
                selected = select_robo(hist, top_n, pre_filtered, cur_date)
            else:
                selected = select_robo(hist, top_n, precomputed, cur_date)
        elif strategy == 'random':
            if rng is None:
                rng = np.random.default_rng()
            selected = select_random(hist, top_n, rng)
        else:
            selected = []

        if not selected:
            continue

        w = 1.0 / len(selected)
        period_px = price_df.loc[cur_date:next_date]
        if len(period_px) < 2:
            continue
        period_ret = 0.0
        valid_count = 0
        for t in selected:
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
            continue
        if valid_count < len(selected):
            period_ret = period_ret * len(selected) / valid_count

        equity.append(equity[-1] * (1 + period_ret))
        eq_dates.append(next_date)
        holdings_log.append(dict(date=cur_date, tickers=selected))

    eq_series = pd.Series(equity, index=pd.to_datetime(eq_dates))

    # 성과지표
    freq = 12 if rebalance == 'M' else 4 if rebalance == 'Q' else 52
    n = len(eq_series) - 1
    if n < 3:
        return {'error': '데이터 부족', 'equity': equity, 'n_rebalance': len(holdings_log)}

    years = n / freq
    total = float(eq_series.iloc[-1] / eq_series.iloc[0])
    cagr = total**(1/years) - 1 if years > 0 else 0
    rets = eq_series.pct_change().dropna()
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = (cagr - 0.03) / vol if vol > 0 else 0
    peak = eq_series.cummax()
    mdd = float(((eq_series - peak) / peak).min())

    # KOSPI 벤치마크
    bench_cagr = None
    bench_total = None
    if kospi is not None and len(kospi) > 30:
        kb = kospi.reindex(eq_series.index, method='ffill').dropna()
        if len(kb) > 1:
            kb = kb / kb.iloc[0] * 100
            bench_total = float(kb.iloc[-1] / kb.iloc[0])
            bench_cagr = bench_total**(1/years) - 1 if years > 0 else 0

    return dict(
        equity=eq_series,
        cagr=round(cagr * 100, 2),
        total_return=round((total - 1) * 100, 2),
        vol=round(vol * 100, 2),
        sharpe=round(sharpe, 3),
        mdd=round(mdd * 100, 2),
        bench_cagr=round(bench_cagr * 100, 2) if bench_cagr is not None else None,
        bench_total=round((bench_total - 1) * 100, 2) if bench_total is not None else None,
        alpha=round((cagr - bench_cagr) * 100, 2) if bench_cagr is not None else None,
        n_rebalance=len(holdings_log),
    )


if __name__ == '__main__':
    # 단위 테스트
    print("=" * 60)
    print("validation_core.py 단위 테스트")
    print("=" * 60)

    # server.py에서 UNIVERSE 가져오기
    import server
    universe = server.UNIVERSE
    ticker_to_code = server._ticker_to_code

    print(f"\n[1] 데이터 준비 (시총 상위 200)")
    t0 = time.time()
    data = prepare_bt_data(
        universe, ticker_to_code,
        use_top_mcap=True, top_n_mcap=200, point_in_time=True,
        fin_data=server.EXCEL_FIN,
        fin_tickers_cache=server._dl_mod._FIN_TICKERS_CACHE,
    )
    print(f"  소요: {time.time()-t0:.1f}초")
    print(f"  price_df: {data['price_df'].shape}")
    print(f"  종목 수: {len(data['price_df'].columns)}")

    print(f"\n[2] 퀀트 백테스트")
    t0 = time.time()
    r1 = run_bt_core(data, strategy='quant', top_n=5)
    print(f"  소요: {time.time()-t0:.1f}초")
    print(f"  CAGR: {r1.get('cagr')}%, Sharpe: {r1.get('sharpe')}, MDD: {r1.get('mdd')}%")
    print(f"  KOSPI CAGR: {r1.get('bench_cagr')}%, 알파: {r1.get('alpha')}%p")

    print(f"\n[3] 로보 백테스트")
    t0 = time.time()
    r2 = run_bt_core(data, strategy='robo', top_n=5)
    print(f"  소요: {time.time()-t0:.1f}초")
    print(f"  CAGR: {r2.get('cagr')}%, Sharpe: {r2.get('sharpe')}, MDD: {r2.get('mdd')}%")
    print(f"  알파: {r2.get('alpha')}%p")

    print(f"\n[4] 무작위 백테스트 5회")
    rng = np.random.default_rng(42)
    cagrs = []
    t0 = time.time()
    for i in range(5):
        r = run_bt_core(data, strategy='random', top_n=5, rng=rng)
        cagrs.append(r.get('cagr'))
        print(f"  시도 {i+1}: CAGR {r.get('cagr')}%")
    print(f"  5회 평균: {np.mean(cagrs):.2f}%, 소요: {time.time()-t0:.1f}초")

    print(f"\n✅ Task 1 완료")



