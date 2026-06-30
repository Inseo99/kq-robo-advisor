"""
validation_alpha_decay_is_oos.py — Alpha Decay IS/OOS 검증

목적:
  Alpha Decay를 "수익률 예측 엔진"이 아니라
  "신호 유효기간/재점검 기간"으로 쓸 수 있는지 검증한다.

검증 방식:
  1. 2014~2020(IS): 신호별 Alpha Decay 유효기간 추정
  2. 2021~2026(OOS): IS에서 정한 유효기간을 그대로 적용
  3. Placebo: 같은 종목/같은 방향/같은 보유기간의 무작위 날짜와 비교

한국시장 보정:
  - KOSPI/KOSDAQ 시총 상위 유니버스 사용
  - 상/하한가급 움직임(기본 ±29.5%) 발생일 주변 신호 제외
  - 매도성 신호는 하락을 양의 edge return으로 변환해 측정
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

import argparse
import time
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_loader as dl
import server


IS_START = pd.Timestamp('2014-01-01')
IS_END = pd.Timestamp('2020-12-31')
OOS_START = pd.Timestamp('2021-01-01')
HORIZONS = [1, 3, 5, 7, 10, 15, 20]
SIGNALS = {
    'RSI 과매도': (+1, 's_rsi'),
    'RSI 과매수': (-1, 's_rsi'),
    'MACD 골든크로스': (+1, 's_macd'),
    'MACD 데드크로스': (-1, 's_macd'),
    'BB 하단터치': (+1, 's_bb'),
    'BB 상단터치': (-1, 's_bb'),
}


def _exp(t, a, lam, c):
    return a * np.exp(-lam * t) + c


def signal_masks(close):
    rsi = server._rsi(close)
    macd, sig = server._macd(close)
    bb_up, _, bb_low = server._bb(close)
    return {
        'RSI 과매도': rsi < 30,
        'RSI 과매수': rsi > 70,
        'MACD 골든크로스': (macd > sig) & (macd.shift(1) <= sig.shift(1)),
        'MACD 데드크로스': (macd < sig) & (macd.shift(1) >= sig.shift(1)),
        'BB 하단터치': close <= bb_low,
        'BB 상단터치': close >= bb_up,
    }


def get_universe(top_n):
    try:
        hist = server.get_mcap_history()
        tickers = server.get_top_mcap_at(hist, IS_END, n=top_n)
        if tickers:
            return tickers[:top_n], 'IS_END 시점 시총 상위'
    except Exception:
        pass
    return server.get_top_marketcap_tickers(limit=top_n), '최근 시총 상위 fallback'


def load_price_map(tickers):
    out = {}
    for i, ticker in enumerate(tickers, 1):
        code = server._ticker_to_code(ticker)
        df = dl.get_ohlcv_df(code, adjusted=False)
        if df is None or 'Close' not in df or df['Close'].dropna().shape[0] < 260:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        out[ticker] = df
        if i % 50 == 0:
            print(f"  가격 로드: {i}/{len(tickers)}")
    return out


def valid_signal_dates(close, mask, start, end, limit_threshold):
    limit_hit = close.pct_change().abs() >= limit_threshold
    dates = []
    raw_dates = mask[mask].index
    excluded = 0
    for dt in raw_dates:
        if dt < start or dt > end:
            continue
        idx = close.index.get_loc(dt)
        if isinstance(idx, slice):
            idx = idx.stop - 1
        bad_limit = bool(limit_hit.iloc[idx])
        if idx + 1 < len(close):
            bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
        if bad_limit:
            excluded += 1
            continue
        dates.append(dt)
    return dates, excluded


def edge_return(close, dt, horizon, direction):
    idx = close.index.get_loc(dt)
    if isinstance(idx, slice):
        idx = idx.stop - 1
    if idx + horizon >= len(close):
        return None
    p0 = float(close.iloc[idx])
    p1 = float(close.iloc[idx + horizon])
    if p0 <= 0 or not np.isfinite(p0) or not np.isfinite(p1):
        return None
    return direction * ((p1 - p0) / p0)


def estimate_is_decay(price_map, limit_threshold):
    stats = {}
    for sname, (direction, _) in SIGNALS.items():
        hvals = {h: [] for h in HORIZONS}
        count = 0
        excluded_total = 0
        for ticker, df in price_map.items():
            close = df['Close'].dropna()
            mask = signal_masks(close)[sname]
            dates, excluded = valid_signal_dates(close, mask, IS_START, IS_END, limit_threshold)
            excluded_total += excluded
            count += len(dates)
            for dt in dates:
                for h in HORIZONS:
                    ret = edge_return(close, dt, h, direction)
                    if ret is not None:
                        hvals[h].append(ret)

        avg = {h: (float(np.mean(v)) if v else None) for h, v in hvals.items()}
        valid = [(h, r) for h, r in avg.items() if r is not None]
        half_life = None
        basis = 'none'
        selected_day = None
        if len(valid) >= 4:
            try:
                x = np.array([h for h, _ in valid], dtype=float)
                y = np.array([r for _, r in valid], dtype=float)
                popt, _ = curve_fit(_exp, x, y, p0=[y[0], 0.1, 0.0], maxfev=3000)
                if popt[1] > 0:
                    hl = float(np.log(2) / popt[1])
                    if 0 < hl <= max(HORIZONS) * 1.5:
                        half_life = round(hl, 1)
                        selected_day = int(max(1, min(max(HORIZONS), round(half_life))))
                        basis = 'half_life'
            except Exception:
                pass

        if selected_day is None:
            positives = [(h, r) for h, r in valid if r is not None and r > 0]
            if positives:
                selected_day = int(max(positives, key=lambda x: x[1])[0])
                basis = 'peak'

        stats[sname] = dict(
            count=count,
            excluded_limit=excluded_total,
            half_life=half_life,
            selected_day=selected_day,
            basis=basis,
            avg={h: (round(avg[h] * 100, 3) if avg[h] is not None else None) for h in HORIZONS},
        )
    return stats


def collect_oos_events(price_map, is_stats, limit_threshold):
    events_by_signal = {s: [] for s in SIGNALS}
    for sname, (direction, _) in SIGNALS.items():
        h = is_stats[sname].get('selected_day')
        if not h:
            continue
        for ticker, df in price_map.items():
            close = df['Close'].dropna()
            mask = signal_masks(close)[sname]
            dates, _ = valid_signal_dates(close, mask, OOS_START, close.index[-1], limit_threshold)
            for dt in dates:
                ret = edge_return(close, dt, h, direction)
                if ret is not None:
                    events_by_signal[sname].append((ticker, dt, h, direction, ret))
    return events_by_signal


def random_return_pool(price_map, horizon, direction, limit_threshold):
    vals = []
    for _, df in price_map.items():
        close = df['Close'].dropna()
        limit_hit = close.pct_change().abs() >= limit_threshold
        idxs = np.flatnonzero(np.asarray(close.index >= OOS_START))
        idxs = idxs[idxs + horizon < len(close)]
        for idx in idxs:
            bad_limit = bool(limit_hit.iloc[idx])
            if idx + 1 < len(close):
                bad_limit = bad_limit or bool(limit_hit.iloc[idx + 1])
            if bad_limit:
                continue
            p0 = float(close.iloc[idx])
            p1 = float(close.iloc[idx + horizon])
            if p0 > 0 and np.isfinite(p0) and np.isfinite(p1):
                vals.append(direction * ((p1 - p0) / p0))
    return np.array(vals, dtype=float)


def evaluate_oos(price_map, is_stats, n_placebo, limit_threshold):
    events_by_signal = collect_oos_events(price_map, is_stats, limit_threshold)
    rng = np.random.default_rng(42)
    results = {}
    pool_cache = {}

    all_events = []
    for sname, events in events_by_signal.items():
        all_events.extend(events)
        actual_vals = [x[-1] for x in events]
        if not actual_vals:
            results[sname] = dict(count=0, actual=None, placebo_mean=None, percentile=None, p=None)
            continue
        h = is_stats[sname].get('selected_day')
        direction = SIGNALS[sname][0]
        pool_key = (h, direction)
        if pool_key not in pool_cache:
            pool_cache[pool_key] = random_return_pool(price_map, h, direction, limit_threshold)
        pool = pool_cache[pool_key]
        placebo_means = []
        if len(pool):
            for _ in range(n_placebo):
                vals = rng.choice(pool, size=len(actual_vals), replace=True)
                placebo_means.append(float(np.mean(vals)))
        actual = float(np.mean(actual_vals))
        placebo = np.array(placebo_means)
        p = float((placebo >= actual).mean()) if len(placebo) else None
        pct = float((placebo <= actual).mean() * 100) if len(placebo) else None
        results[sname] = dict(
            count=len(actual_vals),
            actual=actual,
            placebo_mean=float(placebo.mean()) if len(placebo) else None,
            percentile=pct,
            p=p,
        )

    actual_vals = [x[-1] for x in all_events]
    placebo_means = []
    if actual_vals:
        for _ in range(n_placebo):
            vals = []
            for sname, events in events_by_signal.items():
                if not events:
                    continue
                h = is_stats[sname].get('selected_day')
                if not h:
                    continue
                direction = SIGNALS[sname][0]
                pool_key = (h, direction)
                if pool_key not in pool_cache:
                    pool_cache[pool_key] = random_return_pool(price_map, h, direction, limit_threshold)
                pool = pool_cache[pool_key]
                if len(pool):
                    vals.extend(rng.choice(pool, size=len(events), replace=True).tolist())
            if vals:
                placebo_means.append(float(np.mean(vals)))
    placebo = np.array(placebo_means)
    actual = float(np.mean(actual_vals)) if actual_vals else None
    results['전체'] = dict(
        count=len(actual_vals),
        actual=actual,
        placebo_mean=float(placebo.mean()) if len(placebo) else None,
        percentile=float((placebo <= actual).mean() * 100) if len(placebo) else None,
        p=float((placebo >= actual).mean()) if len(placebo) else None,
    )
    return results


def run(top=100, n_placebo=500, limit_threshold=0.295):
    print("=" * 72)
    print("Alpha Decay IS/OOS 검증")
    print("=" * 72)
    print(f"IS:  {IS_START.date()} ~ {IS_END.date()}")
    print(f"OOS: {OOS_START.date()} ~ 데이터 끝")
    print(f"가격제한폭 필터: 일간 ±{limit_threshold*100:.1f}% 근처 제외")

    tickers, universe_note = get_universe(top)
    print(f"\n[1] 유니버스: {universe_note} {len(tickers)}개")
    price_map = load_price_map(tickers)
    print(f"  유효 가격 데이터: {len(price_map)}개")

    print("\n[2] IS Alpha Decay 추정")
    is_stats = estimate_is_decay(price_map, limit_threshold)
    print(f"{'신호':<18} {'IS표본':>7} {'제외':>6} {'선택일':>7} {'근거':>10} {'IS edge':>10}")
    for sname, st in is_stats.items():
        day = st['selected_day'] if st['selected_day'] is not None else '-'
        edge = st['avg'].get(day) if isinstance(day, int) else None
        edge_txt = f"{edge:+.3f}%" if edge is not None else '-'
        print(f"{sname:<18} {st['count']:>7} {st['excluded_limit']:>6} {str(day):>7} {st['basis']:>10} {edge_txt:>10}")

    print(f"\n[3] OOS Placebo 검증 ({n_placebo}회)")
    t0 = time.time()
    oos = evaluate_oos(price_map, is_stats, n_placebo, limit_threshold)
    print(f"  완료: {time.time()-t0:.1f}초")

    print("\nOOS 결과: edge return은 신호 방향 수익률입니다.")
    print(f"{'신호':<18} {'OOS표본':>7} {'실제':>10} {'무작위평균':>12} {'percentile':>11} {'p':>8}")
    for sname in list(SIGNALS.keys()) + ['전체']:
        r = oos[sname]
        actual = '-' if r['actual'] is None else f"{r['actual']*100:+.3f}%"
        pm = '-' if r['placebo_mean'] is None else f"{r['placebo_mean']*100:+.3f}%"
        pct = '-' if r['percentile'] is None else f"{r['percentile']:.1f}%"
        p = '-' if r['p'] is None else f"{r['p']:.3f}"
        print(f"{sname:<18} {r['count']:>7} {actual:>10} {pm:>12} {pct:>11} {p:>8}")

    agg = oos['전체']
    print("\n해석:")
    if agg['p'] is not None and agg['p'] < 0.05 and agg['actual'] is not None and agg['actual'] > 0:
        print("  ✅ Alpha Decay 유효기간이 OOS에서 무작위 날짜보다 유의하게 좋습니다.")
        print("  → 로보신호의 재점검 기간/유효기간 레이어로 쓸 근거가 있습니다.")
    elif agg['actual'] is not None and agg['actual'] > 0:
        print("  🟡 OOS edge는 양수지만 통계적으로 강하진 않습니다.")
        print("  → 고객 화면에서는 '유효 추정/재점검' 표현을 유지하는 것이 안전합니다.")
    else:
        print("  ⚪ OOS에서 유의한 edge가 확인되지 않았습니다.")
        print("  → Alpha Decay는 매매 엔진이 아니라 보수적 재점검 타이머로만 사용하세요.")

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alpha_decay_is_oos_results.npz')
    np.savez(
        save_path,
        signals=np.array(list(SIGNALS.keys()), dtype=object),
        selected_days=np.array([is_stats[s].get('selected_day') for s in SIGNALS], dtype=object),
        bases=np.array([is_stats[s].get('basis') for s in SIGNALS], dtype=object),
        oos_actual=np.array([oos[s].get('actual') for s in SIGNALS] + [oos['전체'].get('actual')], dtype=object),
        oos_p=np.array([oos[s].get('p') for s in SIGNALS] + [oos['전체'].get('p')], dtype=object),
    )
    print(f"\n결과 저장: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--top', type=int, default=100, help='IS_END 시총 상위 N개')
    parser.add_argument('--n', type=int, default=500, help='Placebo 반복 횟수')
    parser.add_argument('--limit', type=float, default=0.295, help='가격제한폭 근처 제외 기준')
    args = parser.parse_args()
    run(top=args.top, n_placebo=args.n, limit_threshold=args.limit)
