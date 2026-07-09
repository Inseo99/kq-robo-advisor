"""src/kq_tool/api/reco_track.py — 추천 규칙 모의 트랙 v3
월별 국면(PiT)→ERC v1 비중 + 일별 가격 → 일별 NAV · 종목별 기여도 · 세후

교수 피드백 대응: 1-a(리밸런싱 구성→일별 평가금액 역산), 1-c(종목별 수익 기여도).
원리: 월초 비중으로 보유 수량 고정 → 월중 일별 평가 → 월말 새 비중으로 재구성.
정합 계약: 월간 수익률 == 일별 NAV 월말/월초 − 1 (동일 가격 패널에서 산출되므로 정확 일치).
"""
from __future__ import annotations

import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_WEIGHTS_CSV = os.path.join(_ROOT, "data", "analysis_outputs", "regime_erc_v1_weights.csv")
_LABELS_CSV = os.path.join(_ROOT, "data", "regime", "labels.csv")

ASSET_TICKERS = {
    "stocks": ["069500", "229200"],
    "bonds": ["148070", "114260"],
    "gold": ["132030"],
    "cash": ["153130"],
}
TICKER_NAMES = {
    "069500": "KODEX 200", "229200": "KODEX 코스닥150",
    "148070": "KOSEF 국고채10년", "114260": "KODEX 국고채3년",
    "132030": "KODEX 골드선물(H)", "153130": "KODEX 단기채권",
}
TAXABLE_ASSETS = {"bonds", "gold", "cash"}
TAX_RATE = 0.154

_CAVEATS = ("추천 규칙 소급 적용 모의 성과(실제 추천 기록 아님) · "
            "ERC v1 가중치는 전 기간 산출로 완전한 OOS 아님 · 국면 라벨은 PiT(vintage) · "
            "월초 완전 리밸런싱(군내 동일가중) 가정, 밴드·부분이동·비용 미반영")
_TAX_CAVEATS = ("세후: 2026년 일반계좌 개인 기준 근사 — 채권·금·현금성 ETF 매매차익에 "
                "배당소득세 15.4%(지방세 포함), 국내주식형 비과세, 손실 상계 없음(세목 특성), "
                "완전 리밸런싱 가정상 매월 실현 과세(밴드 운용 대비 보수적) · "
                "과표기준가·금융소득종합과세·ISA/연금계좌 미반영 · 일별 곡선은 세전 기준 · "
                "장중 고저 밴드는 종목 고저의 가중합 = 포락선(한계)이며 실제 포트 장중 고저가 아님 — "
                "종목별 고저 발생 시각이 달라 실제 경로는 밴드 안쪽에 있음")

_DATE_CANDS = ("date", "month", "일자", "날짜", "기준월", "asof")
_LABEL_CANDS = ("regime", "label", "국면", "regime_label", "국면라벨")


def _load_labels() -> pd.Series:
    df = pd.read_csv(_LABELS_CSV)
    dcol = next((c for c in df.columns if c.strip().lower() in _DATE_CANDS), df.columns[0])
    lcol = next((c for c in df.columns if c.strip().lower() in _LABEL_CANDS), None)
    if lcol is None:
        lcol = next(c for c in df.columns if c != dcol and df[c].dtype == object)
    s = df.set_index(pd.to_datetime(df[dcol]))[lcol].astype(str)
    return s.resample("ME").last().dropna()


def _daily_price_panel() -> pd.DataFrame:
    """전 종목 일별 가격 패널 (공통 거래일 교집합)."""
    from kq_tool.data.app_data import get_price_series
    cols = {}
    for tickers in ASSET_TICKERS.values():
        for t in tickers:
            px = get_price_series(t, "max")
            px.index = pd.to_datetime(px.index)
            cols[t] = px.astype(float)
    return pd.DataFrame(cols).dropna()


_OHLC_CSV = os.path.join(_ROOT, "data", "prices", "etf_ohlc.csv")


def _ohlc_panels() -> dict | None:
    """실제 시/고/저/종 패널 (scripts/fetch_etf_ohlc.py 산출물 전용).
    파일이 없거나 종목이 빠지면 None → 장중 분해 자동 비활성.
    합성 OHLC는 사용하지 않는다 — 실제 데이터가 없으면 표시하지 않는 것이 원칙."""
    if not os.path.exists(_OHLC_CSV):
        return None
    try:
        df = pd.read_csv(_OHLC_CSV, parse_dates=["date"])
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        need = {t for ts in ASSET_TICKERS.values() for t in ts}
        if not need.issubset(set(df["ticker"].unique())):
            return None
        panels = {}
        for key in ("open", "high", "low", "close"):
            p = df.pivot(index="date", columns="ticker", values=key)[sorted(need)]
            panels[key] = p.astype(float)
        common = panels["close"].dropna().index
        return {k: v.loc[common].dropna() for k, v in panels.items()}
    except Exception:
        return None


def _ticker_weights(class_w: pd.Series) -> dict:
    """자산군 비중 → 종목 비중 (군내 동일가중)."""
    out = {}
    for asset, tickers in ASSET_TICKERS.items():
        for t in tickers:
            out[t] = float(class_w.get(asset, 0.0)) / len(tickers)
    return out


def build_reco_track(months: int = 36) -> dict:
    try:
        weights = pd.read_csv(_WEIGHTS_CSV, index_col=0)
        labels = _load_labels()
        panel = _daily_price_panel()
    except FileNotFoundError as e:
        return {"error": f"입력 파일 없음: {e.filename}"}
    except Exception as e:
        return {"error": f"모의 트랙 계산 실패: {e}"}

    month_ends = panel.resample("ME").last().index
    month_ends = [m for m in month_ends if m <= panel.index[-1]]  # 부분월 제외
    month_ends = month_ends[-(months + 1):]
    if len(month_ends) < 2:
        return {"error": "가격 데이터 기간 부족"}

    rows, daily_points = [], []
    cum = cum_at = 1.0
    for i in range(1, len(month_ends)):
        m0, m1 = month_ends[i - 1], month_ends[i]
        past = labels[labels.index < m1]         # PiT: 해당 월 시작 이전 라벨
        if past.empty:
            continue
        regime = past.iloc[-1]
        if regime not in weights.index:
            continue
        tw = _ticker_weights(weights.loc[regime])

        window = panel[(panel.index > m0) & (panel.index <= m1)]
        base = panel[panel.index <= m0].iloc[-1]  # 월초(전월말) 가격 = 수량 고정 기준
        if window.empty:
            continue

        # 종목별 기여 (수량 고정: w_i × (P_end/P_start − 1))
        t_contrib = {t: round(tw[t] * (float(window[t].iloc[-1]) / float(base[t]) - 1.0), 6)
                     for t in tw}
        c_contrib = {a: round(sum(t_contrib[t] for t in ASSET_TICKERS[a]), 6)
                     for a in ASSET_TICKERS}
        port = sum(t_contrib.values())
        tax = round(sum(TAX_RATE * max(0.0, c_contrib[a]) for a in TAXABLE_ASSETS), 6)

        # 일별 NAV: 월초 수량 고정 평가 → 전월 누적계수에 연쇄 (정합 계약 성립)
        rel = (window / base).mul(pd.Series(tw)).sum(axis=1)   # 월초=1 기준 경로
        month_pts = [{"date": dt.strftime("%Y-%m-%d"), "nav": round(cum * float(v), 6)}
                     for dt, v in rel.items()]
        daily_points.extend(month_pts)

        cum *= (1.0 + port)
        cum_at *= (1.0 + port - tax)
        rows.append(dict(
            month=m1.strftime("%Y-%m"), regime=regime,
            port_return=round(port, 6), port_return_at=round(port - tax, 6),
            tax_paid=tax,
            cum_return=round(cum - 1.0, 6), cum_return_at=round(cum_at - 1.0, 6),
            weights={a: round(float(weights.loc[regime].get(a, 0.0)), 4) for a in ASSET_TICKERS},
            contributions=c_contrib,
            ticker_contributions=t_contrib,
        ))

    if not rows:
        return {"error": "라벨·가격 겹치는 구간 없음 — labels.csv 기간 확인"}

    intraday = _intraday_decomposition(weights, labels, month_ends, daily_points)

    contrib_sum = {a: round(sum(r["contributions"][a] for r in rows), 6) for a in ASSET_TICKERS}
    ticker_sum = {t: round(sum(r["ticker_contributions"][t] for r in rows), 6)
                  for a in ASSET_TICKERS for t in ASSET_TICKERS[a]}
    return dict(
        rows=rows, n_months=len(rows),
        cum_return=rows[-1]["cum_return"], cum_return_at=rows[-1]["cum_return_at"],
        total_tax=round(sum(r["tax_paid"] for r in rows), 6),
        contrib_totals=contrib_sum, ticker_totals=ticker_sum,
        daily=daily_points,
        assets=list(ASSET_TICKERS), asset_tickers=ASSET_TICKERS, ticker_names=TICKER_NAMES,
        intraday=intraday,
        caveats=_CAVEATS, tax_caveats=_TAX_CAVEATS, readonly=True,
    )


def _intraday_decomposition(weights, labels, month_ends, daily_points) -> dict | None:
    """오버나이트/장중 분해 + 고저 포락선. 항등식: (1+on)(1+id) = 일수익률(프레임 종가 기준)."""
    p = _ohlc_panels()
    if p is None:
        return None
    O, H, L, C = p["open"], p["high"], p["low"], p["close"]
    nav_by_date = {pt["date"]: pt["nav"] for pt in daily_points}
    on_cum = id_cum = 1.0
    navF = peak = 1.0
    mdd_close = mdd_bound = 0.0
    bands = []
    for i in range(1, len(month_ends)):
        m0, m1 = month_ends[i - 1], month_ends[i]
        past = labels[labels.index < m1]
        if past.empty or past.iloc[-1] not in weights.index:
            continue
        tw = pd.Series(_ticker_weights(weights.loc[past.iloc[-1]]))
        idx = C.index[(C.index > m0) & (C.index <= m1)]
        base = C[C.index <= m0]
        if idx.empty or base.empty:
            continue
        q = tw / base.iloc[-1]                    # 월초 종가 기준 고정 수량
        prev_val = float((q * base.iloc[-1]).sum())
        for dt in idx:
            v_open = float((q * O.loc[dt]).sum()); v_close = float((q * C.loc[dt]).sum())
            v_low = float((q * L.loc[dt]).sum());  v_high = float((q * H.loc[dt]).sum())
            on = v_open / prev_val - 1.0
            iday = v_close / v_open - 1.0          # 항등식 성립: (1+on)(1+id)=v_close/prev_val
            on_cum *= (1.0 + on); id_cum *= (1.0 + iday)
            lo_b = v_low / prev_val - 1.0; hi_b = v_high / prev_val - 1.0
            nav_prev = navF
            navF *= v_close / prev_val
            peak = max(peak, nav_prev * (1.0 + hi_b))
            mdd_bound = min(mdd_bound, nav_prev * (1.0 + lo_b) / peak - 1.0)
            mdd_close = min(mdd_close, navF / max(peak, navF) - 1.0)
            ds = dt.strftime("%Y-%m-%d")
            if ds in nav_by_date:                  # 차트 밴드: 종가 NAV 스케일에 비율 적용
                base_nav = nav_by_date[ds] / (v_close / prev_val)
                bands.append({"date": ds,
                              "lo": round(base_nav * (1.0 + lo_b), 6),
                              "hi": round(base_nav * (1.0 + hi_b), 6)})
            prev_val = v_close
    if not bands:
        return None
    return dict(
        overnight_cum=round(on_cum - 1.0, 6),
        intraday_cum=round(id_cum - 1.0, 6),
        mdd_close=round(mdd_close, 6),
        mdd_intraday_bound=round(mdd_bound, 6),
        bands=bands,
    )
