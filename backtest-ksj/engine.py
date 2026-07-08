"""백테스트 엔진 (backtest-ksj): 리밸런싱 루프 + 거래비용 + 현금."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data as D
import signals as S


# ── eval / exec 날짜 생성 ────────────────────────────────────────────────
def build_eval_dates(cadence: str, trading_index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """월간=각 월의 마지막 거래일, 주간=각 주(월~금)의 마지막 거래일(금요일 기준)."""
    idx = pd.DatetimeIndex(trading_index).sort_values()
    s = pd.Series(idx, index=idx)
    rule = "ME" if cadence == "M" else "W-FRI"
    last = s.resample(rule).last().dropna()
    return list(last.values.astype("datetime64[ns]"))


def next_trading_day(trading_index: pd.DatetimeIndex, date) -> pd.Timestamp | None:
    """date 이후 첫 거래일(익영업일 시초가 실행일)."""
    d = pd.Timestamp(date)
    after = trading_index[trading_index > d]
    return after[0] if len(after) else None


# ── 실행가/보유수익 헬퍼 ────────────────────────────────────────────────
def _open_asof(open_ff: pd.DataFrame, date):
    return S.asof_row(open_ff, date)


def _mom_step(panels: dict, d_eval, d_exec, d_next, invested, cash_period):
    """모멘텀20 선정: 시초가→시초가 동일가중. (w_new, gross, n_hold) 반환."""
    close_ff = panels["close_ff"]
    adj_close = panels["adj_close"]
    open_ff = panels["open_ff"]
    mcap_hist = panels["mcap_hist"]

    if invested >= 1.0:
        selected = S.select_top(close_ff, adj_close, d_eval,
                                 D.top_mcap_at(mcap_hist, d_eval, C.UNIVERSE_SIZE))
    else:
        selected = []

    o_entry = _open_asof(open_ff, d_exec)
    o_exit = _open_asof(open_ff, d_next)
    valid = []
    if selected and o_entry is not None:
        for t in selected:
            pe = o_entry.get(t)
            if pe is not None and np.isfinite(pe) and pe > 0:
                valid.append(t)

    if not valid:
        return {"__CASH__": 1.0}, cash_period, 0

    w_new = {t: 1.0 / len(valid) for t in valid}
    rets = []
    for t in valid:
        p0 = float(o_entry.get(t))
        p1 = o_exit.get(t) if o_exit is not None else None
        if p1 is None or not np.isfinite(p1) or p1 <= 0:
            c = S.asof_row(close_ff, d_next)          # 청산 시초가 결측 -> 종가 근사(상폐 등)
            p1 = c.get(t) if c is not None else None
        if p1 is None or not np.isfinite(p1) or p1 <= 0:
            continue
        rets.append(p1 / p0 - 1.0)
    gross = float(np.mean(rets)) if rets else cash_period
    return w_new, gross, len(valid)


def _kd200_step(kodex_px: pd.Series, d_exec, d_next, invested, cash_period):
    """KODEX200 단일 보유: 종가→종가(다음 실행일). (w_new, gross, n_hold) 반환.

    069500 시초가 시계열이 없어 종가 기준으로 보유수익을 계산한다(실행 지연은 동일하게 유지).
    """
    if invested >= 1.0:
        p0 = S.asof_val(kodex_px, d_exec)
        p1 = S.asof_val(kodex_px, d_next)
        if p0 is not None and p1 is not None and p0 > 0 and p1 > 0:
            return {"069500": 1.0}, p1 / p0 - 1.0, 1
    return {"__CASH__": 1.0}, cash_period, 0


def _sp500_step(sp500_px: pd.Series, d_exec, d_next, invested, cash_period):
    """S&P500 단일 보유: 종가→종가. d_exec==d_eval(지연없는 즉시 진입, 2026-07-08 사용자 확정)."""
    if invested >= 1.0:
        p0 = S.asof_val(sp500_px, d_exec)
        p1 = S.asof_val(sp500_px, d_next)
        if p0 is not None and p1 is not None and p0 > 0 and p1 > 0:
            return {"SP500": 1.0}, p1 / p0 - 1.0, 1
    return {"__CASH__": 1.0}, cash_period, 0


def _us_sec_step(panels: dict, d_eval, d_exec, d_next, invested, cash_period):
    """미국 섹터ETF 모멘텀 상위3 동일가중: 종가→종가, 즉시 진입(지연없음).

    시가 시계열이 없어 종가만 사용. d_exec==d_eval(평가에 쓴 종가로 바로 진입,
    2026-07-08 사용자 확정). PIT: 상장 전(NaN) 티커는 momentum()/active_mask()가 자동 제외.
    """
    etf_ff = panels["sector_etf_ff"]
    etf_raw = panels["sector_etf"]

    if invested >= 1.0:
        mom = S.momentum(etf_ff, d_eval)
        if mom.empty:
            selected = []
        else:
            elig = S.active_mask(etf_raw, d_eval)
            cand = mom[[t for t in mom.index if elig.get(t, False)]]
            selected = (cand.nlargest(min(C.TOP_N_US_SEC, len(cand))).index.tolist()
                        if not cand.empty else [])
    else:
        selected = []

    p_entry = S.asof_row(etf_ff, d_exec)
    valid = []
    if selected and p_entry is not None:
        for t in selected:
            pe = p_entry.get(t)
            if pe is not None and np.isfinite(pe) and pe > 0:
                valid.append(t)

    if not valid:
        return {"__CASH__": 1.0}, cash_period, 0

    w_new = {t: 1.0 / len(valid) for t in valid}
    p_exit = S.asof_row(etf_ff, d_next)
    rets = []
    for t in valid:
        p0 = float(p_entry.get(t))
        p1 = p_exit.get(t) if p_exit is not None else None
        if p1 is None or not np.isfinite(p1) or p1 <= 0:
            continue
        rets.append(p1 / p0 - 1.0)
    gross = float(np.mean(rets)) if rets else cash_period
    return w_new, gross, len(valid)


def run_strategy(strat: dict, panels: dict) -> pd.DataFrame:
    """단일 전략 실행 -> 기간별 레코드 DataFrame.

    반환 컬럼: entry(실행일), exit(다음실행일=수익실현일), invested, n_hold,
              turnover, cost, gross, net, buys, sells
    """
    cadence, variant = strat["cadence"], strat["variant"]
    selection = strat.get("selection", "mom20")
    kodex_daily = panels["kodex_daily"]
    vix = panels["vix"]
    credit = panels["credit"]
    sp500 = panels.get("sp500")

    # 종목선정별 실행 캘린더: mom20/kd200(국내)는 국내 거래일+익영업일 지연,
    # us_sec/sp500(미국)은 각 소스 자체 캘린더+지연없는 즉시 진입(2026-07-08 사용자 확정)
    if selection == "us_sec":
        tidx = panels["sector_etf_ff"].index
        lag = False
    elif selection == "sp500":
        tidx = sp500.index
        lag = False
    else:
        tidx = panels["adj_close"].index
        lag = True

    eval_dates = build_eval_dates(cadence, tidx)
    eval_dates = [pd.Timestamp(d) for d in eval_dates]

    # t2(s3) 추세신호 기준 지수: 미국 종목선정은 S&P500, 그 외 KODEX200(기본)
    t2_index = sp500 if selection in ("us_sec", "sp500") else None
    gate, _detail = S.build_gate(cadence, variant, eval_dates, kodex_daily, vix, credit, sp500, t2_index)
    cash_period = C.CASH_MONTHLY if cadence == "M" else C.CASH_WEEKLY

    # 각 eval -> exec: 국내(mom20/kd200)는 익영업일, 미국(us_sec/sp500)은 지연없이 eval 당일
    if lag:
        execs = [next_trading_day(tidx, d) for d in eval_dates]
    else:
        execs = list(eval_dates)

    rows = []
    prev_w = {}   # 직전 타깃 비중 (코드->w), 현금은 '__CASH__'
    for i in range(len(eval_dates) - 1):
        d_eval = eval_dates[i]
        d_exec = execs[i]
        d_next = execs[i + 1]
        if d_exec is None or d_next is None:
            continue

        invested = gate.get(d_eval, 1.0)

        # 종목선정 방식별 타깃비중·보유수익 (다른 조건은 동일)
        if selection == "kd200":
            w_new, gross, n_hold = _kd200_step(kodex_daily, d_exec, d_next, invested, cash_period)
        elif selection == "sp500":
            w_new, gross, n_hold = _sp500_step(sp500, d_exec, d_next, invested, cash_period)
        elif selection == "us_sec":
            w_new, gross, n_hold = _us_sec_step(panels, d_eval, d_exec, d_next, invested, cash_period)
        else:
            w_new, gross, n_hold = _mom_step(panels, d_eval, d_exec, d_next, invested, cash_period)

        # 회전율 & 비용 (왕복 0.5% = 편도 0.25% * Σ|Δw|)
        keys = set(w_new) | set(prev_w)
        turnover = sum(abs(w_new.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys)
        cost = C.COST_ONE_WAY * turnover
        # 매매횟수: 편입(신규 매수) + 편출(전량 매도) 종목 수
        new_stocks = set(w_new) - {"__CASH__"}
        old_stocks = set(prev_w) - {"__CASH__"}
        buys = len(new_stocks - old_stocks)
        sells = len(old_stocks - new_stocks)

        net = (1.0 + gross) * (1.0 - cost) - 1.0

        rows.append({
            "eval": d_eval, "entry": d_exec, "exit": d_next,
            "invested": invested, "n_hold": n_hold,
            "turnover": turnover, "cost": cost,
            "gross": gross, "net": net, "buys": buys, "sells": sells,
        })
        prev_w = w_new

    df = pd.DataFrame(rows)
    if not df.empty:
        df["equity"] = (1.0 + df["net"]).cumprod()
    return df
