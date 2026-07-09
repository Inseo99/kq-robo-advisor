"""미국 섹터ETF 전략 폴드별 워크포워드 최적화 (backtest-ksj).

목적: IS/OOS 분할을 '진짜' 의미 있게 만든다.
  - 각 폴드의 IS 구간에서 그리드 서치로 파라미터를 최적화(목적함수=CAGR, 2026-07-08 사용자 확정)
  - 그 파라미터를 같은 폴드의 OOS 구간에 적용해 '실현' 성과 측정
  - 4개 폴드의 OOS를 이어붙여(stitch) 워크포워드 최적화 트랙 완성
파라미터가 폴드마다 바뀌므로 이 OOS 트랙은 '단일 연속 실행'과 달라진다(= IS/OOS 분할이 실체를 가짐).

폴드: config.ML_FOLDS(IS 넓힌 확장형 3폴드, 2026-07-08 사용자 요청) — IS 7/9/11년, OOS 2021~2026.
대상: 종목선정=us_sec 6종만 (s1/s2/s3 × 월간/주간). 코드에 _ml 접미사.
최적화 변수(2026-07-08 사용자 확정: 선정변수 + t2 + t1 임계값 = 최대 범위):
  - 공통(선정): mom_lookback_m ∈ {3,6,9,12}, top_n ∈ {2,3,4,5}
  - s3(t2):    t2_ma_m ∈ {6,8,10,12}
  - s2(t1):    t1_trend_ma_m ∈ {6,9,12}, t1_vix_th ∈ {15,18.6,22},
               t1_credit_z_th ∈ {1.5,1.78,2.0}, t1_min_on ∈ {1,2}

실행: python optimize_us_sec_ml.py
출력: results/period_records_{code}_ml.csv, ml_chosen_params.csv, ml_metrics.csv,
      ml_compare.csv(같은 스티칭 구간에서 ml vs non-ml vs S&P500 공정비교)
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

import config as C
import data as D
import engine as E
import metrics as M
import signals as S
from run import load_panels, _utf8_stdout

# ── 파라미터 그리드 ──────────────────────────────────────────────────────
GRID_SEL = {"mom_lookback_m": [3, 6, 9, 12], "top_n": [2, 3, 4, 5]}
GRID_T2 = {"t2_ma_m": [6, 8, 10, 12]}
GRID_T1 = {"t1_trend_ma_m": [6, 9, 12], "t1_vix_th": [15.0, 18.6, 22.0],
           "t1_credit_z_th": [1.5, 1.78, 2.0], "t1_min_on": [1, 2]}

US_SEC_STRATS = [s for s in C.STRATEGIES if s["selection"] == "us_sec"]


def param_grid(variant: str):
    keys = list(GRID_SEL)
    grids = [GRID_SEL[k] for k in keys]
    if variant == "s3":
        keys += list(GRID_T2)
        grids += [GRID_T2[k] for k in GRID_T2]
    elif variant == "s2":
        keys += list(GRID_T1)
        grids += [GRID_T1[k] for k in GRID_T1]
    for combo in itertools.product(*grids):
        yield dict(zip(keys, combo))


def grid_size(variant: str) -> int:
    n = 1
    for v in GRID_SEL.values():
        n *= len(v)
    if variant == "s3":
        for v in GRID_T2.values():
            n *= len(v)
    elif variant == "s2":
        for v in GRID_T1.values():
            n *= len(v)
    return n


# ── 폴드별 최적화 (factored 스코어러: 게이트/선정 선계산 후 조합) ─────────
# 정확성: 이 스코어러는 engine._us_sec_step / build_gate 회계(동일가중 종가→종가, 회전율
# Σ|Δw|×0.25%, 위험회피 시 현금)를 그대로 복제한다. IS 랭킹용이며, OOS 실현치는 공식
# engine.run_strategy(best_params)로 재계산하므로 미세한 차이가 있어도 최종 수치엔 무영향.
def _prep_fold(strat, panels, is_win):
    """IS 구간에서 선정/게이트 선계산. periods, 선정별 (holdings,gross), 게이트 신호소스 반환."""
    cad = strat["cadence"]
    etf_ff = panels["sector_etf_ff"]
    etf_raw = panels["sector_etf"]
    sp500 = panels["sp500"]
    cash_period = C.CASH_MONTHLY if cad == "M" else C.CASH_WEEKLY

    tidx = etf_ff.index
    evd = E.build_eval_dates(cad, tidx)
    evd = [pd.Timestamp(d) for d in evd]
    ws, we = pd.Timestamp(is_win[0]), pd.Timestamp(is_win[1])
    in_win = [d for d in evd if ws <= d <= we]
    after = [d for d in evd if d > we]
    evs = in_win + after[:1]              # 마지막 구간수익 실현용 1개 추가
    n_per = len(evs) - 1
    if n_per <= 1:
        return None

    # 선정 선계산: eval별 가격행/활성마스크, lookback별 모멘텀
    price_rows = {d: S.asof_row(etf_ff, d) for d in evs}
    act_masks = {d: S.active_mask(etf_raw, d) for d in evs[:-1]}
    mom_by_lb = {}
    for lb in GRID_SEL["mom_lookback_m"]:
        mom_by_lb[lb] = {d: S.momentum(etf_ff, d, lb, C.MOM_SKIP_M) for d in evs[:-1]}

    # 선정 조합별 (holdings[i], gross_invested[i])
    sel_cache = {}
    for lb in GRID_SEL["mom_lookback_m"]:
        for tn in GRID_SEL["top_n"]:
            holds, gross = [], []
            for i in range(n_per):
                d0, d1 = evs[i], evs[i + 1]
                mom = mom_by_lb[lb][d0]
                act = act_masks[d0]
                p0 = price_rows[d0]
                p1 = price_rows[d1]
                if mom.empty or p0 is None:
                    holds.append(()); gross.append(cash_period); continue
                cand = mom[[t for t in mom.index if act.get(t, False)]]
                names = cand.nlargest(min(tn, len(cand))).index.tolist() if not cand.empty else []
                valid, rets = [], []
                for t in names:
                    pe = p0.get(t)
                    if pe is None or not np.isfinite(pe) or pe <= 0:
                        continue
                    px = p1.get(t) if p1 is not None else None
                    if px is None or not np.isfinite(px) or px <= 0:
                        continue
                    valid.append(t); rets.append(px / pe - 1.0)
                if valid:
                    holds.append(tuple(valid)); gross.append(float(np.mean(rets)))
                else:
                    holds.append(()); gross.append(cash_period)
            sel_cache[(lb, tn)] = (holds, gross)

    # 게이트 신호 소스 선계산
    variant = strat["variant"]
    gate_ctx = {"variant": variant, "cash": cash_period}
    if variant == "s1":
        gate_ctx["invested"] = [1.0] * n_per
    elif variant == "s3":
        sp_r = S._resample(sp500, cad)
        inv_by_t2 = {}
        for t2 in GRID_T2["t2_ma_m"]:
            ma = sp_r.rolling(S._win(t2, cad)).mean()
            inv_by_t2[t2] = [1.0 if (S.asof_val(sp_r, evs[i]) is not None and
                                     S.asof_val(ma, evs[i]) is not None and
                                     S.asof_val(sp_r, evs[i]) > S.asof_val(ma, evs[i])) else 0.0
                             for i in range(n_per)]
        gate_ctx["inv_by_t2"] = inv_by_t2
    else:  # s2
        sp_r = S._resample(sp500, cad)
        trend_by = {}
        for tm in GRID_T1["t1_trend_ma_m"]:
            ma = sp_r.rolling(S._win(tm, cad)).mean()
            trend_by[tm] = [(S.asof_val(sp_r, evs[i]) is not None and
                             S.asof_val(ma, evs[i]) is not None and
                             S.asof_val(sp_r, evs[i]) < S.asof_val(ma, evs[i]))
                            for i in range(n_per)]
        vix_r = S._resample(panels["vix"], cad)
        vix_val = [S.asof_val(vix_r, evs[i]) for i in range(n_per)]
        c = panels["credit"].sort_index()
        cz = (c - c.rolling(C.T1_CREDIT_Z_WIN_D).mean()) / c.rolling(C.T1_CREDIT_Z_WIN_D).std()
        cz_val = [S.asof_val(cz, evs[i]) for i in range(n_per)]
        gate_ctx.update({"trend_by": trend_by, "vix_val": vix_val, "cz_val": cz_val})

    # metrics.strategy_window_metrics와 동일하게 exit(=evs[i+1]) ∈ [is_s, is_e]만 집계
    incl = [i for i in range(n_per) if evs[i + 1] <= we]
    return {"cad": cad, "evs": evs, "n_per": n_per, "cash": cash_period,
            "sel_cache": sel_cache, "gate_ctx": gate_ctx, "incl": incl}


def _gate_invested(gate_ctx, params, n_per):
    v = gate_ctx["variant"]
    if v == "s1":
        return gate_ctx["invested"]
    if v == "s3":
        return gate_ctx["inv_by_t2"][params["t2_ma_m"]]
    tr = gate_ctx["trend_by"][params["t1_trend_ma_m"]]
    vth, zth, mon = params["t1_vix_th"], params["t1_credit_z_th"], params["t1_min_on"]
    vv, zz = gate_ctx["vix_val"], gate_ctx["cz_val"]
    out = []
    for i in range(n_per):
        n_on = int(tr[i]) + int(vv[i] is not None and vv[i] > vth) + int(zz[i] is not None and zz[i] > zth)
        out.append(0.0 if n_on >= mon else 1.0)
    return out


def _score(prep, params) -> tuple[float, float] | None:
    """조합 조립 -> (CAGR, Sharpe). engine 회계(회전율/비용/현금) 복제."""
    n = prep["n_per"]
    cash = prep["cash"]
    incl = prep["incl"]
    if not incl:
        return None
    holds, gross_inv = prep["sel_cache"][(params["mom_lookback_m"], params["top_n"])]
    invested = _gate_invested(prep["gate_ctx"], params, n)
    prev_w = {}
    all_nets = []
    for i in range(n):                       # 회전율은 전체 순서로 계산(engine과 동일)
        if invested[i] >= 1.0 and holds[i]:
            hs = holds[i]; w = 1.0 / len(hs)
            w_new = {t: w for t in hs}; gross = gross_inv[i]
        else:
            w_new = {"__CASH__": 1.0}; gross = cash
        keys = set(w_new) | set(prev_w)
        turnover = sum(abs(w_new.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys)
        cost = C.COST_ONE_WAY * turnover
        all_nets.append((1.0 + gross) * (1.0 - cost) - 1.0)
        prev_w = w_new
    nets = np.array([all_nets[i] for i in incl])   # 집계는 IS 창 내(exit<=is_e) 구간만
    total = float(np.prod(1.0 + nets) - 1.0)
    days = (prep["evs"][incl[-1] + 1] - prep["evs"][incl[0]]).days
    years = max(days / 365.25, 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0
    ppy = C.PERIODS_PER_YEAR[prep["cad"]]
    sd = float(nets.std(ddof=1)) if len(nets) > 1 else 0.0
    sharpe = float((nets - cash).mean() / sd * np.sqrt(ppy)) if sd > 0 else -1e9
    return cagr, sharpe


def optimize_fold(strat, panels, is_win) -> tuple[dict, dict]:
    """IS 구간 그리드 서치 -> (best_params, {cagr,sharpe}). 목적함수=CAGR(동점시 Sharpe)."""
    prep = _prep_fold(strat, panels, is_win)
    if prep is None:
        return {}, {}
    best = None
    for params in param_grid(strat["variant"]):
        sc = _score(prep, params)
        if sc is None or pd.isna(sc[0]):
            continue
        if best is None or sc > best[0]:
            best = (sc, dict(params))
    if best is None:
        return {}, {}
    return best[1], {"cagr": best[0][0], "sharpe": best[0][1]}


def run_all(folds=None):
    """워크포워드 최적화 실행. folds 미지정 시 config.ML_FOLDS(IS 넓힌 3폴드) 사용."""
    folds = folds or C.ML_FOLDS
    # 스티칭 OOS 구간 = 첫 폴드 OOS 시작 ~ 마지막 폴드 OOS 끝
    stitch_oos = (folds[0]["oos"][0], folds[-1]["oos"][1])
    panels = load_panels()
    sp500 = panels["sp500"]
    all_param_recs = []      # 폴드별 선택 파라미터 + IS/OOS 성과
    all_metric_recs = []     # ml_metrics: 폴드 OOS + 스티칭 전체 OOS (+ 벤치 SP500)
    compare_recs = []        # ml_compare: 같은 스티칭 구간에서 ml vs non-ml vs 벤치
    is_spans = ", ".join(f"{f['is'][0][:4]}~{f['is'][1][:4]}" for f in folds)
    print(f"[fold] {len(folds)}폴드, 스티칭 OOS={stitch_oos[0]}~{stitch_oos[1]}, IS폭=[{is_spans}]")

    for strat in US_SEC_STRATS:
        code, cad, variant = strat["code"], strat["cadence"], strat["variant"]
        ml_code = f"{code}_ml"
        gs = grid_size(variant)
        print(f"\n[opt] {ml_code} (variant={variant}, grid={gs}/fold) ...")

        oos_slices = []
        for fold in folds:
            best_params, is_m = optimize_fold(strat, panels, fold["is"])
            if not best_params:
                print(f"   {fold['name']}: (IS 데이터 부족, 스킵)")
                continue
            # OOS 실현: 최적 파라미터로 전체 실행 후 OOS 구간 슬라이스
            df_full = E.run_strategy(strat, panels, params=best_params)
            oos_s, oos_e = fold["oos"]
            oos_m = M.strategy_window_metrics(df_full, cad, oos_s, oos_e)
            oos_slice = df_full[(df_full["exit"] >= pd.Timestamp(oos_s)) &
                                (df_full["exit"] <= pd.Timestamp(oos_e))].copy()
            oos_slice["fold"] = fold["name"]
            oos_slices.append(oos_slice)

            pstr = ", ".join(f"{k}={v}" for k, v in best_params.items())
            print(f"   {fold['name']}: IS CAGR={is_m['cagr']*100:5.1f}% -> "
                  f"OOS CAGR={oos_m['cagr']*100:5.1f}% | {pstr}")

            all_param_recs.append({
                "code": ml_code, "cadence": cad, "fold": fold["name"],
                "is_start": fold["is"][0], "is_end": fold["is"][1],
                "oos_start": oos_s, "oos_end": oos_e,
                "is_cagr": is_m["cagr"], "oos_cagr": oos_m["cagr"],
                "oos_mdd": oos_m["mdd"], "oos_sharpe": oos_m["sharpe"], "oos_calmar": oos_m["calmar"],
                **best_params,
            })
            all_metric_recs.append({
                "code": ml_code, "cadence": cad, "fold": fold["name"], "segment": "OOS",
                "start": oos_s, "end": oos_e, **oos_m})
            # 해당 폴드 OOS 구간 벤치마크(S&P500) — ML 폴드 창 기준
            bm = M.benchmark_window_metrics(sp500, oos_s, oos_e)
            all_metric_recs.append({
                "code": C.BENCHMARK_US_CODE, "cadence": cad, "fold": fold["name"],
                "segment": "OOS", "start": oos_s, "end": oos_e, **bm})

        if not oos_slices:
            continue
        stitched = pd.concat(oos_slices, ignore_index=True).sort_values("exit").reset_index(drop=True)
        stitched["equity"] = (1.0 + stitched["net"]).cumprod()
        stitched.to_csv(os.path.join(C.RESULTS, f"period_records_{ml_code}.csv"),
                        index=False, encoding="utf-8-sig")

        full_m = M.strategy_window_metrics(stitched, cad, *stitch_oos)
        all_metric_recs.append({
            "code": ml_code, "cadence": cad, "fold": "ALL_OOS", "segment": "OOS",
            "start": stitch_oos[0], "end": stitch_oos[1], **full_m})
        print(f"   => 스티칭 전체 OOS({stitch_oos[0][:4]}~{stitch_oos[1][:4]}): "
              f"CAGR={full_m['cagr']*100:.1f}%  MDD={full_m['mdd']*100:.1f}%  "
              f"Sharpe={full_m['sharpe']:.2f}  Calmar={full_m['calmar']:.2f}")

        # 공정 비교(같은 스티칭 구간): non-ML(기본 파라미터) & 벤치마크
        df_nonml = E.run_strategy(strat, panels)      # 기본 파라미터 = 비최적화
        non_m = M.strategy_window_metrics(df_nonml, cad, *stitch_oos)
        ben_m = M.benchmark_window_metrics(sp500, *stitch_oos)
        cw = float((stitched["n_hold"] == 0).mean()) if not stitched.empty else None
        compare_recs.append({
            "code": ml_code, "cadence": cad, "variant": variant,
            "span_start": stitch_oos[0], "span_end": stitch_oos[1], "ml_cash": cw,
            "ml_cagr": full_m["cagr"], "ml_mdd": full_m["mdd"], "ml_sharpe": full_m["sharpe"],
            "ml_calmar": full_m["calmar"], "ml_trades": full_m["trades"],
            "nonml_cagr": non_m["cagr"], "nonml_mdd": non_m["mdd"],
            "nonml_sharpe": non_m["sharpe"], "nonml_calmar": non_m["calmar"],
            "bench_cagr": ben_m["cagr"], "bench_mdd": ben_m["mdd"],
            "bench_sharpe": ben_m["sharpe"], "bench_calmar": ben_m["calmar"],
        })

    # 스티칭 구간 벤치마크(ALL_OOS)
    ben_full = M.benchmark_window_metrics(sp500, *stitch_oos)
    for cad in ("M", "W"):
        all_metric_recs.append({
            "code": C.BENCHMARK_US_CODE, "cadence": cad, "fold": "ALL_OOS",
            "segment": "OOS", "start": stitch_oos[0], "end": stitch_oos[1], **ben_full})

    pd.DataFrame(all_param_recs).to_csv(
        os.path.join(C.RESULTS, "ml_chosen_params.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_metric_recs).to_csv(
        os.path.join(C.RESULTS, "ml_metrics.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(compare_recs).to_csv(
        os.path.join(C.RESULTS, "ml_compare.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[done] ml_chosen_params.csv, ml_metrics.csv, ml_compare.csv, "
          f"period_records_*_ml.csv -> {C.RESULTS}")
    return all_param_recs, all_metric_recs, compare_recs


if __name__ == "__main__":
    _utf8_stdout()
    run_all()
