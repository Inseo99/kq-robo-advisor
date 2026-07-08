"""워크포워드 최적화(ML) 결과를 엑셀 워크북으로 정리 (backtest-ksj).

전제: optimize_us_sec_ml.py 실행 후 (ml_metrics.csv, ml_chosen_params.csv, period_records_*_ml.csv).
비교 대상: 같은 미국 섹터ETF 비최적화(s*_us_sec) + S&P500 벤치마크(metrics_all.csv).

시트:
  1) 요약(ML)          - 6개 ML전략 전체OOS vs 비최적화(non-ML) vs S&P500, ML초과분
  2) 폴드별_선택파라미터 - 폴드마다 IS에서 고른 파라미터 + IS CAGR -> OOS CAGR(과최적화 갭)
  3~8) 전략별 시트       - 폴드별 OOS 성과 + 그때 선택된 파라미터 + S&P500 벤치 비교 + 스티칭 전체OOS

실행: python make_ml_workbook.py
출력: results/backtest_summary_ml.xlsx
"""

from __future__ import annotations

import os

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import config as C

RESULTS = C.RESULTS
OUT_PATH = os.path.join(RESULTS, "backtest_summary_ml.xlsx")
BENCH_US = C.BENCHMARK_US_CODE
# 스티칭 OOS 구간 = ML 폴드의 첫 OOS 시작 ~ 마지막 OOS 끝 (3폴드: 2021~2026)
FULL_OOS = (C.ML_FOLDS[0]["oos"][0], C.ML_FOLDS[-1]["oos"][1])

PARAM_COLS = ["mom_lookback_m", "top_n", "t2_ma_m",
              "t1_trend_ma_m", "t1_vix_th", "t1_credit_z_th", "t1_min_on"]
PARAM_LABEL = {"mom_lookback_m": "모멘텀룩백(M)", "top_n": "보유종목수", "t2_ma_m": "t2 MA(M)",
               "t1_trend_ma_m": "t1 추세MA(M)", "t1_vix_th": "t1 VIX임계",
               "t1_credit_z_th": "t1 신용z임계", "t1_min_on": "t1 최소켜짐"}

HEAD_FILL = PatternFill("solid", fgColor="305496")
HEAD_FONT = Font(color="FFFFFF", bold=True)
TITLE_FONT = Font(bold=True, size=13)
SUB_FONT = Font(bold=True)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BENCH_FILL = PatternFill("solid", fgColor="FFF2CC")
ML_FILL = PatternFill("solid", fgColor="E2EFDA")


def _hdr(ws, row, n):
    for c in range(1, n + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL; cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _drow(ws, row, n, fill=None):
    for c in range(1, n + 1):
        cell = ws.cell(row=row, column=c)
        cell.border = BORDER
        if fill:
            cell.fill = fill
        if c >= 3:
            cell.alignment = Alignment(horizontal="right")


def _fit(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _pct(x):
    return None if x is None or pd.isna(x) else round(float(x) * 100, 2)


def _num(x, d=2):
    return None if x is None or pd.isna(x) else round(float(x), d)


def _fmt_param(v):
    if v is None or pd.isna(v):
        return "-"
    f = float(v)
    return int(f) if f == int(f) else round(f, 2)


def load():
    mlm = pd.read_csv(os.path.join(RESULTS, "ml_metrics.csv"))
    par = pd.read_csv(os.path.join(RESULTS, "ml_chosen_params.csv"))
    cmp = pd.read_csv(os.path.join(RESULTS, "ml_compare.csv"))
    return mlm, par, cmp


def _mrow(df, code, cad, fold, seg="OOS"):
    q = df[(df["code"] == code) & (df["cadence"] == cad) & (df["fold"] == fold) & (df["segment"] == seg)]
    return q.iloc[0] if not q.empty else None


def cash_weight(code, start, end):
    path = os.path.join(RESULTS, f"period_records_{code}.csv")
    if not os.path.exists(path):
        return None
    pr = pd.read_csv(path, parse_dates=["exit"])
    pr = pr[(pr["exit"] >= pd.Timestamp(start)) & (pr["exit"] <= pd.Timestamp(end))]
    return None if pr.empty else float((pr["n_hold"] == 0).mean())


# ── 시트 1: 요약(ML) ─────────────────────────────────────────────────────
def build_summary(wb, cmp):
    ws = wb.create_sheet("요약(ML)")
    is_desc = ", ".join(f"{f['is'][0][:4]}~{f['is'][1][:4]}" for f in C.ML_FOLDS)
    ws["A1"] = (f"미국 섹터ETF 워크포워드 최적화(ML) — {len(C.ML_FOLDS)}폴드(확장형 IS: {is_desc}) · "
                f"스티칭 OOS {FULL_OOS[0]}~{FULL_OOS[1]}")
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("폴드별 IS에서 CAGR 최대화로 파라미터 최적화 후 OOS 적용. non-ML/벤치마크는 같은 스티칭 구간으로 공정 비교. "
                "⚠️ 2025~2026 합성 데이터로 절대수치 왜곡. 상대 비교로 해석.")
    ws["A2"].font = Font(italic=True, color="C00000")

    headers = ["ML 전략", "캐던스", "위험회피", "CAGR", "MDD", "Sharpe", "Calmar", "매매횟수", "현금비중%",
               "(비교)non-ML CAGR", "ML초과(CAGR)", "(벤치)S&P500 CAGR", "벤치초과(CAGR)"]
    hr = 4
    for j, h in enumerate(headers, 1):
        ws.cell(row=hr, column=j, value=h)
    _hdr(ws, hr, len(headers))

    var_lbl = {"s1": "없음", "s2": "t1", "s3": "t2"}
    r = hr + 1
    ben_by_cad = {}
    for _, row in cmp.iterrows():
        cad = row["cadence"]
        ben_by_cad[cad] = row  # 벤치는 모든 전략 동일(같은 구간)
        vals = [row["code"], "월간" if cad == "M" else "주간", var_lbl.get(row["variant"], row["variant"]),
                _pct(row["ml_cagr"]), _pct(row["ml_mdd"]), _num(row["ml_sharpe"]), _num(row["ml_calmar"]),
                int(row["ml_trades"]), _num((row["ml_cash"] or 0) * 100, 1) if not pd.isna(row["ml_cash"]) else None,
                _pct(row["nonml_cagr"]), _pct(row["ml_cagr"] - row["nonml_cagr"]),
                _pct(row["bench_cagr"]), _pct(row["ml_cagr"] - row["bench_cagr"])]
        for j, v in enumerate(vals, 1):
            ws.cell(row=r, column=j, value=v)
        _drow(ws, r, len(headers), ML_FILL)
        for col in (11, 13):
            cell = ws.cell(row=r, column=col)
            if cell.value is not None:
                cell.font = Font(color="1F7A1F" if cell.value >= 0 else "C00000", bold=True)
        r += 1

    for cad, brow in ben_by_cad.items():
        vals = [f"S&P500({'월간' if cad=='M' else '주간'} 환산)", "-", "-",
                _pct(brow["bench_cagr"]), _pct(brow["bench_mdd"]), _num(brow["bench_sharpe"]),
                _num(brow["bench_calmar"]), 0, 0.0, None, None, None, None]
        for j, v in enumerate(vals, 1):
            ws.cell(row=r, column=j, value=v)
        _drow(ws, r, len(headers), BENCH_FILL)
        r += 1

    ws.freeze_panes = "A5"
    _fit(ws, [16, 8, 8, 9, 9, 9, 9, 9, 10, 16, 13, 16, 14])


# ── 시트 2: 폴드별 선택 파라미터 ─────────────────────────────────────────
def build_params(wb, par):
    ws = wb.create_sheet("폴드별_선택파라미터")
    ws["A1"] = "폴드별 IS 최적화로 선택된 파라미터 (목적함수=CAGR) + IS→OOS 성과"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("IS CAGR(최적화 in-sample)과 OOS CAGR(실현)의 격차가 크면 과최적화 신호. "
                "파라미터가 폴드마다 바뀌므로 이 OOS 트랙은 '단일 연속 실행'과 다름(=IS/OOS 분할이 실체를 가짐).")
    ws["A2"].font = Font(italic=True, color="808080")

    headers = ["ML 전략", "폴드", "IS 구간", "OOS 구간"] + [PARAM_LABEL[c] for c in PARAM_COLS] + \
              ["IS CAGR", "OOS CAGR", "IS→OOS 갭", "OOS MDD", "OOS Sharpe"]
    hr = 4
    for j, h in enumerate(headers, 1):
        ws.cell(row=hr, column=j, value=h)
    _hdr(ws, hr, len(headers))

    r = hr + 1
    for _, row in par.iterrows():
        pvals = [_fmt_param(row.get(c)) for c in PARAM_COLS]
        gap = None
        if not pd.isna(row["is_cagr"]) and not pd.isna(row["oos_cagr"]):
            gap = _pct(row["oos_cagr"] - row["is_cagr"])
        vals = [row["code"], row["fold"], f"{row['is_start']}~{row['is_end']}",
                f"{row['oos_start']}~{row['oos_end']}"] + pvals + \
               [_pct(row["is_cagr"]), _pct(row["oos_cagr"]), gap,
                _pct(row["oos_mdd"]), _num(row["oos_sharpe"])]
        for j, v in enumerate(vals, 1):
            ws.cell(row=r, column=j, value=v)
        _drow(ws, r, len(headers))
        gcell = ws.cell(row=r, column=len(headers) - 2)  # IS→OOS 갭
        if gcell.value is not None:
            gcell.font = Font(color="C00000" if gcell.value < 0 else "1F7A1F")
        r += 1

    ws.freeze_panes = "E5"
    _fit(ws, [15, 7, 20, 20] + [11] * len(PARAM_COLS) + [9, 9, 10, 9, 10])


# ── 시트 3~8: 전략별 상세 ────────────────────────────────────────────────
def build_strategy(wb, strat, mlm, par):
    code, cad, var = strat["code"], strat["cadence"], strat["variant"]
    mlc = f"{code}_ml"
    ws = wb.create_sheet(mlc[:31])
    ws["A1"] = f"{mlc} — {strat['label']} · 워크포워드 최적화(ML)"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = "각 폴드 OOS는 그 폴드 IS에서 CAGR 최대화로 고른 파라미터로 실행. 하단 전체OOS=4폴드 스티칭. 벤치=S&P500."
    ws["A2"].font = Font(italic=True, color="808080")
    ws.merge_cells("A2:P2")

    headers = ["폴드", "OOS 구간"] + [PARAM_LABEL[c] for c in PARAM_COLS] + \
              ["IS CAGR", "OOS CAGR", "OOS MDD", "Sharpe", "Calmar", "벤치 CAGR", "벤치초과"]
    hr = 4
    for j, h in enumerate(headers, 1):
        ws.cell(row=hr, column=j, value=h)
    _hdr(ws, hr, len(headers))

    r = hr + 1
    psub = par[par["code"] == mlc]
    for fold in C.ML_FOLDS:
        prow = psub[psub["fold"] == fold["name"]]
        ben = _mrow(mlm, BENCH_US, cad, fold["name"], "OOS")
        if prow.empty:
            continue
        pr = prow.iloc[0]
        pvals = [_fmt_param(pr.get(c)) for c in PARAM_COLS]
        exc = None
        if ben is not None and not pd.isna(pr["oos_cagr"]):
            exc = _pct(pr["oos_cagr"] - ben["cagr"])
        vals = [fold["name"], f"{pr['oos_start']}~{pr['oos_end']}"] + pvals + \
               [_pct(pr["is_cagr"]), _pct(pr["oos_cagr"]), _pct(pr["oos_mdd"]),
                _num(pr["oos_sharpe"]), _num(pr["oos_calmar"]),
                _pct(ben["cagr"]) if ben is not None else None, exc]
        for j, v in enumerate(vals, 1):
            ws.cell(row=r, column=j, value=v)
        _drow(ws, r, len(headers))
        ec = ws.cell(row=r, column=len(headers))
        if ec.value is not None:
            ec.font = Font(color="1F7A1F" if ec.value >= 0 else "C00000", bold=True)
        r += 1

    # 스티칭 전체 OOS 행
    ml = _mrow(mlm, mlc, cad, "ALL_OOS")
    ben = _mrow(mlm, BENCH_US, cad, "ALL_OOS")
    if ml is not None:
        exc = _pct(ml["cagr"] - ben["cagr"]) if ben is not None else None
        vals = ["전체 OOS", f"{FULL_OOS[0]}~{FULL_OOS[1]}"] + ["(폴드별 상이)"] + [""] * (len(PARAM_COLS) - 1) + \
               [None, _pct(ml["cagr"]), _pct(ml["mdd"]), _num(ml["sharpe"]), _num(ml["calmar"]),
                _pct(ben["cagr"]) if ben is not None else None, exc]
        for j, v in enumerate(vals, 1):
            ws.cell(row=r, column=j, value=v)
        _drow(ws, r, len(headers), ML_FILL)
        ec = ws.cell(row=r, column=len(headers))
        if ec.value is not None:
            ec.font = Font(color="1F7A1F" if ec.value >= 0 else "C00000", bold=True)

    ws.freeze_panes = "A5"
    _fit(ws, [9, 20] + [10] * len(PARAM_COLS) + [9, 9, 9, 8, 8, 9, 9])


def main():
    from run import _utf8_stdout
    _utf8_stdout()
    mlm, par, cmp = load()
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    build_summary(wb, cmp)
    build_params(wb, par)
    for strat in [s for s in C.STRATEGIES if s["selection"] == "us_sec"]:
        build_strategy(wb, strat, mlm, par)
    wb.save(OUT_PATH)
    print(f"[done] {OUT_PATH}")


if __name__ == "__main__":
    main()
