"""24개 전략 결과를 엑셀 워크북(xlsx)으로 정리.

시트 구성:
  1) 요약        - 전략 24종 + 벤치마크(KODEX200/S&P500) 핵심지표(ALL_OOS) 한 눈에 비교
  2) 백테스트_조건 - 유니버스/비용/현금/위험회피(t1,t2)/폴드/미국판 변경사항 등 실행 조건
  3~26) 전략별 시트 - 개요 + 현금비중 + 폴드별(IS/OOS) 성과, 매 구간 해당 벤치마크 병기
                     (국내 mom20/kd200 -> KODEX200, 미국 us_sec/sp500 -> S&P500)

실행:
    python make_summary_workbook.py
출력:
    results/backtest_summary.xlsx
"""

from __future__ import annotations

import os

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import config as C

RESULTS = C.RESULTS
OUT_PATH = os.path.join(RESULTS, "backtest_summary.xlsx")
BENCH = C.BENCHMARK_CODE
BENCH_US = C.BENCHMARK_US_CODE

FOLD_SEG_ORDER = []
for f in C.FOLDS:
    FOLD_SEG_ORDER.append((f["name"], "IS"))
    FOLD_SEG_ORDER.append((f["name"], "OOS"))
FOLD_SEG_ORDER += [("ALL_OOS", "OOS"), ("FULL", "ALL")]

SEG_LABEL = {"IS": "IS(학습)", "OOS": "OOS(검증)", "ALL": "전체"}

SELECTION_DESC = {
    "mom20": "PIT 시총 상위300(KOSPI+KOSDAQ) -> 12-1 모멘텀 상위20 동일가중",
    "kd200": "KODEX200(069500) 1종목 단일 보유",
    "us_sec": "미국 섹터ETF 27종 유니버스 -> 12-1 모멘텀 상위3 동일가중 (2026-07-08 확정)",
    "sp500": "S&P500 1종목 단일 보유 (market_2000_2026.csv)",
}
# t2(s3)는 선정방식에 따라 기준지수가 다름(국내=KODEX200, 미국=S&P500)
T2_INDEX_DESC = {"mom20": "KODEX200", "kd200": "KODEX200", "us_sec": "S&P500", "sp500": "S&P500"}
VARIANT_DESC = {
    "s1": "위험회피 없음(항상 100% 투자)",
    "s2": "t1: (①S&P500<9M(39주)MA, ②VIX>18.6, ③미국신용스프레드 z(105거래일,~5개월)>1.78) "
          "중 2개 이상 켜지면 주식비중 0%(현금)",
    "s3": "t2: {idx} 종가 > 10M(43주) 이동평균 -> 100% 보유 / 아니면 전액 현금",
}
CADENCE_DESC = {"M": "월간(월말 평가)", "W": "주간(금요일 평가)"}
# 종목선정별 진입 규칙: 국내(mom20/kd200)=익영업일 시초가/종가, 미국(us_sec/sp500)=지연없는 즉시 종가진입
EXEC_DESC = {
    "mom20": "익월 첫 거래일(월) / 익영업일(주) 시초가 진입",
    "kd200": "익월 첫 거래일(월) / 익영업일(주) 종가 진입(시가 없음)",
    "us_sec": "지연 없이 평가일 종가 즉시 진입(시가 없음, 2026-07-08 확정)",
    "sp500": "지연 없이 평가일 종가 즉시 진입(시가 없음, 2026-07-08 확정)",
}


def variant_desc(variant: str, selection: str) -> str:
    d = VARIANT_DESC[variant]
    return d.format(idx=T2_INDEX_DESC[selection]) if "{idx}" in d else d


def bench_for(selection: str) -> str:
    return BENCH_US if selection in ("us_sec", "sp500") else BENCH


# ── 스타일 헬퍼 ──────────────────────────────────────────────────────────
HEAD_FILL = PatternFill("solid", fgColor="305496")
HEAD_FONT = Font(color="FFFFFF", bold=True)
SUB_FILL = PatternFill("solid", fgColor="D9E1F2")
SUB_FONT = Font(bold=True)
TITLE_FONT = Font(bold=True, size=13)
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
BENCH_FILL = PatternFill("solid", fgColor="FFF2CC")


def _style_header_row(ws, row, n_cols):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEAD_FILL
        cell.font = HEAD_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def _style_data_row(ws, row, n_cols, bench=False):
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.border = BORDER
        if bench:
            cell.fill = BENCH_FILL
        if c >= 3:
            cell.alignment = Alignment(horizontal="right")


def _autofit(ws, widths):
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _pct(x):
    return None if pd.isna(x) else round(float(x) * 100, 2)


def _num(x, d=2):
    return None if pd.isna(x) else round(float(x), d)


# ── 데이터 로드 ──────────────────────────────────────────────────────────
def load_data():
    mdf = pd.read_csv(os.path.join(RESULTS, "metrics_all.csv"))
    period_records = {}
    for strat in C.STRATEGIES:
        path = os.path.join(RESULTS, f"period_records_{strat['code']}.csv")
        period_records[strat["code"]] = pd.read_csv(path, parse_dates=["entry", "exit"])
    return mdf, period_records


def cash_weight(pr: pd.DataFrame, start=None, end=None) -> float | None:
    """현금비중 = n_hold==0(무투자) 리밸런싱 비중. [start,end]는 exit 기준 슬라이스."""
    df = pr
    if start is not None:
        df = df[(df["exit"] >= pd.Timestamp(start)) & (df["exit"] <= pd.Timestamp(end))]
    if df.empty:
        return None
    return float((df["n_hold"] == 0).mean())


def strat_row(mdf, code, cadence, fold, segment):
    q = mdf[(mdf["code"] == code) & (mdf["cadence"] == cadence) & (mdf["fold"] == fold) & (mdf["segment"] == segment)]
    if q.empty:
        return None
    return q.iloc[0]


def bench_row(mdf, cadence, fold, segment, bench_code=BENCH):
    return strat_row(mdf, bench_code, cadence, fold, segment)


# ── 시트 1: 요약 ─────────────────────────────────────────────────────────
SEL_SHORT = {"mom20": "모멘텀20", "kd200": "KODEX200", "us_sec": "미국섹터ETF", "sp500": "S&P500"}


def build_summary_sheet(wb, mdf, period_records):
    ws = wb.create_sheet("요약")
    ws["A1"] = "backtest-ksj 전략 24종 성과 요약 (전체 OOS: 2019-01 ~ 2026-05)"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = ("⚠️ 2025~2026 가격 데이터는 합성(synthetic)으로 부풀려져 있어 절대 CAGR은 비현실적임. 전략 간 상대 비교로 해석할 것. "
                "국내(mom20/kd200)는 KODEX200, 미국(us_sec/sp500)은 S&P500과 비교.")
    ws["A2"].font = Font(italic=True, color="C00000")

    headers = ["코드", "캐던스", "종목선정", "위험회피", "개요",
               "CAGR", "MDD", "Sharpe", "Calmar", "매매횟수",
               "누적거래비용%", "연환산비용%", "현금비중%(OOS)", "현금비중%(전체)"]
    header_row = 4
    for j, h in enumerate(headers, start=1):
        ws.cell(row=header_row, column=j, value=h)
    _style_header_row(ws, header_row, len(headers))

    r = header_row + 1
    for strat in C.STRATEGIES:
        code, cad, var, sel = strat["code"], strat["cadence"], strat["variant"], strat["selection"]
        row = strat_row(mdf, code, cad, "ALL_OOS", "OOS")
        pr = period_records[code]
        cw_oos = cash_weight(pr, "2019-01-01", "2026-05-31")
        cw_full = cash_weight(pr)
        overview = f"{SELECTION_DESC[sel]} | {variant_desc(var, sel)}"
        vals = [
            code, "월간" if cad == "M" else "주간",
            SEL_SHORT[sel],
            {"s1": "없음", "s2": "t1", "s3": "t2"}[var],
            overview,
            _pct(row["cagr"]) if row is not None else None,
            _pct(row["mdd"]) if row is not None else None,
            _num(row["sharpe"]) if row is not None else None,
            _num(row["calmar"]) if row is not None else None,
            int(row["trades"]) if row is not None else None,
            _num(row["cost_total_pct"]) if row is not None else None,
            _num(row["cost_ann_pct"]) if row is not None else None,
            _num((cw_oos or 0) * 100) if cw_oos is not None else None,
            _num((cw_full or 0) * 100) if cw_full is not None else None,
        ]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=r, column=j, value=v)
        _style_data_row(ws, r, len(headers))
        r += 1

    # 벤치마크 비교행: 국내(KODEX200) + 미국(S&P500), 월/주 환산 각각
    bench_defs = [(BENCH, "KODEX200", "KR"), (BENCH_US, "S&P500", "US")]
    for code, name, _tag in bench_defs:
        for cad in ("M", "W"):
            brow = bench_row(mdf, cad, "ALL_OOS", "OOS", code)
            if brow is None:
                continue
            vals = [f"{code}({'월간' if cad=='M' else '주간'} 환산)", "-", "-", "-", f"벤치마크: {name} 매수후보유",
                    _pct(brow["cagr"]), _pct(brow["mdd"]), _num(brow["sharpe"]), _num(brow["calmar"]),
                    0, 0.0, 0.0, 0.0, 0.0]
            for j, v in enumerate(vals, start=1):
                ws.cell(row=r, column=j, value=v)
            _style_data_row(ws, r, len(headers), bench=True)
            r += 1

    ws.freeze_panes = "A5"
    _autofit(ws, [14, 8, 12, 9, 65, 9, 9, 9, 9, 9, 12, 12, 14, 12])


# ── 시트 2: 백테스트 조건 ────────────────────────────────────────────────
def build_conditions_sheet(wb):
    ws = wb.create_sheet("백테스트_조건")
    ws["A1"] = "백테스트 실행 조건"
    ws["A1"].font = TITLE_FONT

    rows = [
        ("유니버스(국내)", "Point-in-Time 시가총액 상위 300 (KOSPI+KOSDAQ, 분기 시총 ffill, 생존편향 없음)"),
        ("종목선정 - mom20", SELECTION_DESC["mom20"]),
        ("종목선정 - kd200", SELECTION_DESC["kd200"] + " (시초가 시계열 없어 종가→종가로 보유수익 계산, 익영업일 지연 진입)"),
        ("모멘텀 정의", f"P(t-{C.MOM_SKIP_M}개월) / P(t-{C.MOM_LOOKBACK_M}개월) - 1  (최근 {C.MOM_SKIP_M}개월 제외). us_sec도 동일 정의."),
        ("실행(월간, 국내)", "월말 평가 -> 익월 첫 거래일 시초가 진입"),
        ("실행(주간, 국내)", "금요일 종가 평가 -> 익영업일 시초가 진입"),
        ("거래비용", f"왕복 {C.COST_ONE_WAY*2*100:.1f}% (편도 {C.COST_ONE_WAY*100:.2f}% × 회전율 Σ|Δw|). 모든 전략(국내/미국) 공통."),
        ("현금수익", f"연 {C.CASH_ANNUAL*100:.1f}% (월 {C.CASH_MONTHLY*100:.3f}% / 주 {C.CASH_WEEKLY*100:.3f}%, 위험회피 시 전액 현금). 공통."),
        ("벤치마크(국내)", f"{C.BENCHMARK_LABEL}"),
        ("", ""),
        ("t1 위험회피 (s2)", VARIANT_DESC["s2"] + " — 국내/미국 전략 공통 정의(원래부터 S&P500·VIX·미국신용 기준)"),
        ("t1 - ① 추세이탈", f"S&P500 종가 < {C.T1_TREND_MA_M}개월({C.T1_TREND_MA_W}주) 이동평균"),
        ("t1 - ② 공포 급등", f"VIX > {C.T1_VIX_TH}"),
        ("t1 - ③ 신용 경색", f"미국 신용스프레드 z점수(일별 {C.T1_CREDIT_Z_WIN_D}거래일≈5개월 롤링) > {C.T1_CREDIT_Z_TH}"),
        ("t1 - 판정", f"위 3개 중 {C.T1_MIN_ON}개 이상 켜지면 -> 주식비중 0%(전액 현금)"),
        ("t1 데이터 출처", "market_2000_2026.csv(ticker=sp500,vix), liquidity_2000_2026.csv(ticker=credit_spread)"),
        ("", ""),
        ("t2 추세추종 (s3)", "종가 > 10M(43주) 이동평균 -> 100% 보유 / 아니면 전액 현금. "
                          "기준지수: 국내(mom20/kd200)=KODEX200, 미국(us_sec/sp500)=S&P500 (2026-07-08 확정)"),
        ("t2 - 이동평균 창", f"{C.T2_MA_M}개월({C.T2_MA_W}주)"),
        ("", ""),
        ("Walk-forward 폴드", " / ".join(f"{f['name']}: IS {f['is'][0]}~{f['is'][1]}, OOS {f['oos'][0]}~{f['oos'][1]}" for f in C.FOLDS)),
        ("전체 OOS", "2019-01-01 ~ 2026-05-31 (4개 OOS 연속 접합. 파라미터 재추정 없어 단일 연속 실행과 스티칭 결과 동일)"),
        ("모멘텀 워밍업", "12개월 워밍업으로 실제 매매는 2015-02부터 (Fold1 IS 축소)"),
        ("", ""),
        ("★ 미국판 종목선정 (2026-07-08 추가)", "국내 종목/KODEX200을 미국 자산으로 교체한 버전. *_us_sec, *-sp500 코드."),
        ("종목선정 - us_sec", SELECTION_DESC["us_sec"]),
        ("종목선정 - sp500", SELECTION_DESC["sp500"]),
        ("us_sec 유니버스", "27종 ETF(AIQ,ARKG,ARKK,ARKX,BOTZ,CIBR,MCHI,PSQ,SOXX,SQQQ,TMF,TQQQ,USO,VGT,VIXY,"
                          "XBI,XLB,XLC,XLE,XLF,XLI,XLK,XLP,XLRE,XLU,XLV,XLY). 대부분 2014-01부터, 후발 상장분은 자연 편입(생존편향 없음)."),
        ("실행(미국)", "us_sec/sp500 모두 시가 없음 -> 지연 없이 평가일 종가에 즉시 진입(국내판의 '익영업일 진입'과 다른 규칙, 2026-07-08 확정)"),
        ("Top N(미국)", f"us_sec은 상위 {C.TOP_N_US_SEC}종목 동일가중(유니버스 27종뿐이라 국내 20종목과 다르게 축소, 2026-07-08 확정). "
                      "sp500은 국내 kd200과 동일하게 1종목 단일보유."),
        ("벤치마크(미국)", f"{C.BENCHMARK_US_LABEL} — us_sec/sp500 전략은 이 벤치마크와 비교 (2026-07-08 확정)"),
        ("", ""),
        ("⚠️ 데이터 주의", "가격 데이터 2025~2026 구간은 합성(synthetic)으로 크게 부풀려짐(예: KODEX200 약4배, 삼성전자 약6배). "
                       "2024년까지는 실제와 유사. Fold4·전체OOS 절대 CAGR은 비현실적이므로 전략 간 상대 비교로만 해석할 것."),
        ("현금비중 정의", "리밸런싱 시점 중 n_hold==0(무투자, 현금 100%)인 비율. 위험회피 발동 또는 종목선정 실패 시 발생."),
    ]
    r = 3
    for k, v in rows:
        ws.cell(row=r, column=1, value=k).font = SUB_FONT if k else Font()
        ws.cell(row=r, column=2, value=v)
        ws.cell(row=r, column=1).alignment = Alignment(vertical="top")
        ws.cell(row=r, column=2).alignment = Alignment(wrap_text=True, vertical="top")
        r += 1
    _autofit(ws, [20, 110])


# ── 시트 3~26: 전략별 상세 ───────────────────────────────────────────────
def build_strategy_sheet(wb, mdf, strat, pr):
    code = strat["code"]
    sel = strat["selection"]
    bcode = bench_for(sel)
    bname = "KODEX200" if bcode == BENCH else "S&P500"
    ws = wb.create_sheet(code[:31])

    ws["A1"] = f"{code} — {strat['label']}"
    ws["A1"].font = TITLE_FONT
    ws["A2"] = (f"캐던스: {CADENCE_DESC[strat['cadence']]}   |   "
                f"종목선정: {SELECTION_DESC[sel]}   |   "
                f"실행: {EXEC_DESC[sel]}   |   비교 벤치마크: {bname}")
    ws["A3"] = f"위험회피: {variant_desc(strat['variant'], sel)}"
    ws["A2"].alignment = Alignment(wrap_text=True)
    ws["A3"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("A2:N2")
    ws.merge_cells("A3:N3")

    cw_oos = cash_weight(pr, "2019-01-01", "2026-05-31")
    cw_full = cash_weight(pr)
    ws["A5"] = "현금비중(OOS, 2019-01~2026-05):"
    ws["A5"].font = SUB_FONT
    ws["C5"] = _num((cw_oos or 0) * 100, 1)
    ws["D5"] = "%"
    ws["E5"] = "현금비중(전체기간):"
    ws["E5"].font = SUB_FONT
    ws["G5"] = _num((cw_full or 0) * 100, 1)
    ws["H5"] = "%"

    headers = ["구간", "세그먼트", "기간", "CAGR", "MDD", "Sharpe", "Calmar",
               "매매횟수", "누적거래비용%", "연환산비용%", "현금비중%",
               f"벤치({bname}) CAGR", f"벤치({bname}) MDD", f"벤치({bname}) Sharpe", f"벤치({bname}) Calmar",
               "CAGR 초과분(전략-벤치)"]
    header_row = 7
    for j, h in enumerate(headers, start=1):
        ws.cell(row=header_row, column=j, value=h)
    _style_header_row(ws, header_row, len(headers))

    r = header_row + 1
    cad = strat["cadence"]
    for fold, seg in FOLD_SEG_ORDER:
        row = strat_row(mdf, code, cad, fold, seg)
        brow = bench_row(mdf, cad, fold, seg, bcode)
        if row is None:
            continue
        cw_seg = cash_weight(pr, row["start"], row["end"])
        excess = None
        if row is not None and brow is not None and pd.notna(row["cagr"]) and pd.notna(brow["cagr"]):
            excess = _pct(row["cagr"] - brow["cagr"])
        period_txt = f"{row['start']} ~ {row['end']}"
        vals = [
            fold, SEG_LABEL.get(seg, seg), period_txt,
            _pct(row["cagr"]), _pct(row["mdd"]), _num(row["sharpe"]), _num(row["calmar"]),
            int(row["trades"]), _num(row["cost_total_pct"]), _num(row["cost_ann_pct"]),
            _num((cw_seg or 0) * 100, 1) if cw_seg is not None else None,
            _pct(brow["cagr"]) if brow is not None else None,
            _pct(brow["mdd"]) if brow is not None else None,
            _num(brow["sharpe"]) if brow is not None else None,
            _num(brow["calmar"]) if brow is not None else None,
            excess,
        ]
        for j, v in enumerate(vals, start=1):
            ws.cell(row=r, column=j, value=v)
        is_bench_compare_row = fold in ("ALL_OOS", "FULL")
        _style_data_row(ws, r, len(headers), bench=is_bench_compare_row)
        # 손익 하이라이트: CAGR 음수 = 빨강, 초과수익 음수 = 주황
        cagr_cell = ws.cell(row=r, column=4)
        if row["cagr"] is not None and pd.notna(row["cagr"]) and row["cagr"] < 0:
            cagr_cell.font = Font(color="C00000")
        excess_cell = ws.cell(row=r, column=16)
        if excess is not None:
            excess_cell.font = Font(color="C00000" if excess < 0 else "1F7A1F", bold=True)
        r += 1

    ws.freeze_panes = "A8"
    _autofit(ws, [9, 11, 22, 8, 8, 8, 8, 8, 12, 12, 10, 9, 9, 9, 9, 14])


def main():
    mdf, period_records = load_data()
    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    build_summary_sheet(wb, mdf, period_records)
    build_conditions_sheet(wb)
    for strat in C.STRATEGIES:
        build_strategy_sheet(wb, mdf, strat, period_records[strat["code"]])

    wb.save(OUT_PATH)
    print(f"[done] {OUT_PATH}")


if __name__ == "__main__":
    main()
