"""백테스트 실행 오케스트레이터 (backtest-ksj).

사용법:
    python run.py

동작:
  1) 데이터 패널 1회 로드 (가격/시총/KODEX200/VIX·credit)
  2) 전략별 백테스트 (s2/s2w는 VIX·미국신용 데이터 있을 때만 실행)
  3) 폴드별 IS/OOS + 전체 OOS(2019~2026.5) + 전체기간 지표 산출
  4) results/ 에 CSV + report.md 저장
"""

from __future__ import annotations

import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

import config as C
import data as D
import engine as E
import metrics as M


# 전체 OOS = 4개 OOS 창을 이은 연속 구간 (2019-01 ~ 2026-05)
FULL_OOS = ("2019-01-01", "2026-05-31")
FULL_ALL = ("2015-01-01", "2026-06-30")


def load_panels() -> dict:
    print("[load] 가격 패널 로딩...")
    adj_close = D.load_adj_close()
    adj_open = D.load_adj_open()
    panels = {
        "adj_close": adj_close,
        "adj_open": adj_open,
        "close_ff": adj_close.ffill(),
        "open_ff": adj_open.ffill(),
        "mcap_hist": D.load_mcap_history(),
        "kodex_daily": D.load_kodex200_daily(),
        "sp500": D.load_sp500(),
        "vix": D.load_vix(),
        "credit": D.load_credit_us(),
    }
    print(f"[load] 완료. close {adj_close.shape}, "
          f"S&P500={'O' if panels['sp500'] is not None else 'X'}, "
          f"VIX={'O' if panels['vix'] is not None else 'X'}, "
          f"credit={'O' if panels['credit'] is not None else 'X'}")
    return panels


def collect_metrics(code: str, cadence: str, df: pd.DataFrame) -> list[dict]:
    """전략 df에서 폴드별 IS/OOS + 전체OOS + 전체기간 지표 레코드."""
    recs = []
    for fold in C.FOLDS:
        for seg in ("is", "oos"):
            start, end = fold[seg]
            m = M.strategy_window_metrics(df, cadence, start, end)
            recs.append({"code": code, "cadence": cadence, "fold": fold["name"],
                         "segment": seg.upper(), "start": start, "end": end, **m})
    recs.append({"code": code, "cadence": cadence, "fold": "ALL_OOS", "segment": "OOS",
                 "start": FULL_OOS[0], "end": FULL_OOS[1],
                 **M.strategy_window_metrics(df, cadence, *FULL_OOS)})
    recs.append({"code": code, "cadence": cadence, "fold": "FULL", "segment": "ALL",
                 "start": FULL_ALL[0], "end": FULL_ALL[1],
                 **M.strategy_window_metrics(df, cadence, *FULL_ALL)})
    return recs


def collect_benchmark(bench_daily: pd.Series) -> list[dict]:
    recs = []
    for cadence in ("M", "W"):
        for fold in C.FOLDS:
            for seg in ("is", "oos"):
                start, end = fold[seg]
                m = M.benchmark_window_metrics(bench_daily, start, end)
                recs.append({"code": C.BENCHMARK_CODE, "cadence": cadence, "fold": fold["name"],
                             "segment": seg.upper(), "start": start, "end": end, **m})
        recs.append({"code": C.BENCHMARK_CODE, "cadence": cadence, "fold": "ALL_OOS",
                     "segment": "OOS", "start": FULL_OOS[0], "end": FULL_OOS[1],
                     **M.benchmark_window_metrics(bench_daily, *FULL_OOS)})
        recs.append({"code": C.BENCHMARK_CODE, "cadence": cadence, "fold": "FULL",
                     "segment": "ALL", "start": FULL_ALL[0], "end": FULL_ALL[1],
                     **M.benchmark_window_metrics(bench_daily, *FULL_ALL)})
    return recs


def main():
    os.makedirs(C.RESULTS, exist_ok=True)
    panels = load_panels()
    t1_ready = panels["vix"] is not None and panels["credit"] is not None

    all_metrics = []
    ran, skipped = [], []
    for strat in C.STRATEGIES:
        code, variant, cadence = strat["code"], strat["variant"], strat["cadence"]
        if variant == "s2" and not t1_ready:
            skipped.append(code)
            continue
        print(f"[run] {code} ({strat['label']}) ...")
        df = E.run_strategy(strat, panels)
        df.to_csv(os.path.join(C.RESULTS, f"period_records_{code}.csv"),
                  index=False, encoding="utf-8-sig")
        all_metrics.extend(collect_metrics(code, cadence, df))
        ran.append(code)

    # 벤치마크
    bench_daily = D.load_kodex200_actual()
    all_metrics.extend(collect_benchmark(bench_daily))

    mdf = pd.DataFrame(all_metrics)
    mdf.to_csv(os.path.join(C.RESULTS, "metrics_all.csv"), index=False, encoding="utf-8-sig")

    # 요약(전체 OOS)
    summary = mdf[mdf["fold"] == "ALL_OOS"].copy()
    summary.to_csv(os.path.join(C.RESULTS, "summary_full_oos.csv"),
                   index=False, encoding="utf-8-sig")

    write_report(mdf, ran, skipped, t1_ready)
    print(f"[done] 실행 전략: {ran} | 대기(s2): {skipped}")
    print(f"[done] 결과: {C.RESULTS}")


# ── 리포트 ───────────────────────────────────────────────────────────────
def _fmt_pct(x):
    return "—" if pd.isna(x) else f"{x*100:.1f}%"


def _fmt_num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def _label(code):
    for s in C.STRATEGIES:
        if s["code"] == code:
            return s["label"]
    return C.BENCHMARK_LABEL if code == C.BENCHMARK_CODE else code


def _metric_table(mdf, fold_name, codes, cadence=None):
    """지정 fold의 전략별 지표표(마크다운)."""
    lines = ["| 전략 | 기간 | CAGR | MDD | Sharpe | Calmar | 매매횟수 | 누적거래비용 | 연환산비용 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for code in codes:
        q = mdf[(mdf["code"] == code) & (mdf["fold"] == fold_name)]
        if cadence is not None:
            q = q[q["cadence"] == cadence]
        if q.empty:
            continue
        r = q.iloc[0]
        yrs = f"{r['years']:.1f}y" if r["years"] and not pd.isna(r["years"]) else "—"
        lines.append(
            f"| {_label(code)} | {yrs} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['mdd'])} | "
            f"{_fmt_num(r['sharpe'])} | {_fmt_num(r['calmar'])} | {int(r['trades'])} | "
            f"{_fmt_pct(r['cost_total_pct']/100)} | {_fmt_pct(r['cost_ann_pct']/100)} |")
    return "\n".join(lines)


def write_report(mdf, ran, skipped, t1_ready):
    monthly_codes = [s["code"] for s in C.STRATEGIES if s["cadence"] == "M" and s["code"] in ran]
    weekly_codes = [s["code"] for s in C.STRATEGIES if s["cadence"] == "W" and s["code"] in ran]
    B = C.BENCHMARK_CODE

    out = []
    out.append("# 백테스트 결과 보고서 — backtest-ksj\n")
    out.append(f"> 생성 시각 기준 데이터: 수정주가 2014-01 ~ 2026-06-05, 분기 시총 PIT, KODEX200(069500)\n")

    out.append("\n> ## ⚠️ 데이터 주의 (반드시 확인)\n"
               "> 이 저장소의 가격 데이터는 **2025~2026 구간이 합성(synthetic)으로 크게 부풀려져** 있습니다.\n"
               "> 예: KODEX200 31,990(2025-01)→131,030(2026-06, 약 4배), 삼성전자 53,400→329,000(약 6배).\n"
               "> 2024년까지는 실제와 유사하나 **2025-2026 급등은 가상**입니다. 따라서 **Fold4(2025~) 및 전체 OOS의 절대 CAGR은 비현실적으로 높습니다.**\n"
               "> 엔진·회계 처리는 정상이며, 의미 있는 해석은 **전략 간 상대 비교**(오버레이의 MDD 축소, 회전율/거래비용 차이, 벤치마크 대비 초과·미달)에 두어야 합니다.\n"
               "> 매매횟수 = 리밸런싱마다 **편입(신규매수)+편출(전량매도) 종목 수의 합**. 거래비용(누적)=기간 내 비용 차감분 합.\n")

    out.append("## 1. 개요\n")
    out.append("- **유니버스**: Point-in-Time 시가총액 상위 **300** (KOSPI+KOSDAQ, 분기 시총 ffill; 현재 생존자 전용 리스트는 아님. 단, 원천 캐시 커버리지 한계 존재)\n"
               "- **종목선정**: 12-1 모멘텀(최근 1개월 제외 과거 11개월) 상위 **20종목**, 동일가중\n"
               "- **실행**: 월간=월말평가→익월 첫 거래일 시초가 / 주간=금요일 종가평가→익영업일 시초가 (수정주가)\n"
               "- **거래비용**: 왕복 0.5% (편도 0.25% × 회전율 Σ|Δw|)\n"
               "- **현금수익**: 연 2% (월 0.167% / 주 0.038%, 위험회피 시 100% 현금)\n"
               "- **벤치마크**: KODEX 200 (069500), 가격수익률 기준(분배금·세금 미반영)\n"
               "- **주의**: 모멘텀 12개월 워밍업으로 실제 매매는 **2015-02**부터. Fold1 IS(2014~) 및 2015-10 이전 벤치마크 구간은 데이터 제약으로 축소됨.\n")

    out.append("\n### 폴드 (Walk-forward)\n")
    out.append("| 폴드 | In-Sample | Out-of-Sample |\n|---|---|---|")
    for f in C.FOLDS:
        out.append(f"| {f['name']} | {f['is'][0]} ~ {f['is'][1]} | {f['oos'][0]} ~ {f['oos'][1]} |")
    out.append("\n> 4개 OOS 구간은 연속 → **전체 OOS = 2019-01 ~ 2026-05**. "
               "전략 규칙은 사전 고정이므로 발표에서는 'OOS 통과'가 아니라 "
               "'구간별 강건성 평가'로 해석한다.\n")

    out.append("\n### 전략 정의\n")
    out.append("| 코드 | 리밸런싱 | 종목선정 | 위험회피 오버레이 |\n|---|---|---|---|")
    out.append("| s1m | 월간 | 12-1 모멘텀 상위20 | 없음 |")
    out.append("| s2m | 월간 | s1 | t1: (①S&P500<9M MA, ②VIX>18.6, ③US신용스프레드 z(5M)>1.78) 중 2개↑ → 주식 0% |")
    out.append("| s3m | 월간 | s1 | t2: KODEX 월말종가 >10M MA → 100% 보유 / 아니면 전액 현금 |")
    out.append("| s1w | 주간 | s1 (금요일) | 없음 |")
    out.append("| s2w | 주간 | s1 (금요일) | t1(①S&P500<39주 MA, ②VIX>18.6, ③US신용 z(일별 105거래일)>1.78) 중 2개↑ → 주식 0% |")
    out.append("| s3w | 주간 | s1 (금요일) | t2(주환산: 43주 MA) |")
    out.append("| s1m-kd200 | 월간 | **KODEX200 단일보유** | 없음 |")
    out.append("| s2m-kd200 | 월간 | KODEX200 | t1 (s2m과 동일) |")
    out.append("| s3m-kd200 | 월간 | KODEX200 | t2 (s3m과 동일) |")
    out.append("| s1w-kd200 | 주간 | **KODEX200 단일보유** | 없음 |")
    out.append("| s2w-kd200 | 주간 | KODEX200 | t1 (s2w과 동일) |")
    out.append("| s3w-kd200 | 주간 | KODEX200 | t2 (s3w과 동일) |")
    out.append("\n> **종목선정 변형**: `*-kd200`은 모멘텀20 포트폴리오 대신 **KODEX200(069500) 1종목**을 보유합니다. "
               "리밸런싱 주기·거래비용(왕복 0.5%)·현금(연2%)·위험회피(t1/t2)는 기존과 동일해 직접 비교 가능합니다. "
               "KODEX200은 시초가 시계열이 없어 보유수익을 **종가→종가**(실행일 다음 거래일 종가 기준, 실행지연 동일)로 계산합니다. "
               "s1*-kd200은 사실상 KODEX200 매수후보유(리밸런싱 회전율 0)이며 벤치마크와 거의 일치해야 합니다.\n")

    if skipped:
        out.append(f"\n> ⚠️ **{', '.join(skipped)} 미실행**: t1 신호용 VIX·미국 신용스프레드 CSV가 아직 없습니다. "
                   f"`backtest-ksj/data/`에 `vix.csv`, `credit_spread_us.csv`(각각 `date,value`)를 넣고 `python run.py` 재실행하면 자동 포함됩니다.\n")

    out.append("\n## 2. 핵심 결과 — 전체 OOS (2019-01 ~ 2026-05)\n")
    out.append("### 월간 전략\n")
    out.append(_metric_table(mdf, "ALL_OOS", monthly_codes + [B], cadence="M"))
    out.append("\n\n### 주간 전략\n")
    out.append(_metric_table(mdf, "ALL_OOS", weekly_codes + [B], cadence="W"))

    out.append("\n\n## 3. 폴드별 OOS 성과\n")
    for f in C.FOLDS:
        out.append(f"\n### {f['name']} OOS ({f['oos'][0]} ~ {f['oos'][1]})\n")
        sub = mdf[(mdf["fold"] == f["name"]) & (mdf["segment"] == "OOS")]
        out.append("**월간**\n")
        out.append(_fold_seg_table(sub, monthly_codes + [B], "M"))
        out.append("\n\n**주간**\n")
        out.append(_fold_seg_table(sub, weekly_codes + [B], "W"))

    out.append("\n\n## 4. 폴드별 IS 성과 (참고)\n")
    for f in C.FOLDS:
        out.append(f"\n### {f['name']} IS ({f['is'][0]} ~ {f['is'][1]})\n")
        sub = mdf[(mdf["fold"] == f["name"]) & (mdf["segment"] == "IS")]
        out.append("**월간**\n")
        out.append(_fold_seg_table(sub, monthly_codes + [B], "M"))
        out.append("\n\n**주간**\n")
        out.append(_fold_seg_table(sub, weekly_codes + [B], "W"))

    out.append("\n\n## 5. 산출 파일\n")
    out.append("- `results/metrics_all.csv` — 전략×폴드×IS/OOS 전체 지표\n"
               "- `results/summary_full_oos.csv` — 전체 OOS 요약\n"
               "- `results/period_records_*.csv` — 전략별 기간 레코드(감사추적: 진입/청산/보유수/회전율/비용/수익)\n")

    out.append("\n\n## 6. 방법론 주의\n")
    out.append("- 유니버스는 분기별 시총 이력에서 판정일 기준 상위 300을 뽑는다. "
               "2014년/2018년 표본에 2026년 현재 비활성 종목이 포함되는 것을 확인했으므로 "
               "현재 생존자 전용 리스트는 아니다. 다만 상장폐지 전체 커버리지는 원천 `data/cache` 범위에 의존한다.\n"
               "- KODEX 200 벤치마크는 가격수익률 기준이다. 분배금 재투자 총수익률을 쓰지 않았으므로 "
               "전략의 벤치마크 대비 초과수익은 일부 높게 보일 수 있다.\n")

    path = os.path.join(C.RESULTS, "report.md")
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(x for x in out if x is not None))
    print(f"[report] {path}")


def _fold_seg_table(sub, codes, cadence):
    lines = ["| 전략 | CAGR | MDD | Sharpe | Calmar | 매매횟수 | 거래비용(누적) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for code in codes:
        q = sub[(sub["code"] == code) & (sub["cadence"] == cadence)]
        if q.empty:
            continue
        r = q.iloc[0]
        lines.append(
            f"| {_label(code)} | {_fmt_pct(r['cagr'])} | {_fmt_pct(r['mdd'])} | "
            f"{_fmt_num(r['sharpe'])} | {_fmt_num(r['calmar'])} | {int(r['trades'])} | "
            f"{_fmt_pct(r['cost_total_pct']/100)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
