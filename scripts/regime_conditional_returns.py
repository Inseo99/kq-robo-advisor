"""scripts/regime_conditional_returns.py — 국면 라벨링 + 국면별 전략 성과
출력 1: data/analysis_outputs/regime_timeline.csv  (월 × PiT 국면 라벨)
출력 2: data/analysis_outputs/regime_conditional_returns.csv (전략 × 국면 성과표)

사용:
  python scripts/regime_conditional_returns.py                       # 국내 12전략 + 벤치마크 전부
  python scripts/regime_conditional_returns.py --codes s1m s3m 069500  # 일부만

방법론 각주 (결과 CSV에도 기록):
  - 라벨은 PiT(vintage) 기준 — 각 월의 수익률에는 '그 월 시작 이전 마지막 라벨'을
    매칭해 힌드사이트를 차단한다.
  - 국면별 표본(n)이 작으면 평균은 노이즈다. n < 12 국면은 '참고' 표기.
  - 이것은 기술 통계이지 유의성 검정이 아니다 (로보필터 p=0.339 원칙과 동일 규율).
"""
from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_CSV = os.path.join(ROOT, "data", "macro", "regime_labels.csv")
RECORDS_DIR = os.path.join(ROOT, "backtest-ksj", "results")
OUT_DIR = os.path.join(ROOT, "data", "analysis_outputs")

_DATE_CANDS = ("date", "month", "일자", "날짜", "기준월", "asof")
_LABEL_CANDS = ("regime", "label", "국면", "regime_label", "국면라벨")
# period_records의 날짜/수익률 컬럼 후보 (스키마 유연 탐색)
_REC_DATE = ("end", "date", "period_end", "eval_date", "기준일", "평가일", "rebalance_date", "exit", "eval", "start", "entry")  # exit=기간 종료일 우선 (eval은 시작일임이 확인됨)
_REC_RET = ("ret", "return", "period_return", "port_ret", "strategy_ret", "수익률", "period_ret", "r", "net", "gross")


def load_labels() -> pd.Series:
    df = pd.read_csv(LABELS_CSV)
    dcol = next((c for c in df.columns if c.strip().lower() in _DATE_CANDS), df.columns[0])
    lcol = next((c for c in df.columns if c.strip().lower() in _LABEL_CANDS), None)
    if lcol is None:
        lcol = next(c for c in df.columns if c != dcol and df[c].dtype == object)
    s = df.set_index(pd.to_datetime(df[dcol]))[lcol].astype(str).sort_index()
    return s


def label_asof(labels: pd.Series, when: pd.Timestamp):
    """when 이전 마지막 라벨 (PiT — 힌드사이트 차단)."""
    past = labels[labels.index < when]
    return past.iloc[-1] if len(past) else None


def write_timeline(labels: pd.Series) -> str:
    monthly = labels.resample("ME").last().dropna()
    if cutoff is not None:
        monthly = monthly[monthly.index <= cutoff]
    out = os.path.join(OUT_DIR, "regime_timeline.csv")
    monthly.rename("regime").to_csv(out, encoding="utf-8-sig", index_label="month")
    # 국면 구간 요약도 화면 출력 (연속 동일 라벨 병합)
    runs, cur, start = [], None, None
    for dt, lab in monthly.items():
        if lab != cur:
            if cur is not None:
                runs.append((start, prev, cur))
            cur, start = lab, dt
        prev = dt
    runs.append((start, prev, cur))
    print(f"\n=== 국면 타임라인 ({monthly.index[0]:%Y-%m} ~ {monthly.index[-1]:%Y-%m}) → {out} ===")
    for s, e, lab in runs:
        print(f"  {s:%Y-%m} ~ {e:%Y-%m}  {lab}  ({(e.year-s.year)*12 + e.month-s.month + 1}개월)")
    return out


def load_strategy_returns(code: str) -> pd.Series | None:
    path = os.path.join(RECORDS_DIR, f"period_records_{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    dcol = next((c for c in df.columns if c.strip().lower() in _REC_DATE), None)
    rcol = next((c for c in df.columns if c.strip().lower() in _REC_RET), None)
    if dcol is None or rcol is None:
        print(f"[SKIP] {code}: 날짜/수익률 컬럼 못 찾음 — 실제 컬럼: {list(df.columns)}")
        print("       → _REC_DATE/_REC_RET 후보에 이름을 추가하세요")
        return None
    s = df.set_index(pd.to_datetime(df[dcol]))[rcol].astype(float).sort_index()
    # 주간 전략은 월 복리 합성으로 통일 (라벨이 월 단위이므로)
    return (1 + s).resample("ME").prod() - 1


def conditional_table(labels: pd.Series, codes: list[str]) -> pd.DataFrame:
    rows = []
    for code in codes:
        rets = load_strategy_returns(code)
        if rets is None:
            continue
        if cutoff is not None:
            rets = rets[rets.index <= cutoff]
        if rets.empty:
            continue
        matched = pd.DataFrame({
            "ret": rets,
            "regime": [label_asof(labels, dt.replace(day=1)) for dt in rets.index],
        }).dropna()
        for regime, grp in matched.groupby("regime"):
            n = len(grp)
            rows.append(dict(
                code=code, regime=regime, n_months=n,
                mean_monthly=round(grp["ret"].mean(), 5),
                median_monthly=round(grp["ret"].median(), 5),
                ann_return=round((1 + grp["ret"].mean()) ** 12 - 1, 4),
                std_monthly=round(grp["ret"].std(ddof=1), 5) if n > 1 else None,
                sharpe_ann=round(grp["ret"].mean() / grp["ret"].std(ddof=1) * (12 ** 0.5), 3)
                           if n > 1 and grp["ret"].std(ddof=1) > 0 else None,
                hit_rate=round((grp["ret"] > 0).mean(), 3),
                worst_month=round(grp["ret"].min(), 5),
                note="" if n >= 12 else f"표본 부족(n={n}) — 참고용",
            ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_codes = ["s1m", "s2m", "s3m", "s1w", "s2w", "s3w",
                     "s1m-kd200", "s2m-kd200", "s3m-kd200", "069500"]
    ap.add_argument("--codes", nargs="*", default=default_codes)
    ap.add_argument("--exclude-after", default=None, metavar="YYYY-MM",
                    help="이 월 이후 수익률 제외 (예: 2024-12 — 합성 데이터 구간 배제)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    labels = load_labels()
    cutoff = None
    if args.exclude_after:
        cutoff = pd.Period(args.exclude_after, freq="M").to_timestamp("M")
        print(f"[제외] {args.exclude_after} 이후 수익률 배제 — 합성 데이터 구간 차단")
    write_timeline(labels)

    table = conditional_table(labels, args.codes)
    if table.empty:
        raise SystemExit("\n[FAIL] 매칭된 전략 없음 — period_records 스키마 확인 필요")
    out = os.path.join(OUT_DIR, "regime_conditional_returns.csv")
    table.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n=== 국면별 성과 (월수익률 기준) → {out} ===")
    pivot = table.pivot_table(index="code", columns="regime", values="ann_return")
    print((pivot * 100).round(1).to_string())
    print("\n[방법론] PiT 라벨 매칭(힌드사이트 차단) · 주간 전략은 월 복리 합성 ·")
    print("        탐색적 기술 통계 — 통계적 유의성 주장이 아니라 거시환경별 성과 특성의 존재 여부 탐색 · worst_month = 해당 국면 월수익률 최솟값(비연속 월이라 MDD 대신 사용) · Sharpe는 rf=0 월간 기준 연율화 · n<12 국면은 note 컬럼에 표본 부족 표기 ·")
    print("        2025~26 합성 데이터 구간 포함 시 절대치 주의 (--exclude-after 2024-12 로 배제 가능) (필요 시 해당 월 제외 후 재실행)")
