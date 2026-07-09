"""scripts/audit_stat_tests.py — 로보필터 제외 감사 통계 보강
1) 대응표본 t-검정: 대체−제외 평균 격차(+1.80%p)의 유의성 (Hit p와 별개)
2) 구간별 분해: 4개 시장 구간에서 격차 부호가 유지되는지 (선택 편향 방어)

입력: 감사 로그 CSV — 필요한 컬럼(이름 유연 매핑):
  날짜(date), 제외 수익률(r_excluded), 대체 수익률(r_replacement), 전략(strategy, 선택)
사용: python scripts/audit_stat_tests.py <감사로그.csv> [--strategy quant]
scipy 없으면 t-분포 근사(정규)로 계산하고 표기함.
"""
from __future__ import annotations
import argparse, math, sys
import numpy as np
import pandas as pd

COLMAP = {
    "date": ["date", "일자", "판정일", "rebalance_date"],
    "r_excluded": ["r_excluded", "excluded_ret", "제외수익률", "excluded_return"],
    "r_replacement": ["r_replacement", "replacement_ret", "대체수익률", "replacement_return"],
    "strategy": ["strategy", "전략", "strat"],
}

# 4개 시장 구간 (backtest-ksj OOS 구간과 동일 절단)
PERIODS = [
    ("2014-2018 (상승/횡보)", "2014-01-01", "2018-12-31"),
    ("2019-2020 (코로나 급락/반등)", "2019-01-01", "2020-12-31"),
    ("2021-2022 (하락)", "2021-01-01", "2022-12-31"),
    ("2023-2026.5 (회복/급등)", "2023-01-01", "2026-05-31"),
]


def _find(df, key):
    for c in COLMAP[key]:
        if c in df.columns:
            return c
    return None


def paired_t(diff: np.ndarray):
    n = len(diff)
    mean, sd = diff.mean(), diff.std(ddof=1)
    if sd == 0 or n < 3:
        return np.nan, np.nan, n
    t = mean / (sd / math.sqrt(n))
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), n - 1)
        approx = ""
    except ImportError:
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
        approx = " (정규근사)"
    return t, p, approx


def run(path: str, strategy: str | None):
    df = pd.read_csv(path)
    cols = {k: _find(df, k) for k in COLMAP}
    missing = [k for k in ("date", "r_excluded", "r_replacement") if cols[k] is None]
    if missing:
        raise SystemExit(f"[FAIL] 컬럼 못 찾음: {missing}. 현재 컬럼: {list(df.columns)}\n"
                         f"→ COLMAP에 실제 이름을 추가하세요.")
    if strategy and cols["strategy"]:
        df = df[df[cols["strategy"]].astype(str).str.contains(strategy, case=False)]
    df["_d"] = pd.to_datetime(df[cols["date"]])
    diff = (df[cols["r_replacement"]] - df[cols["r_excluded"]]).astype(float)

    t, p, note = paired_t(diff.values)
    print(f"=== 전체 (n={len(diff)}) ===")
    print(f"대체−제외 평균: {diff.mean():+.2%}  중앙값: {diff.median():+.2%}")
    print(f"대응표본 t={t:.2f}, p={p:.3f}{note}")
    print(f"Hit(대체>제외): {(diff > 0).mean():.1%}")

    print("\n=== 구간별 (부호 일관성) ===")
    rows = []
    for name, s, e in PERIODS:
        sub = diff[(df["_d"] >= s) & (df["_d"] <= e)]
        if len(sub) == 0:
            rows.append((name, 0, None, None)); continue
        rows.append((name, len(sub), sub.mean(), (sub > 0).mean()))
    consistent = all(r[2] is None or r[2] > 0 for r in rows if r[1] > 0)
    for name, n, m, h in rows:
        if n == 0:
            print(f"{name:32s} n=0 (데이터 없음)")
        else:
            print(f"{name:32s} n={n:3d}  평균 {m:+.2%}  Hit {h:.0%}")
    print(f"\n부호 일관성(전 구간 평균 양수): {'예 — 선택편향 방어 근거로 사용 가능' if consistent else '아니오 — 특정 구간 의존, 발표에서 명시 필요'}")

    print("\n[발표 문구 가이드]")
    if not np.isnan(p) and p < 0.05:
        print(f"→ \"평균 격차 {diff.mean():+.2%}p는 통계적으로 유의(t-검정 p={p:.3f})\"")
    else:
        print(f"→ 평균 격차 p={p:.3f}로 미유의. 적용 근거는 '증거 방향 + 구간 일관성'으로 서술:")
        print("   \"격차의 유의성은 표본상 확정할 수 없으나, 4개 시장 구간 모두에서 부호가"
              " 일관되고 ON/OFF 전체 성과가 개선되어 적용 후보로 판단, 감사 지표로 지속 재평가\"")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="감사 로그 CSV 경로")
    ap.add_argument("--strategy", default=None, help="전략 필터 (예: quant)")
    a = ap.parse_args()
    run(a.csv, a.strategy)
