"""LSV/Kang 12전략(Fama-LSV·Kang × s1/s2/s3 × 월간/주간) 결과를 별도 CSV로 정리.

metrics_all.csv에서 종목선정 krsec_lsv/krsec_kang 전략만 추출 + 비교용 KODEX200 벤치마크.
기존 형식과 동일한 컬럼(code,cadence,fold,segment,start,end,+지표). 실행: python export_lsv_kang.py
출력: results/lsv_kang_metrics.csv (전체 폴드×IS/OOS)
      results/lsv_kang_summary_oos.csv (전체 OOS 요약)
"""
from __future__ import annotations

import os
import pandas as pd

import config as C

RESULTS = C.RESULTS


def main():
    m = pd.read_csv(os.path.join(RESULTS, "metrics_all.csv"))
    codes = [s["code"] for s in C.STRATEGIES if s["selection"] in ("krsec_lsv", "krsec_kang")]
    keep = codes + [C.BENCHMARK_CODE]
    sub = m[m["code"].isin(keep)].copy()
    order = {c: i for i, c in enumerate(keep)}
    seg_order = {"IS": 0, "OOS": 1, "ALL": 2}
    fold_order = {f["name"]: i for i, f in enumerate(C.FOLDS)}
    fold_order.update({"ALL_OOS": 90, "FULL": 91})
    sub["_c"] = sub["code"].map(order)
    sub["_f"] = sub["fold"].map(fold_order)
    sub["_s"] = sub["segment"].map(seg_order)
    sub = sub.sort_values(["_c", "cadence", "_f", "_s"]).drop(columns=["_c", "_f", "_s"])

    out_all = os.path.join(RESULTS, "lsv_kang_metrics.csv")
    sub.to_csv(out_all, index=False, encoding="utf-8-sig")

    summ = sub[sub["fold"] == "ALL_OOS"].copy()
    out_sum = os.path.join(RESULTS, "lsv_kang_summary_oos.csv")
    summ.to_csv(out_sum, index=False, encoding="utf-8-sig")
    print(f"[done] {out_all}  ({len(sub)} rows)")
    print(f"[done] {out_sum}  ({len(summ)} rows)")


if __name__ == "__main__":
    main()
