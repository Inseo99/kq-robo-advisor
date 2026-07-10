# -*- coding: utf-8 -*-
r"""J2 — 국면 생애(lifecycle) 특징의 예측 기여 검증: 무기억 Markov vs 지속 의존.

연구 질문 (사전 등록, 시도 1):
  거시경제 국면의 전이는 무기억(Markov) 과정인가, 국면의 생애(지속·이력)에
  의존하는 과정인가? — cycle_position(look-ahead 위험)의 인과적 재해석.

특징 (전부 확정 라벨만 사용 — labels.shift(label_delay_m=2) 기준):
  regime_duration            현재 확정 국면의 지속 개월
  previous_regime_duration   직전 완료 국면의 지속 개월
  months_since_last_risk     마지막 위험 국면(스태그/디플레) 이후 경과 개월 (위험 중이면 0)
  risk_frequency_24m         최근 24개월 확정 라벨 중 위험 국면 비율
  regime_change_count_24m    최근 24개월 확정 라벨의 전환 횟수
  duration_percentile        현재 지속이 '그 시점까지 완료된 같은 국면 지속 분포'에서
                             갖는 백분위 — 미래에 끝날 국면은 절대 불포함 (요구사항 반영),
                             완료 표본 없으면 0.5(중립)

인과성 자동 검증: 임의 절단 시점들에서 "절단된 라벨 이력만으로 재계산한 특징"과
전체 계산 특징이 일치해야 함 (불일치 시 중단).

실험 설계 (기존 하네스 재사용 — 동일 분류기·시드·purge 규칙):
  기준: walk_forward_predict(시장 특징 8종)  /  처치: + 생애 특징 6종
  과제: nowcast(2개월 지연 정산) · forecast(3개월 선행)
  공통 평가 월(교집합)에서 로그로스 비교

채점 (사전 고정):
  합격 = 두 과제 모두 로그로스 비악화(+0.005 허용) AND 최소 한 과제 개선(−0.005 초과)
  기각 = 어느 과제든 로그로스가 +0.005 초과 악화
  참고 보고: P³ 베이스라인 대비 각 구성의 selected 결과

주의: 세션 환경은 LightGBM 부재 시 sklearn HistGB 폴백 — 로컬(TabPFN/LGBM)과
절대값이 다를 수 있으므로 동일 환경 내 상대 비교만 유효 (보고서에 명시).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from kq_tool.regime import regime_model_v2 as m  # noqa: E402
from kq_tool.regime.macro_data import get_observable_panel  # noqa: E402
from kq_tool.regime.regime_labels import REGIMES  # noqa: E402

RISK = {"스태그플레이션", "디플레이션"}
LIFECYCLE = ["regime_duration", "previous_regime_duration", "months_since_last_risk",
             "risk_frequency_24m", "regime_change_count_24m", "duration_percentile"]


def lifecycle_features(labels: pd.Series, delay: int = 2) -> pd.DataFrame:
    """확정 라벨(shift(delay))만으로 생애 특징 계산 — 각 t는 t-delay까지의 이력만 사용."""
    confirmed = labels.shift(delay)
    idx = labels.index
    out = pd.DataFrame(index=idx, columns=LIFECYCLE, dtype=float)
    hist: list[str] = []
    completed: dict[str, list[int]] = {r: [] for r in REGIMES}
    cur_regime, cur_len, prev_len = None, 0, np.nan
    since_risk = np.nan
    for t in idx:
        v = confirmed.loc[t]
        if pd.isna(v):
            hist.append(None)
            continue
        v = str(v)
        if v == cur_regime:
            cur_len += 1
        else:
            if cur_regime is not None:
                completed[cur_regime].append(cur_len)
                prev_len = cur_len
            cur_regime, cur_len = v, 1
        if v in RISK:
            since_risk = 0
        else:
            since_risk = (since_risk + 1) if not np.isnan(since_risk) else np.nan
        hist.append(v)
        recent = [h for h in hist[-24:] if h is not None]
        if recent:
            out.loc[t, "risk_frequency_24m"] = sum(1 for h in recent if h in RISK) / len(recent)
            out.loc[t, "regime_change_count_24m"] = sum(
                1 for a, b in zip(recent[:-1], recent[1:]) if a != b)
        out.loc[t, "regime_duration"] = cur_len
        out.loc[t, "previous_regime_duration"] = prev_len
        out.loc[t, "months_since_last_risk"] = since_risk
        done = completed[cur_regime]
        out.loc[t, "duration_percentile"] = (
            0.5 if not done else float(np.mean([d <= cur_len for d in done])))
    return out


def causality_check(labels: pd.Series, feats: pd.DataFrame, n_probe: int = 6) -> None:
    """임의 절단 시점에서 특징 재계산 — 절단 이후 정보가 특징에 없음을 검증."""
    rng = np.random.default_rng(0)
    valid = feats.dropna().index
    for t in rng.choice(valid, size=min(n_probe, len(valid)), replace=False):
        truncated = labels.loc[:t]
        re = lifecycle_features(truncated).loc[t]
        assert np.allclose(re.astype(float), feats.loc[t].astype(float), equal_nan=True), \
            f"인과성 위반: {t}"
    print(f"[인과성 검증] 절단 재계산 {n_probe}개 시점 일치 — 통과")


def run_variant(features: pd.DataFrame, labels: pd.Series, task: str, extra: bool):
    feats = m.FEATURES + (LIFECYCLE if extra else [])
    orig = m.FEATURES
    m.FEATURES = feats            # 동일 코드 경로 사용을 위한 런타임 확장
    try:
        cfg = m.ModelConfig(task=task)
        probs, _ = m.walk_forward_predict(features, labels, cfg)
    finally:
        m.FEATURES = orig
    return probs


def main():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rm", os.path.join(ROOT, "scripts", "run_regime_model.py"))
    rm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rm)
    labels = rm.load_labels(Path(ROOT) / "data" / "regime" / "labels.csv")  # 검증된 로더 재사용
    panel = get_observable_panel(["yield_spread_10y_3y", "usdkrw", "credit_spread", "kospi", "base_rate"])
    base_feats = rm._month_index(m.build_features(panel))
    life = lifecycle_features(labels)
    causality_check(labels, life)
    all_feats = base_feats.join(life, how="left")
    # run_regime_model과 동일한 정렬: 라벨-특징 공통 구간으로 절단
    common_idx = labels.index.intersection(all_feats.index)
    labels = labels.loc[common_idx]
    all_feats = all_feats.loc[common_idx]

    print("\n=== J2 ablation (동일 분류기·시드·purged walk-forward) ===")
    verdict_parts = []
    for task in ["nowcast", "forecast"]:
        cfg = m.ModelConfig(task=task)
        p0 = run_variant(all_feats, labels, task, extra=False)
        p1 = run_variant(all_feats, labels, task, extra=True)
        common = p0.index.intersection(p1.index)
        actual = labels if task == "nowcast" else labels.shift(-cfg.horizon_m)
        ll0 = m.log_loss_score(p0.loc[common], actual)
        ll1 = m.log_loss_score(p1.loc[common], actual)
        # P³ 참고 베이스라인
        if task == "forecast":
            pb = m.p3_forecast_baseline(labels, cfg)
        else:
            pb = m.naive_nowcast_baseline(labels, cfg)
        llb = m.log_loss_score(pb.loc[pb.index.intersection(common)], actual)
        d = ll1 - ll0
        print(f"[{task}] 기준(시장 특징) {ll0:.4f} · +생애 특징 {ll1:.4f} · 차이 {d:+.4f} "
              f"· 참고 베이스라인 {llb:.4f} · 평가 {len(common)}개월")
        verdict_parts.append(d)

    worse = any(d > 0.005 for d in verdict_parts)
    better = any(d < -0.005 for d in verdict_parts)
    verdict = "기각 (악화)" if worse else ("합격 (지속 의존성 증거)" if better else "무효과 (보류)")
    print(f"\n[사전 고정 판정] {verdict}")
    print("[환경 주의] 본 실행 분류기 = " + type(m._make_classifier(0)).__name__ +
          " — 로컬(TabPFN/LGBM)과 절대값 상이 가능, 동일 환경 내 상대 비교만 유효")


if __name__ == "__main__":
    main()
