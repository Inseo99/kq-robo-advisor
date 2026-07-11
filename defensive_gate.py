# -*- coding: utf-8 -*-
"""조건부 위험회피 게이트 — 채택 국면 라벨 + 2개월 관측 지연 기준.

사용법 (나중에 재실행할 때):
  1) 이 파일을 kq_tool 폴더(labels.csv가 있는 저장소 루트)에 저장
  2) 백테스터에서:  from defensive_gate import defensive_gate
     매월 리밸런스 시  defensive_gate("2018-03")  이 True면 t2 규칙 적용
  3) 동작 확인만 해보려면 터미널에서:  python defensive_gate.py

원칙 (메인 국면 파트와 동일한 관측 가능성 기준):
  - 채택 라벨 파일(data/regime/labels.csv)을 읽기만 한다 (재생성 금지)
  - t월 판정에는 t-2월 라벨만 사용 (GDP 발표지연 2개월 = label_delay_m=2)
  - 라벨이 없는 월(2011-02 이전, 2026-02 이후 판정)은 게이트 미적용(False)
"""
import os

import pandas as pd

RISK = {"스태그플레이션", "디플레이션"}

_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in [_HERE, os.path.dirname(_HERE)]:
    _path = os.path.join(_cand, "data", "regime", "labels.csv")
    if os.path.exists(_path):
        break
else:
    raise FileNotFoundError("data/regime/labels.csv 를 찾지 못했습니다 — "
                            "이 파일을 kq_tool 저장소 루트에 두세요.")

_lab = pd.read_csv(_path, encoding="utf-8-sig",
                   parse_dates=["date"]).set_index("date")["regime"]
_lab.index = _lab.index.to_period("M")


def defensive_gate(month) -> bool:
    """t월 리밸런스에 사용 가능한 판정: t-2월 채택 라벨이 위험 국면인가."""
    ref = pd.Period(month, freq="M") - 2
    return str(_lab.get(ref, "")) in RISK


if __name__ == "__main__":
    print(f"라벨 {len(_lab)}개월 로드: {_lab.index[0]} ~ {_lab.index[-1]}  ({_path})")
    print("\n예시 판정 (t월 리밸런스 → t-2월 라벨 기준):")
    for m in ["2018-03", "2018-09", "2020-03", "2020-05", "2022-05", "2025-12", "2026-03"]:
        ref = pd.Period(m, freq="M") - 2
        lab = _lab.get(ref, "(라벨 없음)")
        print(f"  {m} 리밸런스 → {ref} 라벨 = {lab:>8s} → 게이트 {'ON (t2 방어)' if defensive_gate(m) else 'OFF (100% 투자)'}")
    n_on = sum(defensive_gate(str(p + 2)) for p in _lab.index)
    print(f"\n전체 180개월 중 게이트 ON: {n_on}개월")
