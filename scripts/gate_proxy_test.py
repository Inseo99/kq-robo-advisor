# -*- coding: utf-8 -*-
r"""게이트 로직 프록시 검증 — 이진 게이트(원안) vs 확률 비례(수정안), KOSPI 자산 수준.

사전 등록 (실행 전 고정, 시도 1회):
  대상: 게이트 로직 단독 효과. 종목선정(Kang 등)은 분리 — 본 검증은 자산(주식 인덱스)
        수준 프록시이며, 팀원 백테스터의 재실행을 대체하지 않는다.
  변형 3종 (공통 기간, 월간 리밸런스, 비주식분은 현금):
    A 기준     : 주식 100% 상시
    B 이진(원안): t월에 t-2월 채택 라벨이 위험(스태그/디플레)이면 주식 0%, 아니면 100%
    C 확률(수정): 주식 목표비중 = 1 − p_risk(t−1), p_risk = P(스태그)+P(디플레) 나우캐스트
                 (t−1월 말에 확정된 확률만 사용) + 70% 부분 이동(메인 배분 레이어 상수 재사용)
  지표: CAGR, MDD, Sharpe(연율), Calmar. 거래비용 미반영(상대 비교용 프록시임을 명시).
  사전 판정 기준:
    - B 평가: A 대비 MDD 개선 없으면 "이진 게이트 프록시 기각"
    - C 평가: A 대비 MDD 개선 AND Calmar ≥ A 이면 "확률 조절 프록시 통과(후속 검증 지지)"
  시점 규율: 라벨 t−2 (defensive_gate와 동일), 확률 t−1 (월말 확정 후 익월 적용).
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RISK = {"스태그플레이션", "디플레이션"}
PARTIAL = 0.7  # 메인 배분 레이어의 부분 이동 상수 재사용 (신규 튜닝 아님)

ret = pd.read_csv(os.path.join(ROOT, "data/assets/monthly_returns.csv"), parse_dates=["date"]).set_index("date")
ret.index = ret.index.to_period("M")
lab = pd.read_csv(os.path.join(ROOT, "data/regime/labels.csv"), encoding="utf-8-sig",
                  parse_dates=["date"]).set_index("date")["regime"]
lab.index = lab.index.to_period("M")
probs = pd.read_csv(os.path.join(ROOT, "data/analysis_outputs/probs_nowcast_under_n5.csv"),
                    parse_dates=["date"]).set_index("date")
probs.index = probs.index.to_period("M")
p_risk = (probs["스태그플레이션"] + probs["디플레이션"]).shift(1)  # t-1월 말 확정 확률

# 공통 기간: 확률(2016-04 이후 +1) ∩ 수익률 ∩ 라벨 유효(t-2)
months = [m for m in ret.index if m in p_risk.dropna().index and (m - 2) in lab.index]
months = pd.PeriodIndex(months, freq="M")
print(f"공통 평가 기간: {months[0]} ~ {months[-1]} ({len(months)}개월)\n")

stock, cash = ret.loc[months, "stocks"], ret.loc[months, "cash"]


def metrics(r: pd.Series) -> dict:
    curve = (1 + r).cumprod()
    yrs = len(r) / 12
    cagr = curve.iloc[-1] ** (1 / yrs) - 1
    mdd = (curve / curve.cummax() - 1).min()
    sharpe = r.mean() / r.std() * np.sqrt(12)
    return {"CAGR%": round(cagr * 100, 2), "MDD%": round(mdd * 100, 2),
            "Sharpe": round(sharpe, 2), "Calmar": round(cagr / abs(mdd), 2)}


# A: 상시 주식
rA = stock.copy()

# B: 이진 게이트 (t-2 라벨 위험 → 주식 0)
gate_on = pd.Series([str(lab.get(m - 2, "")) in RISK for m in months], index=months)
rB = stock.where(~gate_on, cash)

# C: 확률 비례 + 70% 부분 이동
w_t = (1 - p_risk.loc[months]).clip(0, 1)
w = []
prev = w_t.iloc[0]
for tgt in w_t:
    prev = prev + PARTIAL * (tgt - prev)
    w.append(prev)
w = pd.Series(w, index=months)
rC = w * stock + (1 - w) * cash

rows = {}
rows["A 주식 100% (기준)"] = metrics(rA)
rows[f"B 이진 게이트 (원안, ON {int(gate_on.sum())}개월)"] = metrics(rB)
rows[f"C 확률 비례+70%이동 (수정안, 평균비중 {w.mean():.0%})"] = metrics(rC)
df = pd.DataFrame(rows).T
print(df.to_string())

# 사전 판정
mA, mB, mC = metrics(rA), metrics(rB), metrics(rC)
b_ok = mB["MDD%"] > mA["MDD%"]
c_ok = (mC["MDD%"] > mA["MDD%"]) and (mC["Calmar"] >= mA["Calmar"])
print(f"\n[사전 판정] B 이진(원안): {'MDD 개선 → 프록시 유지' if b_ok else 'MDD 개선 없음 → 프록시 기각'}")
print(f"[사전 판정] C 확률(수정): {'MDD 개선+Calmar 비악화 → 프록시 통과' if c_ok else '기준 미달 → 프록시 기각'}")

# 에피소드: 2020 코로나
ep = pd.period_range("2020-01", "2020-06", freq="M")
epdf = pd.DataFrame({"주식수익률%": (stock.loc[ep] * 100).round(1),
                     "B게이트": gate_on.loc[ep].map({True: "ON", False: "off"}),
                     "C주식비중%": (w.loc[ep] * 100).round(0)})
print("\n== 2020 코로나 에피소드 ==")
print(epdf.to_string())

out = os.path.join(ROOT, "data", "analysis_outputs", "gate_proxy_test.csv")
df.to_csv(out, encoding="utf-8-sig")
print(f"\n저장 → {out}")
print("[면책] 자산 수준 프록시 — 종목선정 전략의 재실행이 아니며, 게이트 로직 단독 비교용.")
