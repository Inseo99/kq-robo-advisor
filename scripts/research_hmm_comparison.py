# -*- coding: utf-8 -*-
r"""R1 — 규칙 기반 라벨 vs 가우시안 HMM 잠재 상태 비교 (연구 벤치마크, 채택 실험 아님).

목적 (사전 등록):
  Discussion §5의 "HMM/Markov-switching 비교". 잠재 상태를 데이터에서 직접 추정하는
  접근이 본 연구의 규칙 기반 4국면 정의와 얼마나 수렴하는지 측정한다. 수렴하면
  규칙 정의의 독립적 타당성 근거, 크게 어긋나면 규칙이 놓치는 구조의 단서가 된다.

방법:
  - 입력: 분기 GDP QoQ·CPI YoY (라벨과 동일 축), 2001Q1~2025Q4, 표준화
  - 모형: 4상태 가우시안 HMM (대각 공분산), 해밀턴 필터 + EM (numpy 자체 구현)
  - 다중 초기화(고정 시드 20개) 후 최대 로그우도 해 선택 — 재현 가능
  - 상태→국면 매핑: 상태별 평균 벡터의 부호 사분면 (성장±, 물가±) — 튜닝 없음
  - ** 전체 표본 추정(ex-post) — 실시간 판정·백테스트와 비교 불가, 벤치마크 전용 **

보고 (기술 통계, 합격/기각 없음):
  월 확장(분기 상태를 3개월 복제) 후 2011-01~2025-12에서
  vs 실시간 라벨(N5)·확정 라벨(N8) 일치율, 위기 앵커 월의 HMM 상태, 전환 수.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "analysis_outputs")
REGIMES = ["골디락스", "리플레이션", "스태그플레이션", "디플레이션"]


# ---------------- 가우시안 HMM (대각 공분산) ----------------
def _log_gauss(X, mu, var):
    """(T,D),(K,D),(K,D) -> (T,K) 로그밀도."""
    T, D = X.shape
    K = mu.shape[0]
    out = np.empty((T, K))
    for k in range(K):
        out[:, k] = -0.5 * (np.log(2 * np.pi * var[k]) + (X - mu[k]) ** 2 / var[k]).sum(axis=1)
    return out


def _forward_backward(logB, logA, logpi):
    T, K = logB.shape
    la = np.empty((T, K)); lb = np.zeros((T, K))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + _logsumexp(la[t - 1][:, None] + logA, axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = _logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = _logsumexp(la[-1], axis=0)
    lg = la + lb - ll
    gamma = np.exp(lg - _logsumexp(lg, axis=1)[:, None])
    # xi 합 (전이 기대 횟수)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        m = la[t][:, None] + logA + logB[t + 1][None, :] + lb[t + 1][None, :] - ll
        xi_sum += np.exp(m - _logsumexp(m))  # 정규화 후 합=1이 아니라, 아래서 재정규화
    return gamma, xi_sum, ll


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    r = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(r, axis=axis) if axis is not None else float(np.squeeze(r))


def fit_hmm(X, K=4, seed=0, n_iter=200, tol=1e-6):
    rng = np.random.default_rng(seed)
    T, D = X.shape
    mu = X[rng.choice(T, K, replace=False)].copy()
    var = np.tile(X.var(axis=0), (K, 1))
    A = np.full((K, K), 0.1 / (K - 1)); np.fill_diagonal(A, 0.9)
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(n_iter):
        logB = _log_gauss(X, mu, var)
        gamma, xi_sum, ll = _forward_backward(logB, np.log(A), np.log(pi))
        if ll - prev < tol and prev != -np.inf:
            break
        prev = ll
        pi = gamma[0] / gamma[0].sum()
        A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)
        mu = (gamma.T @ X) / w[:, None]
        for k in range(K):
            var[k] = (gamma[:, k][:, None] * (X - mu[k]) ** 2).sum(axis=0) / w[k]
        var = np.maximum(var, 1e-4)
    return dict(mu=mu, var=var, A=A, pi=pi, loglik=prev, gamma=gamma)


def map_states(mu):
    """상태 평균 부호 사분면 -> 국면명. (표준화 공간: 0 = 표본 평균)"""
    names = {}
    for k, (g, p) in enumerate(mu):
        if g >= 0 and p < 0:
            names[k] = "골디락스"
        elif g >= 0 and p >= 0:
            names[k] = "리플레이션"
        elif g < 0 and p >= 0:
            names[k] = "스태그플레이션"
        else:
            names[k] = "디플레이션"
    return names


def main():
    gdp = pd.read_csv(os.path.join(ROOT, "data/macro/gdp_qoq.csv"), parse_dates=["date"]).set_index("date")["value"].resample("QE").last()
    cpi = pd.read_csv(os.path.join(ROOT, "data/macro/cpi_yoy.csv"), parse_dates=["date"]).set_index("date")["value"].resample("QE").last()
    df = pd.DataFrame({"gdp": gdp, "cpi": cpi}).dropna().loc["2001-01-01":"2025-12-31"]
    X = ((df - df.mean()) / df.std(ddof=0)).to_numpy()
    print(f"입력: {len(df)}분기 ({df.index[0]:%Y-%m} ~ {df.index[-1]:%Y-%m})")

    best = None
    for seed in range(20):
        fit = fit_hmm(X, K=4, seed=seed)
        if best is None or fit["loglik"] > best["loglik"]:
            best = fit
    names = map_states(best["mu"])
    print(f"최적 로그우도: {best['loglik']:.2f} · 상태 매핑: " +
          ", ".join(f"S{k}→{v}(g={best['mu'][k][0]:+.2f},p={best['mu'][k][1]:+.2f})" for k, v in names.items()))
    dup = len(set(names.values())) < 4
    if dup:
        print("[주의] 상태 사분면 중복 — 4국면 완전 대응 아님 (그대로 보고)")

    state = best["gamma"].argmax(axis=1)
    q_lab = pd.Series([names[s] for s in state], index=df.index, name="hmm")
    # 분기 -> 월 확장
    m_idx = pd.date_range("2011-01-31", "2025-12-31", freq="ME")
    hmm_m = q_lab.resample("ME").ffill().reindex(m_idx).ffill()
    hmm_m.rename_axis("date").to_csv(os.path.join(OUT, "labels_hmm_benchmark.csv"), encoding="utf-8-sig", date_format="%Y-%m-%d")

    n5 = pd.read_csv(os.path.join(OUT, "labels_walkforward_10y_n5.csv"), index_col=0, parse_dates=True)["regime"].reindex(m_idx)
    n8 = pd.read_csv(os.path.join(OUT, "labels_expost_confirmed_10y.csv"), index_col=0, parse_dates=True)["regime"].reindex(m_idx)
    both5 = pd.DataFrame({"a": hmm_m, "b": n5}).dropna()
    both8 = pd.DataFrame({"a": hmm_m, "b": n8}).dropna()
    RISK = {"스태그플레이션", "디플레이션"}
    risk5 = ((both5.a.isin(RISK)) == (both5.b.isin(RISK))).mean()
    print(f"\nHMM vs 실시간(N5): 4국면 일치 {(both5.a == both5.b).mean():.1%} · 위험/안전 2분류 일치 {risk5:.1%}")
    print(f"HMM vs 확정(N8):  4국면 일치 {(both8.a == both8.b).mean():.1%} · 위험/안전 2분류 일치 {((both8.a.isin(RISK)) == (both8.b.isin(RISK))).mean():.1%}")
    tr = int((q_lab != q_lab.shift()).sum() - 1)
    print(f"HMM 전환(분기 기준, 2001~2025): {tr}회")
    print("\n앵커 월의 HMM 상태:")
    for ym in ["2018-07", "2020-03", "2020-12", "2021-09", "2022-03"]:
        v = hmm_m[hmm_m.index.strftime("%Y-%m") == ym]
        print(f"  {ym}: {v.iloc[0] if len(v) else '-'}")
    print("\n[성격] 전체 표본 사후 추정 벤치마크 — 실시간 판정·백테스트와 직접 비교 불가")


if __name__ == "__main__":
    main()
