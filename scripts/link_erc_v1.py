"""scripts/link_erc_v1.py — ERC v1 기준배분 → 추천 탭 연결 (1~5단계 지원 도구)
사용: python scripts/link_erc_v1.py            # 1~3단계: 로드·검증·비교 리포트
      python scripts/link_erc_v1.py --apply    # (연결 코드 반영 후) 4단계: 추천 diff

단계 대응:
  1) regime_erc_v1_weights.csv 로드 + 무결성 검증  → 이 스크립트
  2) TEA가 4행을 사용하도록 연결                   → 아래 [통합 지점] 주석대로 수동 1곳
  3) 기존 REGIME_TARGETS와 비교                    → 이 스크립트 (diff 표 출력)
  4) 추천 결과 변화 검증                            → --apply (recommender 호출 훅)
  5) 화면 "ERC v1 기준배분 사용" 표시               → PATCH_NOTES.md의 HTML 스니펫
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

WEIGHTS_CSV = os.path.join(ROOT, "data", "analysis_outputs", "regime_erc_v1_weights.csv")
EXPECTED_REGIMES = 4          # 4국면
TOL = 1e-6

ASSET_CLASS_MAP = {
    "stocks": ["069500.KS", "229200.KS"],
    "bonds": ["148070.KS", "114260.KS"],
    "gold": ["132030.KS"],
    "cash": ["153130.KS"],
}


def load_and_validate() -> pd.DataFrame:
    """1단계: 로드 + 무결성 검증 (합=1, 4국면, 음수 없음)"""
    df = pd.read_csv(WEIGHTS_CSV, index_col=0)
    problems = []
    if len(df) != EXPECTED_REGIMES:
        problems.append(f"국면 수 {len(df)} != {EXPECTED_REGIMES}")
    row_sums = df.sum(axis=1)
    bad = row_sums[(row_sums - 1.0).abs() > 1e-4]
    if not bad.empty:
        problems.append(f"비중 합≠1 국면: {dict(bad.round(4))}")
    if (df < -TOL).any().any():
        problems.append("음수 비중 존재")
    if problems:
        raise SystemExit("[FAIL] ERC v1 무결성:\n  - " + "\n  - ".join(problems))
    print(f"[OK] 1단계: {WEIGHTS_CSV}")
    print(f"     국면 {list(df.index)}, 자산 {len(df.columns)}개, 행합 전부 1.0")
    return df


def _targets_to_frame(targets: dict, regimes) -> pd.DataFrame:
    return pd.DataFrame(targets).T.reindex(regimes).fillna(0.0)


def _to_asset_class_frame(weights: pd.DataFrame) -> pd.DataFrame:
    """Convert ETF-level weights to the ERC v1 four asset classes when needed."""

    if set(ASSET_CLASS_MAP).issubset(weights.columns):
        return weights[list(ASSET_CLASS_MAP)].copy()
    rows = {}
    for asset_class, tickers in ASSET_CLASS_MAP.items():
        existing = [ticker for ticker in tickers if ticker in weights.columns]
        rows[asset_class] = weights[existing].sum(axis=1) if existing else 0.0
    return pd.DataFrame(rows, index=weights.index)


def compare_with_regime_targets(erc: pd.DataFrame):
    """3단계: 기존 하드코딩 국면표와 ERC v1을 같은 자산군 층위에서 비교."""
    targets = None
    for modpath, attr in (("kq_tool.portfolio.recommender", "LEGACY_REGIME_TARGETS"),
                          ("kq_tool.config", "REGIME_TARGETS"),
                          ("kq_tool.portfolio.recommender", "REGIME_TARGETS"),
                          ("kq_tool.portfolio.transition_allocation", "REGIME_TARGETS")):
        try:
            mod = __import__(modpath, fromlist=[attr])
            targets = getattr(mod, attr, None)
            if targets is not None:
                print(f"[OK] REGIME_TARGETS 발견: {modpath}.{attr}")
                break
        except ImportError:
            continue
    if targets is None:
        print("[SKIP] REGIME_TARGETS를 자동으로 못 찾음 — 위 후보 외 위치면 "
              "modpath를 추가하거나 아래에 dict를 직접 붙여넣으세요.")
        return None

    old_raw = _targets_to_frame(targets, erc.index)
    old = _to_asset_class_frame(old_raw).reindex(index=erc.index, columns=erc.columns).fillna(0.0)
    erc_aligned = erc.reindex(index=old.index, columns=old.columns).fillna(0.0)
    if old.empty or erc_aligned.empty or not list(old.columns):
        print("[SKIP] 비교 가능한 공통 자산군이 없습니다.")
        print(f"  ERC columns: {list(erc.columns)}")
        print(f"  TARGET columns: {list(old_raw.columns)}")
        return None

    diff = (erc_aligned - old).round(4)
    print("\n[3단계] ERC v1 - 기존 국면표 (4자산군 합산, 양수=ERC가 더 큼)")
    print(f"  TARGET 원본 자산: {list(old_raw.columns)}")
    print(f"  비교 자산군: {list(diff.columns)}")
    print(diff.to_string())
    abs_diff = diff.abs()
    max_move = abs_diff.max().max()
    max_where = abs_diff.stack().idxmax() if max_move > 0 else ("-", "-")
    print(f"\n최대 이동폭: {max_move:.1%}p (국면·자산군: {max_where})")
    out = os.path.join(ROOT, "data", "analysis_outputs", "erc_v1_vs_targets_diff.csv")
    diff.to_csv(out, encoding="utf-8-sig")
    print(f"저장: {out}")
    return diff


def recommendation_diff():
    """4단계: 기존 vs ERC v1 기준배분으로 추천 결과 비교.
    [통합 지점] recommender가 기준배분을 인자로 받도록 아래 형태를 권장:
        def recommend(..., base_allocation: dict | None = None):
            alloc = base_allocation or REGIME_TARGETS
    그 후 이 함수의 두 호출이 동작합니다.
    """
    try:
        from kq_tool.portfolio import recommender
    except ImportError as e:
        raise SystemExit(f"[FAIL] recommender import 실패: {e}")
    if not hasattr(recommender, "recommend"):
        raise SystemExit("[TODO] recommender.recommend(...)에 base_allocation 인자를 "
                         "추가한 뒤 이 함수 내부의 호출부를 맞춰주세요. "
                         "(위 docstring의 [통합 지점] 참조)")
    erc = load_and_validate()
    print("[TODO] 아래 두 줄을 실제 recommend 시그니처에 맞춰 활성화:")
    print("  old = recommender.recommend()")
    print("  new = recommender.recommend(base_allocation=erc.to_dict('index'))")
    print("비교 항목: 자산별 비중 delta, 회전율, 예상 성격 변화(방어/공격)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="4단계 추천 diff 실행")
    args = ap.parse_args()
    erc = load_and_validate()
    compare_with_regime_targets(erc)
    if args.apply:
        recommendation_diff()
    else:
        print("\n다음: 2단계 [통합 지점] 연결 → python scripts/link_erc_v1.py --apply")
