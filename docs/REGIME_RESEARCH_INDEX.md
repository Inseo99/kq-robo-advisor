# 국면 연구 산출물 인덱스 (2026-07-10 기준)

처음 보는 사람은 위에서 아래로 읽으면 된다.

## 읽는 순서

| # | 문서 | 내용 |
|---|---|---|
| 1 | docs/REGIME_FINAL_DECISION.md | **최종 결정문** — N5 채택, 근거, 거버넌스 기록 |
| 2 | regime_next_실험로그.md | 실험 전문 — 사이클 1~4, 시도별 사전 등록·채점·판정 |
| 3 | docs/REGIME_N7_ADOPTION.md | N7 심의 자료 — 정밀 명세, 과최적화 평가, 확률 레이어 검증 |
| 4 | docs/METHODS_초안.md · docs/DISCUSSION_초안.md | 논문 절 초안 |
| 5 | docs/HOLDOUT_2026_FINDINGS.md | 홀드아웃 1차 대조 발견 (지속성 앵커링) |

## 라벨 파일 (정본과 참고본)

| 파일 | 성격 | 용도 |
|---|---|---|
| data/regime/labels.csv · data/macro/regime_labels.csv | **채택 정본 (N5, 실시간 트랙)** | 앱·백테스트·ERC |
| data/regime/labels_v2_adopted_backup.csv | V2 백업 (채택 직전) | 대조 |
| data/analysis_outputs/labels_expost_confirmed_*.csv | 사후 확정 트랙 (N8) | **회고 분석·그림 전용 — PiT 문맥 금지** |
| data/analysis_outputs/labels_walkforward_*_n{1,3,5,6,7}.csv | 실험 후보들 | 이력 |
| data/analysis_outputs/labels_hmm_benchmark.csv | HMM 벤치마크 (전체 표본) | 비교 연구 전용 |

## 스크립트

| 스크립트 | 역할 |
|---|---|
| scripts/regime_next_n5.py | **채택 라벨 재생성 + 채점 전문** (국면 목록 보기: 이것) |
| scripts/walkforward_labels.py | V2(구 규칙) 워크포워드 — 검산용 |
| scripts/regime_next_n8_confirmed.py | 사후 확정 트랙 생성 |
| scripts/provisional_judgment.py | 라벨 공백기 잠정 판정 (공표 지연 반영) |
| scripts/log_judgment.py | **운영 판정 월간 로거 — 월초 1회 실행 권장** |
| scripts/holdout_2026_check.py | 2026 홀드아웃 대조표 |
| scripts/regime_next_fold_check.py | 시기별 폴드 강건성 |
| scripts/research_hmm_comparison.py | HMM 벤치마크 (numpy 자체 구현) |
| scripts/make_labels.py | 구 규칙 생성기 — **가드 있음** (--force 없이는 채택본 덮어쓰기 불가) |

## 발표·심의 자료

| 파일 | 상태 |
|---|---|
| n5_final_decision.pptx | **최신 — 최종 결정 요약 (팀·캡스톤용)** |
| n7_adoption_review.pptx | 구버전 — N7 심의 시점 자료 (이력 보존용, 결정과 다름) |
| before_after_labels.pptx | 발표용 — look-ahead 교정 전후 비교 |

## 남은 후속 과제 (사전 등록 필요)

1. 판정 로직의 앱 통합 — ui_payload에서 라벨 공백기에 잠정 판정 표시 (+ "잠정" 배지)
2. N7 재심의 — 분류기 확률 경화의 근본 처방(학습 방식 개선) 이후
3. 금융 변수 축 추가 검토 · 워크포워드 기반 HMM/MS 엄밀 비교
