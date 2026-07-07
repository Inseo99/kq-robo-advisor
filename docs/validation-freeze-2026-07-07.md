# KQ Quant Tool 검증 상태 고정본

기준일: 2026-07-07

이 문서는 발표/제출 전에 "무엇을 구현했다고 말할 수 있고, 무엇은 고도화 항목인지"를 고정하기 위한 요약본이다.

## 1. 발표에서 말해도 되는 검증 범위

다음 문구는 현재 코드 상태와 일치한다.

> AI가 작성하거나 보조한 피처/전략 코드는 단독으로 신뢰하지 않고, Point-in-Time 데이터 검증, IS/OOS, permutation, 게이트웨이, UI E2E 관문을 통과한 경우에만 채택했다.

> 국면 모델은 일반 K-Fold가 아니라 라벨 확정 시점 기준의 purged walk-forward 구조를 사용했다. 다만 Lopez de Prado식 Purged K-Fold/CPCV는 추가 고도화 검증 항목으로 남겨두었다.

## 2. 구현됨 / 고도화 항목 구분

| 항목 | 현재 상태 | 발표 표현 |
|---|---:|---|
| Macro PiT | 구현/실증 PASS | 발표지연 lag와 vintage 테스트로 미래 누수 차단 |
| Price Integrity | 구현/실증 PASS | 수정주가 패널 기준 분할/비정상 점프 검증 |
| Financial PiT | 구현/실증 PASS | 엑셀 재무도 공시지연 기준 asof 조회 |
| Single Gateway | 구현/실증 PASS | 원천 데이터 직접 접근 차단 |
| App Wiring / UI E2E | 구현/실증 PASS | 앱 화면과 검증 파이프라인이 같은 데이터 사용 |
| Transition PiT | 구현/실증 PASS | 백테스트 시점별 확장 전이행렬 사용 |
| Label Vintage | 구현/실증 PASS | 과거 라벨이 미래 데이터 추가로 바뀌지 않는지 검증 |
| TEA Band | 구현/실증 PASS | TEA 목표배분 도입 후 밴드 거래 빈도 점검 |
| Regime ERC v1 | 구현/실증 PASS | 4국면 기준 배분 v1 산출 |
| Purged K-Fold / CPCV | 미구현 | 고도화 항목 |

## 3. 관문 대시보드 상태

최신 대시보드:

- 파일: `tests/analysis_outputs/gates_dashboard.json`
- `overall_status`: `PASS`
- `data_source`: `real`
- `validation_stage`: `empirical_real_data`

실증 관문 9개:

| 관문 | 스크립트 | 상태 |
|---|---|---:|
| macro_pit | `tests/validation_macro_pit.py` | PASS |
| price_integrity | `tests/validation_price_integrity.py` | PASS |
| financial_pit | `tests/validation_financial_pit.py` | PASS |
| single_gateway | `tests/validation_single_gateway.py` | PASS |
| app_wiring | `tests/validation_app_wiring.py` | PASS |
| transition_pit | `tests/validation_transition_pit.py` | PASS |
| label_vintage | `tests/validation_label_vintage.py` | PASS |
| tea_band | `tests/validation_tea_band.py` | PASS |
| regime_erc_v1 | `tests/validation_regime_erc_v1.py` | PASS |

## 4. 전이 레이어 결과

`transition_pit` 실데이터 결과:

- 라벨 기간: 2001-06 ~ 2026-06, 301개월
- 초기 look-ahead 영향 평균 L1: `0.1420`
- 후기 look-ahead 영향 평균 L1: `0.0585`
- 최대 L1: `0.3440`
- 결론: 전체표본 전이행렬을 과거 백테스트에 쓰면 배분이 달라질 수 있으므로, 백테스트에는 확장 윈도우 P를 사용한다.

`label_vintage` 주의:

- 안전 라벨러 vintage 안정성: agreement/kappa `1.000 / 1.000`
- 저장된 라벨과 composite confirm=3 재생성 라벨 일치율: `0.635`
- 해석: 현재 저장 라벨은 다른 승인 config로 생성된 것으로 보이며, 라벨 config는 버전 고정 대상이다.

## 5. TEA와 리밸런싱 해석

TEA는 목표배분 산식이고, 회전율 제어는 리밸런싱 정책이 담당한다.

추천 탭 설명 문구:

> 현재 국면 확률과 P^3 전이확률로 TEA 목표배분을 계산하고, 로보/Alpha Decay는 구성 자산의 진입/청산 시점을 보조하며, 실제 거래는 밴드 리밸런싱과 국면 전환 트리거로 제어한다.

`tea_band` 실데이터 결과:

| 밴드 | Step 거래 | TEA 거래 | Step 총회전율 | TEA 총회전율 |
|---:|---:|---:|---:|---:|
| 3%p | 36 | 41 | 4.912 | 4.795 |
| 5%p | 28 | 30 | 4.761 | 4.541 |
| 7%p | 28 | 26 | 4.874 | 4.343 |

해석:

- 5%p 기본 밴드에서 거래 횟수는 2회 증가했지만 총회전율은 감소했다.
- TEA는 거래를 과하게 늘리는 기능이 아니라 전환 충격을 분산하는 목표배분 레이어로 설명하는 것이 안전하다.

## 6. 4국면 ERC v1 기준 배분

산출 파일:

- `data/analysis_outputs/regime_erc_v1_weights.csv`
- `data/analysis_outputs/regime_erc_v1_audit.csv`
- `data/analysis_outputs/regime_erc_v1_meta.json`

v1 원칙:

- 프록시 미사용
- ETF 실거래 월간 수익률만 사용
- 국면별 경제 논리 상하한 적용
- kappa-shrinkage ERC: `kappa = n_regime / (n_regime + n0)`, `n0 = 36`

| 국면 | 주식 | 채권 | 금 | 현금 |
|---|---:|---:|---:|---:|
| 골디락스 | 35.0% | 35.0% | 15.0% | 15.0% |
| 리플레이션 | 30.0% | 25.0% | 31.0% | 14.0% |
| 스태그플레이션 | 8.0% | 30.0% | 20.0% | 42.0% |
| 디플레이션 | 5.0% | 44.4% | 5.6% | 45.0% |

발표 한계 문구:

> v1은 상장 전 지수 프록시를 사용하지 않고 ETF 실거래 표본만 사용했다. 따라서 장기 국면별 표본이 제한되는 한계가 있으며, 향후 v2에서는 KOSPI200/채권지수/금/원자재 프록시로 장기 표본을 확장할 수 있다.

## 7. 최종 발표용 한 문장

> 이 프로젝트는 수익률이 좋은 전략만 고르는 방식이 아니라, 데이터 관문, 국면 라벨, 전이확률, 기준배분, 추천 UI까지 이어지는 E2E 검증 구조를 만들고, 통과한 기능만 추천 파이프라인에 올리는 RA 시스템이다.
