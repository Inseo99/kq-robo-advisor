# Tasks: AI 시장 국면 인식 (TabPFN + HMM)

PRD: `prd-regime-ai.md`
생성일: 2026-06-26

---

## 관련 파일 (Relevant Files)

### 기존 파일 (수정)
- `data_loader.py` — 변경 없음 (이미 모듈 호환)
- `server.py` — regime_model import + API 엔드포인트 추가
- `index.html` — "AI 시장 국면" 탭 UI 새로 작성
- `requirements.txt` — lightgbm, hmmlearn 추가

### 새 파일 (생성)
- `regime_model.py` — TabPFN + HMM 계층 모델 (이미 작성 완료 ✅)
- `tasks/prd-regime-ai.md` — PRD (이미 작성 완료 ✅)
- `tasks/tasks-prd-regime-ai.md` — 본 파일 ✅

### 본인 PC 설치 필요
- `pip install lightgbm hmmlearn` (필수)
- `pip install tabpfn` (선택, 없으면 LightGBM 자동 사용)

---

## Tasks

### [x] Task 0: 사전 준비 (완료)

- [x] 0.1 `regime_model.py` 모듈 작성 (TabPFN/LightGBM 자동 폴백, HMM 카운팅, 국면 라벨링)
- [x] 0.2 매크로 데이터로 모듈 단위 테스트 (49 분기 학습 성공)
- [x] 0.3 PRD 작성 및 승인

---

### [ ] Task 1: 본인 PC 환경 준비

**목적**: regime_model.py 파일과 필수 패키지를 본인 PC에 설치

- [ ] 1.1 `regime_model.py` 파일 다운로드 후 `kq_tool/` 폴더에 저장
- [ ] 1.2 PowerShell에서 `pip install lightgbm hmmlearn` 실행
- [ ] 1.3 (선택) `pip install tabpfn` — 디스크 1GB 필요. 없어도 OK
- [ ] 1.4 진단 스크립트 실행: `python regime_model.py`
- [ ] 1.5 진단 결과 확인 — `active_classifier`가 `TabPFN` 또는 `LightGBM`이어야 함

**완료 기준**: 콘솔에 다음 출력
```
=== Regime Model 모듈 진단 ===
  tabpfn_available: True 또는 False
  lightgbm_available: True
  hmm_available: True
  active_classifier: TabPFN 또는 LightGBM
```

---

### [ ] Task 2: server.py에 모델 학습 코드 추가

**목적**: 서버 시작 시 regime_model을 자동 학습

수정 위치: `server.py` 상단 import 영역 + 매크로 정의 영역

변경 사항:
- 2.1 상단에 `import regime_model as _rm` 추가
- 2.2 매크로 데이터 로드 다음에 `REGIME_MODEL` 전역 변수 추가
- 2.3 서버 시작 시 `REGIME_MODEL.fit(EXCEL_MACRO)` 호출
- 2.4 실패 시 None 유지 (서버는 계속 동작)
- 2.5 로그 출력: `[regime] 모델 학습 완료 — 사용 분류기: ...`

**완료 기준**:
- 서버 재시작 시 로그에 `[regime] 모델 학습 완료` 메시지 출력
- 기존 다른 기능(스크리너, 백테스트) 영향 없음

**테스트**:
```powershell
python server.py
# 콘솔에 [regime] 메시지 확인
# 브라우저에서 기존 기능 동작 확인
```

---

### [ ] Task 3: `/api/regime_ai` 엔드포인트 추가

**목적**: 프론트엔드가 호출할 새 API 추가

수정 위치: `server.py`의 Handler 클래스

변경 사항:
- 3.1 `do_GET` 라우팅에 `elif p == '/api/regime_ai': self._regime_ai()` 추가
- 3.2 `_regime_ai` 메서드 구현 — REGIME_MODEL.predict_current() 호출
- 3.3 응답에 regime_desc(색상, 라벨, 자산 추천) 포함
- 3.4 모델 학습 안 됐을 때 폴백 응답 (기존 MACRO 반환)

**응답 JSON 스키마**: PRD FR-2.2 참고

**완료 기준**:
- `curl http://127.0.0.1:8888/api/regime_ai` 가 정상 JSON 응답
- 응답에 `current_regime`, `probs`, `transition_matrix` 모두 포함
- `probs` 합계 ≈ 1.0

**테스트**:
```powershell
# 서버 실행 중
curl http://127.0.0.1:8888/api/regime_ai
# JSON 응답 확인
```

---

### [ ] Task 4: index.html "AI 시장 국면" 탭 UI 작성

**목적**: 학습 모델 결과를 시각화

수정 위치: `index.html` — `#panel-macro` 영역

변경 사항:
- 4.1 기존 매크로 탭 내용 백업 (주석 처리)
- 4.2 새 레이아웃 작성:
  - 상단: 현재 국면 큰 배지 + 사용 모델 표시
  - 중단: 4국면 확률 막대그래프
  - 하단: 전환 매트릭스 4×4 표 (색상 강도)
  - 추가: 다음 분기 전환 확률 박스
  - 추가: 국면별 평균 지속 기간 표
- 4.3 JS: `loadMacro()` → `/api/regime_ai` 호출로 변경
- 4.4 JS: `renderRegimeAI()` 새 함수 작성

**완료 기준**:
- 브라우저에서 탭 클릭 시 데이터 정상 표시
- 4국면 확률이 막대그래프로 보임
- 전환 매트릭스가 표로 보임
- 콘솔 에러 없음

**테스트**:
1. 브라우저에서 Ctrl+F5
2. "AI 시장 국면" 탭 클릭
3. 모든 요소 표시 확인
4. 개발자 도구 콘솔에 에러 없는지 확인

---

### [ ] Task 5: 통합 테스트 + 본인 환경 검증

**목적**: 전체 기능이 정상 동작하는지 최종 확인

체크리스트 (PRD 11번 수용 기준):

**백엔드**:
- [ ] 서버 시작 로그에 `[regime] 모델 학습 완료` 메시지
- [ ] `/api/regime_ai` 응답에 `current_regime`, `probs`, `transition_matrix` 포함
- [ ] `probs` 합계가 0.99~1.01
- [ ] `transition_matrix` 각 행 합계가 0.99~1.01

**프론트엔드**:
- [ ] "AI 시장 국면" 탭에서 현재 국면이 큰 글씨로 표시
- [ ] 4국면 확률 막대그래프 표시
- [ ] 전환 매트릭스 4×4 표 표시
- [ ] 사용 모델 이름 표시

**기존 기능 영향 없음**:
- [ ] 종목 검색 (삼성전자) 정상
- [ ] 스크리너 실행 정상
- [ ] ETF 배분 정상
- [ ] 전략검증 정상
- [ ] 로보신호 정상

**완료 기준**: 모든 체크박스 ✅

---

## Important Rules

작업 진행 시 지켜야 할 규칙:

1. **한 번에 하나의 Task만 진행** — 동시에 여러 파일 수정 금지
2. **각 Task 완료 후 본인이 테스트** — OK 받기 전엔 다음 단계 진행 금지
3. **기존 인터페이스 변경 금지** — 기존 API, 함수 시그니처 유지
4. **요청 안 한 리팩터링 금지** — 코드 정리는 별도 작업
5. **실패하면 해당 작업만 롤백** — 전체 안 망가지게
6. **로그 남기기** — 각 Task 완료 시 어떤 파일을 어떻게 바꿨는지 기록

---

## Implementation Notes (구현 기록)

### Task 0 (완료)
- 작성 파일: `regime_model.py` (350줄)
- 테스트 데이터: 매크로 12년 → 49 분기 라벨링 성공
- 분류기: LightGBM 자동 선택 (TabPFN 미설치 환경)
- 전환 매트릭스: 한국 시장 12년 통계 도출
  - 골디락스 → 디플레이션 전환 81.8%
  - 디플레이션 → 골디락스 회복 69.2%
- 현재 국면 (2026 Q1): 디플레이션, 다음 분기 골디락스 회복 확률 69%

### Task 1 (대기)
- 시작 전: 본인 PC에서 regime_model.py 파일 받기
- 진행 중에 작성

### Task 2~5 (대기)
- 각 Task 완료 후 여기에 기록

---

## 진행 상태

| Task | 상태 | 담당 |
|---|---|---|
| Task 0 | ✅ 완료 | Claude |
| Task 1 | ⏳ 대기 (사용자 PC 설치) | 사용자 |
| Task 2 | ⏳ 대기 | Claude |
| Task 3 | ⏳ 대기 | Claude |
| Task 4 | ⏳ 대기 | Claude |
| Task 5 | ⏳ 대기 (사용자 검증) | 사용자 + Claude |

---

## 다음 작업

**Task 1**: 본인 PC 환경 준비

먼저 `regime_model.py` 파일과 설치 가이드를 받아서 진행하면 됩니다.
Task 1이 완료되면 (사용자가 콘솔 출력 보여주면) → Task 2 진행.
