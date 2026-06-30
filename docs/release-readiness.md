# KQ Quant Tool Release Readiness

이 문서는 팀원 공유, 데모, 발표 전 최종 확인용 요약입니다.

## 현재 상태

- 기존 `server.py` + `index.html` 앱은 계속 실행 가능한 상태입니다.
- 핵심 계산 로직은 `src/kq_tool` 패키지로 단계적으로 분리되어 있습니다.
- 단위 테스트 기준: `tests/unit` 285개.
- 기본 회귀검증 스크립트: `RUN_REGRESSION_CHECKS.bat`.
- 팀원 실행 스크립트: `RUN_KQ_TOOL.bat`.

## 실행 명령

PowerShell에서 직접 실행할 때:

```powershell
cd C:\Users\paenco0313\Downloads\kq_tool
.\RUN_KQ_TOOL.bat
```

패키지 진입점으로 직접 실행할 때:

```powershell
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
python -m kq_tool
```

브라우저에서:

```text
http://127.0.0.1:8888/
```

## 최종 검증 명령

기본 회귀검증:

```powershell
powershell -ExecutionPolicy Bypass -File .\RUN_REGRESSION_CHECKS.ps1
```

서버 실행 후 기본 API smoke:

```powershell
python tests\smoke_api.py
```

서버 실행 후 주요 API smoke:

```powershell
python tests\smoke_api.py --include-stock --include-core --timeout 180
```

이 smoke는 스크리너의 `S2모멘텀`, 전략검증의 `quant`, `quant_s2`,
`quant_compare`, 추천 포트폴리오 응답을 함께 확인합니다.

## 통과해야 하는 기준

- Python syntax 통과.
- `src/kq_tool` package compile 통과.
- `tests/unit` 전체 통과.
- `/api/ping`, `/api/health` smoke 통과.
- 데모 전 `/api/stock`, `/api/screen`, `/api/stratbt`, `/api/recommend_portfolio` smoke 통과.
  특히 `/api/stratbt?s=quant_compare`는 3자 비교 응답을 반환해야 함.
- 브라우저에서 로보신호, 퀀트 스크리너, 전략검증, 추천 포트폴리오 탭이 열림.
- 전략검증에서 `quant_compare` API 응답으로 퀀트(모멘텀), 퀀트(S2모멘텀), KOSPI 3자 비교가 표시됨.

## 핵심 설명 포인트

- 이 프로젝트는 단순히 QUANT/ROBO가 KOSPI를 이기는지 보는 도구가 아닙니다.
- 목적은 국면 필터, 정적/동적 자산배분, 로보신호, Alpha Decay를 결합해 손실 구간과 재점검 시점을 설명하는 것입니다.
- 국면 모델은 초과수익 엔진이라기보다 자산배분 위험 상태와 재평가 트리거로 사용합니다.
- Alpha Decay는 매수/매도 보장 신호가 아니라 신호 유효기간과 재점검 기간을 안내하는 보조 레이어입니다.
- 비용/슬리피지 반영 후에도 유효한 결과만 채택한다는 운영 기준을 둡니다.
- S2 모멘텀은 최근 1개월을 제외한 12-1개월 모멘텀으로, 기존 최근 3개월 모멘텀과 별도로 검증합니다.

## 남은 선택 작업

- 서버를 직접 실행한 상태에서 `--include-core` smoke를 실제 데이터로 1회 확인.
- 발표용 슬라이드 또는 면접용 1페이지 요약 작성.
- `server.py` HTTP handler를 더 얇게 만드는 추가 리팩토링.
- FastAPI 전환은 선택 과제입니다. 현재 로컬 데모/팀원 실행에는 필수는 아닙니다.




