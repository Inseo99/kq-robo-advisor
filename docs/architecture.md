# KQ Quant Tool — 시스템 아키텍처

## 1. 시스템 개요

```
┌─────────────────────────────────────────────────────────────┐
│                       사용자 (브라우저)                       │
│                  Windows Chrome / Edge                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  서버: WSL Ubuntu에서 python3 server.py (포트 8888)         │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HTTPServer (Python 표준 라이브러리)                  │  │
│  │  - Handler 클래스: do_GET 라우팅                      │  │
│  │  - 정적 파일: index.html, *.html                      │  │
│  │  - API 엔드포인트:                                    │  │
│  │    /api/ping, /api/stock, /api/screener,             │  │
│  │    /api/etf_alloc, /api/backtest,                    │  │
│  │    /api/macro, /api/regime_ai                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  분석 엔진 (server.py 내장)                           │  │
│  │  - _fetch_one: 종목 단일 분석                         │  │
│  │  - _run_screener: 시총 상위 200 스크리너              │  │
│  │  - _backtest: 12년 백테스트                           │  │
│  │  - _build_macro: 매크로 국면 분류                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AI 시장 국면 모델 (regime_model.py)                  │  │
│  │  - HierarchicalRegimeModel: 통합 파이프라인           │  │
│  │  - RegimeClassifier: TabPFN/LightGBM 자동 폴백        │  │
│  │  - RegimeTransitionModel: HMM 카운팅                  │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────┬──────────────────────────────────────────┬───────┘
           │                                          │
           ▼                                          ▼
┌──────────────────────────┐     ┌──────────────────────────────┐
│  데이터 레이어            │     │  외부 API (보조)              │
│  (data_loader.py)        │     │                              │
│                          │     │  yfinance:                    │
│  - load_stocks()         │     │  - ETF, KOSPI 지수            │
│  - load_financials()     │     │  - 실시간 현재가              │
│  - load_macro()          │     │  - 종목 메타정보              │
│  - get_ohlcv_df()        │     │                              │
│  - get_metric_value()    │     │  TabPFN (선택적):             │
│  - get_price_series()    │     │  - 한 번 인증 후 캐시 사용     │
└──────────┬───────────────┘     └──────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│  파일 저장소 (data/)                                          │
│                                                              │
│  엑셀 원본 (data/*.xlsx):                                     │
│  - 1. 종목 정보 (년간) — 2.1 MB                              │
│  - 2. 주가 데이터 (파트 1, 2) — 1.2 GB                       │
│  - 3. 재무 데이터 (분기) — 43 MB                             │
│  - 주식 이외 데이터.xlsx — 352 KB                            │
│                                                              │
│  parquet 캐시 (data/cache/):                                  │
│  - stocks.parquet — 자동 생성                                │
│  - fin.parquet — 자동 생성                                   │
│  - macro.parquet — 자동 생성                                 │
│  - price_*.parquet (10개) — convert.py로 생성                │
│  - metric_*.parquet (9개) — convert.py로 생성                │
└─────────────────────────────────────────────────────────────┘
```

## 2. 모듈 책임

### 2.1 `server.py` (분석 서버)
- HTTP 서버 + 라우팅
- 종목별 분석 로직
- 스크리너 / 백테스트 엔진
- 매크로 국면 통합

### 2.2 `data_loader.py` (데이터 레이어)
- 엑셀 ↔ parquet 변환
- 종목 / 재무 / 매크로 / 주가 / 메트릭 로드
- 캐시 관리 (메모리 + 디스크)
- 변경 빈도: 낮음 (데이터 스키마 안정)

### 2.3 `regime_model.py` (AI 모델)
- 매크로 국면 분류 (TabPFN/LightGBM)
- 국면 전환 매트릭스 (HMM)
- 국면 라벨링
- 변경 빈도: 중간 (모델 개선 가능)

### 2.4 `index.html` (UI)
- 단일 페이지 SPA
- 5개 탭: 로보신호, 스크리너, ETF배분, 전략검증, AI 시장 국면
- Chart.js 활용
- 변경 빈도: 높음 (시각화 개선)

### 2.5 `convert.py` (1회성 도구)
- 1.2 GB 주가 엑셀 → parquet 변환
- 본인 PC에서 한 번만 실행
- 결과: 19개 parquet 파일

## 3. 데이터 흐름

### 3.1 서버 시작 시
```
1. data_loader.load_stocks() → stocks.parquet 캐시
2. data_loader.load_financials() → fin.parquet 캐시
3. data_loader.load_macro() → macro.parquet 캐시
4. UNIVERSE 생성 (3,070 종목)
5. regime_model.HierarchicalRegimeModel.fit() → TabPFN 학습
6. HTTPServer 시작 (포트 8888)
```

### 3.2 종목 분석 요청
```
사용자: "삼성전자 분석"
  ↓
브라우저: GET /api/stock?t=005930.KS
  ↓
server: _fetch_one('005930.KS')
  ↓
data_loader: get_ohlcv_df('005930') → parquet 캐시
  ↓
server: 기술적 지표 계산 (RSI, MACD, 볼린저)
  ↓
server: yfinance 실시간 현재가 (옵션)
  ↓
JSON 응답 → 브라우저 렌더링
```

### 3.3 AI 시장 국면 요청
```
사용자: "AI 시장 국면 탭"
  ↓
브라우저: GET /api/regime_ai
  ↓
server: REGIME_MODEL.predict_current()
  ↓
regime_model:
  - TabPFN: 4국면 확률
  - HMM: 다음 분기 전환 확률
  ↓
JSON 응답 → renderRegimeAI() → 화면
```

## 4. 외부 시스템 연동

### 4.1 yfinance
- **목적**: 한국 주식 yfinance 우선, 엑셀 보조
- **제한**: 가끔 IP 차단, 응답 지연
- **대응**: 캐시 활용, 타임아웃 짧게, 엑셀 폴백

### 4.2 TabPFN (Prior Labs)
- **목적**: 매크로 국면 분류 (선택적)
- **제한**: 라이선스 가입 필요, Windows 호환성 문제
- **대응**: 환경변수 + WSL Ubuntu, LightGBM 폴백

## 5. 배포 환경

### 5.1 개발 환경
- WSL Ubuntu (Windows 10/11)
- Python 3.14
- pip 패키지 (`pip install --break-system-packages`)

### 5.2 실행 명령
```bash
# 1회만 (주가 엑셀 → parquet)
cd /mnt/c/Users/paenco0313/Downloads/AI퀀트분석데이터셋
python3 convert.py

# 매번
cd /mnt/c/Users/paenco0313/Downloads/kq_tool
python3 server.py
```

### 5.3 발표 환경
- WSL Ubuntu 또는 Linux Mac
- 미리 모든 캐시 생성
- TabPFN API 키 환경변수 설정
- 인터넷 의존도 낮춤 (오프라인 시연 가능)

## 6. 확장 포인트

### 6.1 새 데이터 소스
- `data_loader.py`에 새 함수 추가
- 캐시 파일명 규칙: `{source}_{type}.parquet`
- PRD 필수

### 6.2 새 분석 알고리즘
- `server.py`에 새 함수 + 라우팅
- 또는 새 모듈 (`xxx_model.py`)
- 학술 출처 명시

### 6.3 새 UI 탭
- `index.html`에 새 `<div id="panel-xxx">`
- 새 `loadXxx()`, `renderXxx()` 함수
- 다른 탭에 영향 없도록 격리

## 7. 알려진 제약

- **단일 파일 구조**: 파일이 큼 (server.py 60KB+)
- **단일 서버 인스턴스**: 동시 사용자 1명 가정
- **메모리 사용**: parquet 캐시 약 500MB ~ 1GB
- **데이터 갱신**: 엑셀 수동 다운로드 필요
- **외부 API 의존**: yfinance 차단 시 일부 기능 제한
