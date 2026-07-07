# ECOS 데이터 수집 가이드 (data/macro 구축용)

목적: `kq_tool` 국면 레이어에 필요한 매크로/시장 시리즈를 한국은행 ECOS Open API에서 수집해 `data/macro/<key>.csv` 형식으로 저장한다.

## 0. 준비

1. [ECOS Open API](https://ecos.bok.or.kr/api/)에서 API 인증키를 발급한다.
2. PowerShell에서 환경변수로 설정한다.

```powershell
$env:ECOS_API_KEY = "발급받은_API_KEY"
```

또는 실행 때 직접 넘긴다.

```powershell
python scripts\fetch_ecos.py --api-key 발급받은_API_KEY --discover 817Y002
```

3. `--discover`로 항목코드를 확인한 뒤 [scripts/fetch_ecos.py](../scripts/fetch_ecos.py)의 `SERIES`에서 `None` 값을 채운다.
4. 수집을 실행한다.

```powershell
python scripts\fetch_ecos.py
```

## 1. 통계표 코드

| 산출 파일 | ECOS 통계표코드 | 주기 | 항목코드 | 비고 |
|---|---:|---:|---|---|
| `base_rate` | `722Y001` | M | `0101000` | 한국은행 기준금리 |
| `cpi_index` | `901Y009` | M | `0` | 소비자물가지수 총지수, `cpi_yoy` 파생 |
| `treasury_3y` | `817Y002` | D | 확인 필요 | 국고채 3년 |
| `treasury_10y` | `817Y002` | D | 확인 필요 | 국고채 10년 |
| `yield_spread_10y_3y` | 파생 | - | - | `treasury_10y - treasury_3y` |
| `corp_aa_3y` | `817Y002` | D | 확인 필요 | 회사채 3년 AA- |
| `credit_spread` | 파생 | - | - | `corp_aa_3y - treasury_3y` |
| `usdkrw` | `731Y001` | D | 확인 필요 | 원/달러 환율 |
| `kospi` | `802Y001` | D | 확인 필요 | KOSPI 지수 |
| `gdp_qoq` | `200Y*` | Q | 확인 필요 | 실질 GDP 전기비, 계절조정 여부 확인 |

항목코드는 통계표 개편으로 바뀔 수 있다. 블로그나 예전 문서의 값을 그대로 믿지 말고 `StatisticItemList` 응답을 기준으로 확정한다.

## 2. 항목코드 확인 명령

```powershell
python scripts\fetch_ecos.py --discover 817Y002
python scripts\fetch_ecos.py --discover 731Y001
python scripts\fetch_ecos.py --discover 802Y001
python scripts\fetch_ecos.py --discover 200Y101
python scripts\fetch_ecos.py --discover 200Y104
```

출력된 항목명에서 `국고채(3년)`, `국고채(10년)`, `회사채(3년, AA-)`, `원/달러`, `KOSPI`, `실질 GDP 전기비(계절조정)`에 해당하는 item code를 `SERIES`에 채운다.

## 3. 수집 후 체크리스트

1. `python tests\validation_macro_pit.py`로 발표 지연 lag 검증을 돌린다.
2. GDP 값을 육안으로 확인한다. UI에서 보였던 `-5.81%` 같은 값이 다시 나오면 전기비/전년동기비/연율화/계절조정 혼동 가능성이 높다.
3. CPI YoY가 상식 범위인지 확인한다. 대략 0~6% 구간과 2022년 고물가 구간이 보여야 한다.
4. 라벨을 생성한 뒤 평균 지속기간과 전환 횟수를 본다. 목표 감각은 평균 지속 3~4분기, 20년 기준 전환 10~15회다.

## 4. 주의

- ECOS 일별 데이터는 영업일만 존재한다. `macro_data.py` 로더는 월말 리샘플링 시 마지막 관측값을 사용한다.
- 자산군 수익률, 예를 들어 주식/채권/금/현금 ETF 월수익률은 ECOS가 아니라 KRX, yfinance, FinanceDataReader 등 시세 데이터에서 받아 `data/assets/monthly_returns.csv`로 준비한다.
- API 키는 커밋하지 않는다. 가능하면 `ECOS_API_KEY` 환경변수를 사용한다.
