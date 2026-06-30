김소연

"""
주가 엑셀 파일 → parquet 변환 스크립트
실행 위치: 주가 엑셀 2개 파일이 있는 폴더
"""
import pandas as pd
import os

OUT_DIR = 'kq_v4/data/cache'
os.makedirs(OUT_DIR, exist_ok=True)

files = {
    '2. 2014년 이후 주가 데이터 (일간) (파트 1 주가 관련).xlsx': 'price',
    '2. 2014년 이후 주가 데이터 (일간) (파트 2 그외 지표).xlsx': 'metric',
}

for f, kind in files.items():
    if not os.path.exists(f):
        print(f'❌ 파일 없음: {f}')
        continue

    print(f'\n처리 중: {f}')
    xl = pd.ExcelFile(f)
    for sn in xl.sheet_names:
        print(f'  [{sn}] 로딩...')
        df = pd.read_excel(f, sheet_name=sn, header=0, skiprows=[1, 2])
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        long_df = df.melt(id_vars=['date'], var_name='ticker', value_name='value')
        long_df = long_df.dropna(subset=['value'])
        long_df['ticker'] = long_df['ticker'].astype(str).str.lstrip('A').str.zfill(6)
        safe = sn.replace('(', '').replace(')', '').replace(' ', '_').replace('/', '_')
        out = f'{OUT_DIR}/{kind}_{safe}.parquet'
        long_df.to_parquet(out, compression='snappy', index=False)
        size_mb = os.path.getsize(out) / 1024 / 1024
        print(f'    ✓ {out} ({len(long_df):,}행, {size_mb:.1f}MB)')

print('\n✅ 완료!')
print(f'다음 단계: cd kq_v4 && python server.py')
