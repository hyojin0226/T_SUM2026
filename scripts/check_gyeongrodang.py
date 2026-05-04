import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(
    r'c:\Tsum2026\T_SUM2026\data\raw\IVI_social_isolation\서울시 경로당 정보.csv',
    encoding='cp949'
)

gu_count = df.groupby('시군구명').size().reset_index(name='경로당수')
print('=== 구별 경로당 수 ===')
print(gu_count.sort_values('경로당수', ascending=False).to_string(index=False))
print(f'\n총 경로당: {gu_count["경로당수"].sum()}개, {len(gu_count)}개 구 커버')

# 주소에서 법정동 추출 가능 여부
import re
def extract_dong(addr):
    m = re.search(r'\(([가-힣0-9·]+?)[,\s\)]', str(addr))
    return m.group(1) if m else None

df['법정동_추출'] = df['주소(도로명)'].apply(extract_dong)
has_dong = df['법정동_추출'].notna().sum()
print(f'\n법정동 추출 가능: {has_dong}/{len(df)} ({has_dong/len(df)*100:.1f}%)')
print('법정동 추출 샘플:')
print(df[['시군구명','주소(도로명)','법정동_추출']].head(10).to_string(index=False))
