import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

v2 = pd.read_csv(r'c:\Tsum2026\T_SUM2026\data\processed\ivi_social_isolation_2021_v2.csv', encoding='utf-8-sig')
print('=== IVI v2 파일 검증 ===')
print(f'행 수: {len(v2)}, 열 수: {len(v2.columns)}')
print(f'결측치: {v2["IVI_v2"].isna().sum()}건')
print(f'IVI_v2 범위: {v2["IVI_v2"].min():.4f} ~ {v2["IVI_v2"].max():.4f}')
print(f'IVI v1  평균: {v2["IVI"].mean():.4f}  표준편차: {v2["IVI"].std():.4f}')
print(f'IVI v2  평균: {v2["IVI_v2"].mean():.4f}  표준편차: {v2["IVI_v2"].std():.4f}')
print(f'v1 vs v2 상관계수: {v2["IVI"].corr(v2["IVI_v2"]):.4f}')
print()

print('=== 구별 care_shortage_score (경로당 부족도 높은 순) ===')
care = v2.groupby('gu').agg(
    경로당_per_1000eld=('경로당_per_1000eld', 'first'),
    care_shortage_score=('care_shortage_score', 'first')
).sort_values('care_shortage_score', ascending=False)
print(care.round(3).to_string())
print()

print('=== IVI v2 상위 10개 동 ===')
cols = ['gu', 'dong', 'elderly_ratio_score', 'elderly_alone_score',
        'care_shortage_score', 'IVI', 'IVI_v2', 'IVI_v2_rank']
top10 = v2.nsmallest(10, 'IVI_v2_rank')[cols]
print(top10.round(3).to_string(index=False))
print()

print('=== triple_high 7개 동 IVI 변화 ===')
triple = [('강남구','개포2동'),('노원구','상계3동'),('강북구','미아2동'),
          ('노원구','중계2동'),('중랑구','신내2동'),('노원구','상계2동'),('광진구','군자동')]
th = pd.DataFrame(triple, columns=['gu','dong'])
th_detail = v2.merge(th, on=['gu','dong'])
print(th_detail[['gu','dong','IVI','IVI_v2','IVI_rank','IVI_v2_rank','care_shortage_score']].round(3).to_string(index=False))
