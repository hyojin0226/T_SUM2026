import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r'c:\Tsum2026\T_SUM2026\data\output\final_crisis_index_by_dong_2021.csv', encoding='utf-8-sig')
top = pd.read_csv(r'c:\Tsum2026\T_SUM2026\data\output\top_risk_dongs_2021.csv', encoding='utf-8-sig')

print('=== CCI 최종 분포 (IVI v2 적용) ===')
print(df[['IVI','CDI','RII','CCI']].describe().round(3))
print()

print('=== triple_high 11개 동 ===')
cols = ['자치구명','행정동명','IVI','CDI','RII','CCI','CCI_rank','risk_type']
print(top[cols].sort_values('CCI_rank').round(3).to_string(index=False))
print()

print('=== risk_type 분포 ===')
print(df['risk_type'].value_counts().to_string())
print()

# IVI v1 triple_high(7) vs v2 triple_high(11) 비교
old_7 = [('강남구','개포2동'),('노원구','상계3동'),('강북구','미아2동'),
          ('노원구','중계2동'),('중랑구','신내2동'),('노원구','상계2동'),('광진구','군자동')]
old_names = {f"{g}_{d}" for g,d in old_7}
new_11_names = set((top['자치구명'] + '_' + top['행정동명']).values)

newly_entered = new_11_names - old_names
exited = old_names - new_11_names

print('=== IVI v1→v2 변화로 신규 진입한 동 (triple_high 신규) ===')
for key in newly_entered:
    row = top[(top['자치구명']+'_'+top['행정동명'])==key]
    if len(row):
        r = row.iloc[0]
        print(f"  {r['자치구명']} {r['행정동명']} — IVI={r['IVI']:.3f} CDI={r['CDI']:.3f} RII={r['RII']:.3f} CCI={r['CCI']:.3f}")
    else:
        gu, dong = key.split('_', 1)
        row2 = df[(df['자치구명']==gu) & (df['행정동명']==dong)]
        if len(row2):
            r = row2.iloc[0]
            print(f"  {r['자치구명']} {r['행정동명']} — IVI={r['IVI']:.3f} CDI={r['CDI']:.3f} RII={r['RII']:.3f} CCI={r['CCI']:.3f} (triple_high 미포함 확인)")

print()
print('=== v1에서 triple_high였으나 v2에서 이탈한 동 ===')
for key in exited:
    gu, dong = key.split('_', 1)
    row = df[(df['자치구명']==gu) & (df['행정동명']==dong)]
    if len(row):
        r = row.iloc[0]
        print(f"  {r['자치구명']} {r['행정동명']} — IVI={r['IVI']:.3f} CDI={r['CDI']:.3f} RII={r['RII']:.3f} CCI={r['CCI']:.3f} triple={r['triple_high_flag']}")
