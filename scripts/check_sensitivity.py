import pandas as pd, numpy as np, sys
from scipy.stats import kendalltau
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv(r'c:\Tsum2026\T_SUM2026\data\output\final_crisis_index_by_dong_2021.csv', encoding='utf-8-sig')
N = len(df)
TOP_10PCT = int(N * 0.10)

SCENARIOS = {
    'Main': {'IVI': 0.40, 'CDI': 0.25, 'RII': 0.35},
    'S1':   {'IVI': 1/3,  'CDI': 1/3,  'RII': 1/3},
    'S2':   {'IVI': 0.50, 'CDI': 0.25, 'RII': 0.25},
    'S3':   {'IVI': 0.30, 'CDI': 0.20, 'RII': 0.50},
}

result = df[['자치구명','행정동명','IVI','CDI','RII','triple_high_flag']].copy()
result['dong_label'] = result['자치구명'].str.replace('구','',regex=False) + ' ' + result['행정동명']

for k, w in SCENARIOS.items():
    result[f'CCI_{k}']  = df['IVI']*w['IVI'] + df['CDI']*w['CDI'] + df['RII']*w['RII']
    result[f'rank_{k}'] = result[f'CCI_{k}'].rank(ascending=False, method='min').astype(int)

rank_cols = [f'rank_{k}' for k in SCENARIOS]
main_triple = set(result[result['triple_high_flag']==1]['dong_label'].values)

print('=== 시나리오별 상위 30위 중 triple_high 유지율 ===')
for k, w in SCENARIOS.items():
    top30 = set(result[result[f'rank_{k}']<=30]['dong_label'])
    kept  = main_triple & top30
    lost  = main_triple - top30
    print(f"  {k} (IVI={w['IVI']:.2f}/CDI={w['CDI']:.2f}/RII={w['RII']:.2f}): "
          f"{len(kept)}/{len(main_triple)}개 유지 "
          f"| 이탈: {', '.join(sorted(lost)) if lost else '없음'}")

print()
print('=== Kendall τ (Main 기준) ===')
for k in ['S1','S2','S3']:
    tau, _ = kendalltau(result['rank_Main'], result[f'rank_{k}'])
    print(f'  Main vs {k}: τ = {tau:.4f}')

print()
invariant = set.intersection(*[set(result[result[f'rank_{k}']<=TOP_10PCT]['dong_label']) for k in SCENARIOS])
print(f'=== 불변의 위험 지역 (전체 {TOP_10PCT}위 이내 고정) — {len(invariant)}개 동 ===')
inv_df = result[result['dong_label'].isin(invariant)].copy().sort_values('rank_Main')
inv_df['변동폭'] = inv_df[rank_cols].max(axis=1) - inv_df[rank_cols].min(axis=1)
print(inv_df[['자치구명','행정동명','IVI','CDI','RII']+rank_cols+['변동폭','triple_high_flag']].round(3).to_string(index=False))

print()
print('=== triple_high 순위 안정성 상세 ===')
tri = result[result['triple_high_flag']==1].copy().sort_values('rank_Main')
tri['최고순위'] = tri[rank_cols].min(axis=1)
tri['최저순위'] = tri[rank_cols].max(axis=1)
tri['변동폭'] = tri['최저순위'] - tri['최고순위']
tri['항상_10%이내'] = (tri[rank_cols] <= TOP_10PCT).all(axis=1)
print(tri[['자치구명','행정동명']+rank_cols+['최고순위','최저순위','변동폭','항상_10%이내']].round(0).to_string(index=False))
print()
print(f'항상 상위10% 유지: {tri["항상_10%이내"].sum()}/{len(tri)}개')
print(f'평균 변동폭: {tri["변동폭"].mean():.1f}위')
