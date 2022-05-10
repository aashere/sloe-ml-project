import pandas as pd

df = pd.read_csv('results/scores/scores.csv', encoding='utf8')
df = df[df['type'] != 'heart']
df = df.rename(columns={'ratio': 'kappa'})

print('Different F1 scores:')
f1_df = df[['n','kappa','type','baseline F1 score','SLOE F1 score']][df['baseline F1 score'] != df['SLOE F1 score']]
f1_df = f1_df.round(4)
f1_df.to_csv('results/tables/f1.csv', index=False, encoding='utf8')
print(f1_df)

print('SLOE is faster for:')
time_df = df[['n','kappa','type','baseline time (s)','SLOE time (s)']][df['baseline time (s)'] > df['SLOE time (s)']]
time_df = time_df.round(4)
time_df.to_csv('results/tables/time.csv', index=False, encoding='utf8')
print(time_df)

print('Highest alpha values:')
alpha_df = df[['n','kappa','type','alpha']].sort_values('alpha',ascending=False).groupby(['n','kappa'], as_index=False).first().rename(columns={'alpha':'max alpha'})
alpha_df = alpha_df.round(2)
alpha_df.to_csv('results/tables/alpha.csv', index=False, encoding='utf8')
print(alpha_df)