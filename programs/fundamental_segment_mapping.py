import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seg = pd.read_csv('data/processed_data/segment_level_data.csv')
df = pd.read_csv('data/raw_data/quarterly_fundamentals.csv')



# Normalize keys for merge
seg['gvkey'] = seg['gvkey'].astype(str).str.zfill(6)
df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
seg['fyear'] = pd.to_numeric(seg['fyear'], errors='coerce').astype('Int64')
df['fyearq'] = pd.to_numeric(df['fyearq'], errors='coerce').astype('Int64')
df['fqtr'] = pd.to_numeric(df['fqtr'], errors='coerce').astype('Int64')



# Mark whether a firm-fiscal-year appears in segment data
seg_keys = seg[['gvkey', 'fyear']].drop_duplicates()
seg_keys['in_seg'] = 1
df = df.merge(seg_keys, left_on=['gvkey', 'fyearq'], right_on=['gvkey', 'fyear'], how='left')
df['in_seg'] = df['in_seg'].fillna(0).astype(int)
df.drop(columns=['fyear'], inplace=True)

# Fundamentals enrichments
df['datadate'] = pd.to_datetime(df['datadate'])
_month_map = {1: 3, 2: 6, 3: 9, 4: 12}
_month_series = df['fqtr'].map(_month_map)
df['date'] = pd.to_datetime({'year': df['fyearq'], 'month': _month_series, 'day': 1}) + pd.offsets.MonthEnd(0)
df['gross_margin'] = np.where(df['revtq'] != 0, (df['revtq'] - df['cogsq']) / df['revtq'], np.nan)
df['aggregate_revenue'] = df.groupby(['date'])['revtq'].transform('sum')
df['aggregate_costs'] = df.groupby(['date'])['cogsq'].transform('sum')
df['aggregate_capital'] = df.groupby(['date'])['icaptq'].transform('sum')
df['aggregate_gross_margin'] = (df['aggregate_revenue'] - df['aggregate_costs']) / df['aggregate_revenue']
df['operating_margin'] = np.where(df['revtq'] != 0, df['oiadpq'] / df['revtq'], np.nan)
df['aggregate_operating_income'] = df.groupby(['date'])['oiadpq'].transform('sum')
df['aggregate_operating_margin'] = df['aggregate_operating_income'] / df['aggregate_revenue']

df['fyear_match'] = df['fyearq'] - 1
df = df.merge(seg, left_on=['gvkey', 'fyear_match'], right_on=['gvkey', 'fyear'], how='left')

hhi = pd.read_csv('data/processed_data/hhi_by_fyear.csv')
df = df.merge(hhi, left_on=['fyear_match'], right_on=['fyear'], how='left')

## date computed above from (fyearq, fqtr)

df = df[['gvkey', 'date', 'fyearq', 'revtq', 'cogsq', 'gross_margin', 'operating_margin', 'housing_share', 'aggregate_gross_margin', 'aggregate_operating_margin', 'hhi_housing_10000' ]].reset_index(drop=True)

# Delta columns: current quarter minus value 4 quarters ago
df = df.sort_values(['gvkey', 'date']).reset_index(drop=True)
df['delta_gross_margin'] = df['gross_margin'] - df.groupby('gvkey')['gross_margin'].shift(4)
agg = (
    df[['date', 'aggregate_gross_margin']]
      .drop_duplicates()
      .sort_values('date')
)
agg['delta_aggregate_gross_margin'] = agg['aggregate_gross_margin'] - agg['aggregate_gross_margin'].shift(4)
df = df.merge(agg[['date', 'delta_aggregate_gross_margin']], on='date', how='left')

df['delta_operating_margin'] = df['operating_margin'] - df.groupby('gvkey')['operating_margin'].shift(4)

agg = (
    df[['date', 'aggregate_operating_margin']]
      .drop_duplicates()
      .sort_values('date')
)
agg['delta_aggregate_operating_margin'] = agg['aggregate_operating_margin'] - agg['aggregate_operating_margin'].shift(4)
df = df.merge(agg[['date', 'delta_aggregate_operating_margin']], on='date', how='left')

df['market_power'] = df['housing_share'] * df['hhi_housing_10000']

df = df[(df['fyearq'] >= 1980) & (df['fyearq'] <= 2024)]
df.dropna(inplace=True)

# Plot aggregate margins against quarter-end date
df_plot = (
    df[['date', 'aggregate_gross_margin', 'aggregate_operating_margin']]
      .drop_duplicates()
      .sort_values('date')
)

plt.figure()
plt.plot(df_plot['date'], df_plot['aggregate_gross_margin'], label='Aggregate Gross Margin')
plt.plot(df_plot['date'], df_plot['aggregate_operating_margin'], label='Aggregate Operating Margin')
plt.xlabel('Quarter End Date')
plt.ylabel('Margin')
plt.title('Aggregate Margins over Time')
plt.legend()
plt.tight_layout()
plt.show()

df.to_csv('data/processed_data/fundamental_segment_mapping.csv', index=False)

