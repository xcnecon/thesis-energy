import pandas as pd
import numpy as np

fund = pd.read_csv('data/processed_data/fundamentals_panel.csv')
macro = pd.read_csv('data/processed_data/macro.csv')
df = fund.merge(macro, on='date', how='left')

mask = (df['revtq'] >= 5) & (df['date'] >= '1986-01-01') & (df['date'] < '2025-01-01')& (df['domestic_sales_share'] > 0.9)
df = df[mask]


df.to_csv('data/working_data/panel.csv', index=False)