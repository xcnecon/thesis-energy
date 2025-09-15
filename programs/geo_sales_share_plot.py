import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_csv('data/raw_data/geo_seg.csv')
# Normalize keys and parse dates
df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
df['datadate'] = pd.to_datetime(df['datadate'])
df['srcdate'] = pd.to_datetime(df['srcdate'])

# Pull fyear and loc from fyear_map based on (gvkey, datadate)
fyear_map = pd.read_csv('data/raw_data/fyear_map.csv', usecols=['gvkey', 'datadate', 'fyear', 'loc'])
fyear_map['gvkey'] = fyear_map['gvkey'].astype(str).str.zfill(6)
fyear_map['datadate'] = pd.to_datetime(fyear_map['datadate'])
df = df.merge(fyear_map, on=['gvkey', 'datadate'], how='left')

# If multiple srcdate report the same (gvkey, datadate, geotp), keep the last (latest srcdate)
df = (
    df.sort_values(['gvkey', 'datadate', 'geotp', 'srcdate'])
      .drop_duplicates(['gvkey', 'datadate', 'geotp'], keep='last')
)

mask = df['sic'].isin([1311, 1321])
df = df[mask]

# Drop srcdate after deduplication
df = df.drop(columns=['srcdate','stype', 'sid'])

US_LIST = [
    "United States",
    "U.S.",
    "USA",
    "US",
    "United States of America",
    "United states",
    "United States,Domestic",
    "United States & Other",
    "United States,Other Foreign",
    "United States, Europe, other regions",
    "USA and Other",
    "Asia,United States",
    "Asia,Great Britain,United States",
    "Africa,Great Britain,United States",
    "Corporate - United States",
    "U.S. Gulf of Mexico",
    "U.S Gulf of Mexico",
    "North America",
    "North America,Domestic",
    "South America,North America,Domestic",
    "Canada & United States"
]

# Flag domestic (USA) sales per row based on firm location and geography/segment name
df['is_domestic_sale'] = np.where(
    df['loc'] == 'USA',
    df['geotp'] == 2,
    df['snms'].isin(US_LIST)
)

# Compute firm-year total sales and domestic (USA) sales
firm_year_totals = (
    df.groupby(['gvkey', 'fyear'])['sales']
      .sum()
      .reset_index(name='total_sales')
)
firm_year_domestic = (
    df[df['is_domestic_sale']]
      .groupby(['gvkey', 'fyear'])['sales']
      .sum()
      .reset_index(name='domestic_sales')
)

# Merge and compute domestic sales share
domestic_share_by_firm = firm_year_totals.merge(firm_year_domestic, on=['gvkey', 'fyear'], how='left')
domestic_share_by_firm['domestic_sales'] = domestic_share_by_firm['domestic_sales'].fillna(0)
DomShareDenom = domestic_share_by_firm['total_sales']
domestic_share_by_firm['domestic_sales_share'] = np.where(
    DomShareDenom > 0,
    domestic_share_by_firm['domestic_sales'] / DomShareDenom,
    np.nan
)

# Export to processed data
domestic_share_by_firm.to_csv('data/processed_data/domestic_sales_share_by_firm.csv', index=False)

# Aggregate US (domestic) sales share by fiscal year and plot
agg_totals = (
    df.groupby('fyear')['sales']
      .sum()
      .reset_index(name='total_sales')
)
agg_usa = (
    df[df['is_domestic_sale']]
      .groupby('fyear')['sales']
      .sum()
      .reset_index(name='usa_sales')
)
agg = agg_totals.merge(agg_usa, on='fyear', how='left')
agg['usa_sales'] = agg['usa_sales'].fillna(0)
agg['usa_sales_share'] = np.where(
    agg['total_sales'] > 0,
    agg['usa_sales'] / agg['total_sales'],
    np.nan
)
agg = agg.sort_values('fyear')
agg = agg[(agg['fyear'] >= 1984) & (agg['fyear'] <= 2024)]

plt.figure(figsize=(10, 6))
plt.plot(agg['fyear'], agg['usa_sales_share'], label='US Sales Share', linewidth=2)
plt.title('Aggregate US Sales Share Over Time')
plt.xlabel('Fiscal Year')
plt.ylabel('US Sales Share (%)')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
plt.xlim(1984, 2024)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('output/geo_sales_share_plot.png')
plt.show()