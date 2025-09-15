import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Purpose:
# - Build firm-year housing vs non-housing sales from segment data
# - Compute market shares and HHIs by fiscal year
# - Plot HHI alongside the number of active firms (unique gvkeys)
# - Persist tidy outputs under data/processed_data/

# Configuration
MIN_PLOT_YEAR = 1980

def plot_hhi_vs_firm_counts(hhi_by_fyear: pd.DataFrame, num_gvkeys_by_fyear: pd.DataFrame) -> None:
    """Plot HHI (left y-axis) and number of firms (right y-axis) using twin axes."""
    hhi_plot = hhi_by_fyear[hhi_by_fyear['fyear'] >= MIN_PLOT_YEAR]
    firms_plot = num_gvkeys_by_fyear[num_gvkeys_by_fyear['fyear'] >= MIN_PLOT_YEAR]

    fig, ax1 = plt.subplots()

    color_hhi = 'tab:blue'
    ax1.set_xlabel('Fiscal Year')
    ax1.set_ylabel('HHI (x10,000)', color=color_hhi)
    line1, = ax1.plot(hhi_plot['fyear'], hhi_plot['hhi_housing_10000'], color=color_hhi, label='HHI Housing')
    ax1.tick_params(axis='y', labelcolor=color_hhi)

    ax2 = ax1.twinx()
    color_gv = 'tab:orange'
    ax2.set_ylabel('Number of Firms', color=color_gv)
    line2, = ax2.plot(firms_plot['fyear'], firms_plot['num_gvkeys'], color=color_gv, label='Number of Firms')
    ax2.tick_params(axis='y', labelcolor=color_gv)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    fig.tight_layout()
    plt.show()

def save_outputs(hhi_by_fyear: pd.DataFrame, df_new: pd.DataFrame, num_gvkeys_by_fyear: pd.DataFrame) -> None:
    """Persist outputs used for analysis and visualization and print firm counts."""
    print(num_gvkeys_by_fyear)
    hhi_by_fyear.to_csv('data/processed_data/hhi_by_fyear.csv', index=False)
    df_new.to_csv('data/processed_data/segment_level_data.csv', index=False)
    num_gvkeys_by_fyear.to_csv('data/processed_data/num_gvkeys_by_fyear.csv', index=False)

df = pd.read_csv('data/raw_data/buz_seg.csv')  # segment-level sales
df['datadate'] = pd.to_datetime(df['datadate'])
df['srcdate'] = pd.to_datetime(df['srcdate'])


# Prefer business segments (BUSSEG) when available; otherwise use operating segments (OPSEG)
df = (
    df[df['stype'].isin(['BUSSEG','OPSEG'])]             # optional guard
      .assign(_prio=(df['stype']!='BUSSEG').astype(int)) # 0=BUSSEG, 1=OPSEG
      .sort_values(['gvkey','datadate','_prio','srcdate'], ascending=[True,True,True,False])
      .drop_duplicates(['gvkey','datadate', 'snms'], keep='first')
      .drop(columns=['_prio'])
)

# Map statement date to fiscal year (fyear) via lookup on (gvkey, datadate)
fyear_map = pd.read_csv('data/raw_data/fyear_map.csv', usecols=['gvkey', 'datadate', 'fyear'])
fyear_map['datadate'] = pd.to_datetime(fyear_map['datadate'])
df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
fyear_map['gvkey'] = fyear_map['gvkey'].astype(str).str.zfill(6)
fyear_map = fyear_map.drop_duplicates(subset=['gvkey', 'datadate'])
df = df.merge(fyear_map, on=['gvkey', 'datadate'], how='left')
df.reset_index(drop=True, inplace=True)

# Drop unused columns for this analysis
df.drop(columns=['srcdate','datadate', 'sid'], inplace=True)

mask = (df['sic'].isin([1311, 1321]))            # energy-related SICs
df = df[mask]

df.to_csv('segment_level_data.csv', index=False)

for names in df['snms'].unique().tolist():
    with open('output/segment_names.txt', 'a') as f:
        f.write(str(names) + '\n')
    f.close()
