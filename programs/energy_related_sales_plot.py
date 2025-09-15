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

def load_segment_data() -> pd.DataFrame:
    """Load and prepare segment-level data with fiscal years and essential columns."""
    df = pd.read_csv('data/raw_data/buz_seg.csv')
    df['datadate'] = pd.to_datetime(df['datadate'])
    df['srcdate'] = pd.to_datetime(df['srcdate'])

    df = (
        df[df['stype'].isin(['BUSSEG','OPSEG'])]
          .assign(_prio=(df['stype']!='BUSSEG').astype(int))
          .sort_values(['gvkey','datadate','_prio','srcdate'], ascending=[True,True,True,False])
          .drop_duplicates(['gvkey','datadate', 'snms'], keep='first')
          .drop(columns=['_prio'])
    )

    fyear_map = pd.read_csv('data/raw_data/fyear_map.csv', usecols=['gvkey', 'datadate', 'fyear'])
    fyear_map['datadate'] = pd.to_datetime(fyear_map['datadate'])
    df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
    fyear_map['gvkey'] = fyear_map['gvkey'].astype(str).str.zfill(6)
    fyear_map = fyear_map.drop_duplicates(subset=['gvkey', 'datadate'])
    df = df.merge(fyear_map, on=['gvkey', 'datadate'], how='left')
    df.reset_index(drop=True, inplace=True)

    df.drop(columns=['srcdate','datadate', 'sid'], inplace=True)
    df = df[df['sic'].isin([1311, 1321])]
    return df

def load_negative_list() -> list:
    neg = pd.read_csv('output/segment_flags.csv')
    neg = neg[neg['is_non_e_and_p'] == True]
    return neg['segment'].unique().tolist()

def build_firm_year_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to firm-year wide with related/non-related sales and clamped enp_share."""
    df_groupby = df.groupby(['gvkey', 'fyear','is_related'])['sales'].sum()
    df_wide = (
        df_groupby
          .unstack('is_related')
          .rename(columns={True: 'related', False: 'non_related'})
          .reset_index()
    )
    df_wide[['related','non_related']] = df_wide[['related','non_related']].fillna(0)
    df_wide['total_sales'] = df_wide['related'] + df_wide['non_related']
    share = np.where(df_wide['total_sales'] > 0, df_wide['related'] / df_wide['total_sales'], np.nan)
    df_wide['enp_share'] = pd.Series(share).clip(lower=0, upper=1)
    return df_wide

def plot_enp_share_distribution(df_wide: pd.DataFrame) -> None:
    plt.hist(df_wide['enp_share'].dropna(), bins=100)
    plt.xlabel('E&P Related Sales Share')
    plt.ylabel('Frequency')
    plt.title('Distribution of E&P Related Sales Share')
    plt.savefig('output/enp_share_distribution.png')
    plt.show()

def compute_aggregates_and_counts(df_wide: pd.DataFrame, threshold: float = 0.8) -> tuple:
    """Compute overall and filtered aggregates plus counts; returns (combined_plot, df_filtered, total, kept)."""
    # Overall aggregates per year
    agg_year = (
        df_wide.groupby('fyear')
          .agg(aggregate_enp_sales=('related', 'sum'), aggregate_sales=('total_sales', 'sum'))
          .reset_index()
          .sort_values('fyear')
    )
    agg_share = np.where(agg_year['aggregate_sales'] > 0, agg_year['aggregate_enp_sales'] / agg_year['aggregate_sales'], np.nan)
    agg_year['aggregate_enp_share'] = pd.Series(agg_share).clip(lower=0, upper=1)

    # Filtered firm-years and aggregates
    df_filtered = df_wide[df_wide['enp_share'] > threshold].copy()
    total = len(df_wide)
    kept = len(df_filtered)

    filtered_year = (
        df_filtered.groupby('fyear')
          .agg(filtered_enp_sales=('related', 'sum'), filtered_total_sales=('total_sales', 'sum'))
          .reset_index()
          .sort_values('fyear')
    )
    filt_share = np.where(filtered_year['filtered_total_sales'] > 0, filtered_year['filtered_enp_sales'] / filtered_year['filtered_total_sales'], np.nan)
    filtered_year['filtered_enp_share'] = pd.Series(filt_share).clip(lower=0, upper=1)

    combined = agg_year.merge(filtered_year[['fyear', 'filtered_enp_share']], on='fyear', how='left')
    combined_plot = combined[combined['fyear'] >= MIN_PLOT_YEAR].copy()
    return combined_plot, df_filtered, total, kept

def plot_aggregate_enp_share_both(combined_plot: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(combined_plot['fyear'], combined_plot['aggregate_enp_share'], marker='o', color='tab:blue', label='Aggregate (all firms)')
    ax.plot(combined_plot['fyear'], combined_plot['filtered_enp_share'], marker='o', color='tab:green', label='Aggregate (enp_share > 0.8)')
    ax.set_xlabel('Fiscal Year')
    ax.set_ylabel('Aggregate E&P Sales Share')
    ax.set_title('Aggregate E&P Sales Share by Fiscal Year')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    plt.savefig('output/aggregate_enp_share_both_by_fyear.png', dpi=200)
    plt.show()

def save_filtered_firms(df_filtered: pd.DataFrame) -> None:
    df_final = df_filtered[['gvkey', 'fyear', 'enp_share']]
    df_final.to_csv('output/firms_with_enp_share_gt_0.8.csv', index=False)

def main() -> None:
    df = load_segment_data()
    negative_list = load_negative_list()
    df['is_related'] = ~df['snms'].isin(negative_list)

    df_wide = build_firm_year_wide(df)
    plot_enp_share_distribution(df_wide)

    combined_plot, df_filtered, total, kept = compute_aggregates_and_counts(df_wide, threshold=0.8)
    print(f"Kept (enp_share > 0.8): {kept} / {total}")
    print(f"Dropped: {total - kept}")

    plot_aggregate_enp_share_both(combined_plot)
    save_filtered_firms(df_filtered)

if __name__ == '__main__':
    main()