import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load quarterly fundamentals
fund = pd.read_csv('data/raw_data/compustat/quarterly_fundamentals.csv')

# Normalize join keys
fund['gvkey'] = fund['gvkey'].astype(str).str.zfill(6)
if 'fyearq' in fund.columns:
    fund['fyearq'] = pd.to_numeric(fund['fyearq'], errors='coerce').astype('Int64')
if 'fqtr' in fund.columns:
    fund['fqtr'] = pd.to_numeric(fund['fqtr'], errors='coerce').astype('Int64')

# Compute calendar quarter end date from fyearq and fqtr
def _quarter_end_date(fiscal_year: pd.Series, fiscal_quarter: pd.Series) -> pd.Series:
    def _compute_single(y, q):
        if pd.isna(y) or pd.isna(q):
            return pd.NaT
        q_int = int(q)
        y_int = int(y)
        if q_int == 1:
            return pd.Timestamp(year=y_int, month=3, day=31)
        if q_int == 2:
            return pd.Timestamp(year=y_int, month=6, day=30)
        if q_int == 3:
            return pd.Timestamp(year=y_int, month=9, day=30)
        if q_int == 4:
            return pd.Timestamp(year=y_int, month=12, day=31)
        return pd.NaT
    return pd.Series((_compute_single(y, q) for y, q in zip(fiscal_year, fiscal_quarter)), index=fiscal_year.index)

if 'fyearq' in fund.columns and 'fqtr' in fund.columns:
    fund['date'] = _quarter_end_date(fund['fyearq'], fund['fqtr'])

# Load ENP share by firm-year
enp = pd.read_csv('data/processed_data/firms_with_enp_share_gt_0.8.csv')
enp['gvkey'] = enp['gvkey'].astype(str).str.zfill(6)
enp['fyear'] = pd.to_numeric(enp['fyear'], errors='coerce').astype('Int64')

# Load domestic share by firm-year
dom = pd.read_csv('data/processed_data/domestic_sales_share_by_firm.csv')
dom['gvkey'] = dom['gvkey'].astype(str).str.zfill(6)
dom['fyear'] = pd.to_numeric(dom['fyear'], errors='coerce').astype('Int64')

# Derive fiscal year in fundamentals to merge on: use fyearq when available
fund['fyear'] = fund.get('fyear', pd.Series(index=fund.index, dtype='Int64'))
if fund['fyear'].isna().all() and 'fyearq' in fund.columns:
    fund['fyear'] = fund['fyearq']
else:
    fund['fyear'] = pd.to_numeric(fund['fyear'], errors='coerce').astype('Int64')

# Drop raw fiscal quarter columns now that date is derived
for col in ['fqtr', 'fyearq']:
    if col in fund.columns:
        fund = fund.drop(columns=col)

# Merge ENP share and fillna with 1 when not available
fund = fund.merge(enp[['gvkey', 'fyear', 'enp_share']], on=['gvkey', 'fyear'], how='left')
fund['enp_share'] = fund['enp_share'].fillna(1.0)

# Merge domestic share and fillna with 1 when not available
fund = fund.merge(dom[['gvkey', 'fyear', 'domestic_sales_share']], on=['gvkey', 'fyear'], how='left')
fund['domestic_sales_share'] = fund['domestic_sales_share'].fillna(1.0)

mask = (fund['enp_share'] > 0.8) & (fund['domestic_sales_share'] > 0.8)
fund = fund[mask]

fund.to_csv('data/processed_data/fundamental_segment_mapping.csv', index=False)

# Compute HHI by quarter (date) using revtq and export
if 'revtq' in fund.columns and 'date' in fund.columns:
    fund['revtq'] = pd.to_numeric(fund['revtq'], errors='coerce')
    rev_by_firm_q = (
        fund[['date', 'gvkey', 'revtq']]
            .dropna(subset=['revtq'])
            .groupby(['date', 'gvkey'], as_index=False)['revtq']
            .sum()
    )
    totals_by_date = (
        rev_by_firm_q.groupby('date', as_index=False)['revtq']
            .sum()
            .rename(columns={'revtq': 'total_revenue'})
    )
    hhi_df = rev_by_firm_q.merge(totals_by_date, on='date', how='left')
    hhi_df = hhi_df[hhi_df['total_revenue'] > 0]
    hhi_df['revenue_share'] = hhi_df['revtq'] / hhi_df['total_revenue']
    hhi_df['share_sq'] = hhi_df['revenue_share'] ** 2
    hhi_by_date = (
        hhi_df.groupby('date', as_index=False)['share_sq']
            .sum()
            .rename(columns={'share_sq': 'hhi'})
    )
    hhi_by_date['hhi_10000'] = (hhi_by_date['hhi'] * 10000).round(2)
    hhi_by_date.to_csv('data/processed_data/hhi_by_quarter_filtered.csv', index=False)

# Plot distributions of ENP and domestic shares for filtered data
if not fund.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].hist(fund['enp_share'].dropna(), bins=30, range=(0, 1), color='#4C78A8', edgecolor='white')
    axes[0].set_title('ENP Share Distribution for Firm-Quarters with ENP Share > 0.8')
    axes[0].set_xlabel('ENP Share')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0.8, 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(fund['domestic_sales_share'].dropna(), bins=30, range=(0, 1), color='#F58518', edgecolor='white')
    axes[1].set_title('Domestic Sales Share Distribution for Firm-Quarters with Domestic Sales Share > 0.8')
    axes[1].set_xlabel('Domestic Sales Share')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0.8, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/fund_enp_domestic_share_distributions.png')
    plt.show()


# Plot number of firms per quarter (filtered fund)
if not fund.empty and 'date' in fund.columns:
    firms_per_quarter = (
        fund.groupby('date')['gvkey']
            .nunique()
            .reset_index(name='num_firms')
            .sort_values('date')
    )
    # Restrict to 1984–2024
    firms_per_quarter = firms_per_quarter[(firms_per_quarter['date'].dt.year >= 1982) & (firms_per_quarter['date'].dt.year <= 2024)]
    plt.figure(figsize=(14, 5))
    plt.plot(
        firms_per_quarter['date'],
        firms_per_quarter['num_firms'],
        marker='o',
        linewidth=1.5,
        label='Firms per Quarter'
    )
    plt.title('Number of Firms per Quarter (Filtered)')
    plt.xlabel('Quarter End Date')
    plt.ylabel('Number of Firms')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/firms_per_quarter_filtered.png')
    plt.show()

    # Plot HHI (left axis) and number of firms (right axis) over time
    try:
        # Align HHI to the same filtered date range
        if 'hhi_by_date' in locals():
            hhi_plot_df = hhi_by_date.copy()
        else:
            # Recompute HHI if not available (robustness)
            tmp_rev = (
                fund[['date', 'gvkey', 'revtq']]
                    .dropna(subset=['revtq'])
                    .groupby(['date', 'gvkey'], as_index=False)['revtq']
                    .sum()
            )
            tmp_tot = (
                tmp_rev.groupby('date', as_index=False)['revtq']
                    .sum()
                    .rename(columns={'revtq': 'total_revenue'})
            )
            tmp = tmp_rev.merge(tmp_tot, on='date', how='left')
            tmp = tmp[tmp['total_revenue'] > 0]
            tmp['share_sq'] = (tmp['revtq'] / tmp['total_revenue']) ** 2
            hhi_plot_df = (
                tmp.groupby('date', as_index=False)['share_sq']
                   .sum()
                   .rename(columns={'share_sq': 'hhi'})
            )
            hhi_plot_df['hhi_10000'] = (hhi_plot_df['hhi'] * 10000).round(2)

        # Filter HHI to the same time window
        hhi_plot_df = hhi_plot_df[(hhi_plot_df['date'].dt.year >= 1982) & (hhi_plot_df['date'].dt.year <= 2024)]

        merged_plot_df = firms_per_quarter.merge(hhi_plot_df[['date', 'hhi_10000']], on='date', how='inner')
        merged_plot_df = merged_plot_df.sort_values('date')

        fig, ax1 = plt.subplots(figsize=(14, 5))
        color_hhi = '#2ca02c'
        color_firms = '#1f77b4'

        ax1.plot(merged_plot_df['date'], merged_plot_df['hhi_10000'], color=color_hhi, linewidth=1.8, label='HHI (x10,000)')
        ax1.set_xlabel('Quarter End Date')
        ax1.set_ylabel('HHI (x10,000)', color=color_hhi)
        ax1.tick_params(axis='y', labelcolor=color_hhi)

        ax2 = ax1.twinx()
        ax2.plot(merged_plot_df['date'], merged_plot_df['num_firms'], color=color_firms, linewidth=1.8, label='Number of Firms')
        ax2.set_ylabel('Number of Firms', color=color_firms)
        ax2.tick_params(axis='y', labelcolor=color_firms)

        fig.suptitle('HHI and Number of Firms per Quarter (Filtered)')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig('output/hhi_and_firms_per_quarter_filtered.png')
        plt.show()
    except Exception as e:
        # Gracefully skip combined plot if any requirement is missing
        print(f"Skipping combined HHI/firms plot due to: {e}")

