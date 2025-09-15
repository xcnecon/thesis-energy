"""
Compute domestic vs. foreign sales shares for homebuilding-related codes and plot.

Steps:
- Load GEOSEG data; restrict to homebuilding SIC/NAICS
- Deduplicate within (gvkey, datadate, geotp) by latest srcdate
- Merge fiscal year (fyear) from (gvkey, datadate) map
- Aggregate sales by fiscal year and geography (2=domestic, 3=foreign)
- Pivot to wide columns, compute shares, plot, and export CSV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Configuration
MIN_YEAR = 1977


def load_geo_data() -> pd.DataFrame:
    """Load geo segment data and parse required dates."""
    df_local = pd.read_csv('data/raw_data/geo_seg.csv')
    df_local['datadate'] = pd.to_datetime(df_local['datadate'])
    df_local['srcdate'] = pd.to_datetime(df_local['srcdate'])
    return df_local


def filter_scope_and_dedupe(df_local: pd.DataFrame) -> pd.DataFrame:
    """Restrict to relevant SIC/NAICS and keep the latest srcdate per (gvkey, datadate, geotp)."""
    mask = (df_local['sic'].isin([1311, 1321]))
    df_local = df_local[mask]
    df_local = (
        df_local
          .sort_values(['gvkey', 'datadate', 'geotp', 'srcdate'])
          .drop_duplicates(['gvkey', 'datadate', 'geotp'], keep='last')
    )
    return df_local


def map_to_fiscal_year(df_local: pd.DataFrame) -> pd.DataFrame:
    """Map datadate into fyear via (gvkey, datadate) lookup, with normalized gvkeys."""
    fyear_map = pd.read_csv('data/raw_data/fyear_map.csv', usecols=['gvkey', 'datadate', 'fyear'])
    fyear_map['datadate'] = pd.to_datetime(fyear_map['datadate'])

    df_local['gvkey'] = df_local['gvkey'].astype(str).str.zfill(6)
    fyear_map['gvkey'] = fyear_map['gvkey'].astype(str).str.zfill(6)

    fyear_map = fyear_map.drop_duplicates(subset=['gvkey', 'datadate'])
    df_local = df_local.merge(fyear_map, on=['gvkey', 'datadate'], how='left')
    df_local = df_local.dropna(subset=['fyear'])
    df_local['fyear'] = df_local['fyear'].astype(int)
    return df_local


def compute_geo_sales_shares(df_local: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to fiscal-year domestic/foreign sales and compute shares (trim to MIN_YEAR)."""
    df_local = df_local[df_local['geotp'].isin([2, 3])]
    geo_sales_df = (
        df_local.groupby(['fyear', 'geotp'])['sales']
          .sum()
          .unstack('geotp')
          .reset_index()
          .rename(columns={2: 'domestic_sales', 3: 'foreign_sales'})
          .fillna(0)
    )
    geo_sales_df['total_sales'] = geo_sales_df['domestic_sales'] + geo_sales_df['foreign_sales']
    geo_sales_df['domestic_sales_share'] = geo_sales_df['domestic_sales'] / geo_sales_df['total_sales']
    geo_sales_df['foreign_sales_share'] = geo_sales_df['foreign_sales'] / geo_sales_df['total_sales']
    geo_sales_df = geo_sales_df[geo_sales_df['fyear'] >= MIN_YEAR]
    return geo_sales_df


def plot_domestic_share(geo_sales_df: pd.DataFrame) -> None:
    """Line plot with title and axis labels for domestic sales share over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        geo_sales_df['fyear'],
        geo_sales_df['domestic_sales_share'],
        label='Domestic Sales Share',
        linewidth=2
    )
    plt.title('Domestic Sales Share Over Time')
    plt.xlabel('Fiscal Year')
    plt.ylabel('Domestic Sales Share (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/geo_sales_share_plot.png')
    plt.show()


def save_geo_sales(geo_sales_df: pd.DataFrame) -> None:
    """Export processed results to CSV."""
    geo_sales_df.to_csv('output/geo_sales_share_plot.csv', index=False)

def main() -> None:
    df_local = load_geo_data()
    df_local = filter_scope_and_dedupe(df_local)
    df_local = map_to_fiscal_year(df_local)
    geo_sales_df = compute_geo_sales_shares(df_local)
    plot_domestic_share(geo_sales_df)
    save_geo_sales(geo_sales_df)


if __name__ == '__main__':
    main()

