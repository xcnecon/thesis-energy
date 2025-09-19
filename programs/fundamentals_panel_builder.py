import pandas as pd
import numpy as np


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
    "Canada & United States",
]


def load_negative_list(path: str = 'data/processed_data/segment_flags.csv') -> list:
    neg = pd.read_csv(path)
    neg = neg[neg['is_non_e_and_p'] == True]
    return neg['segment'].unique().tolist()


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


def compute_enp_share_by_firm_year(buz_df: pd.DataFrame, fyear_map_df: pd.DataFrame, flags_path: str) -> pd.DataFrame:
    df = buz_df.copy()
    df['datadate'] = pd.to_datetime(df['datadate'])
    df['srcdate'] = pd.to_datetime(df['srcdate'])

    # Filter segment types of interest and deduplicate on latest srcdate when duplicate (gvkey, datadate, snms)
    df = df[df['stype'].isin(['BUSSEG', 'OPSEG'])]
    df['_prio'] = (df['stype'] != 'BUSSEG').astype(int)
    df = (
        df.sort_values(['gvkey', 'datadate', '_prio', 'srcdate'], ascending=[True, True, True, False])
          .drop_duplicates(['gvkey', 'datadate', 'snms'], keep='first')
          .drop(columns=['_prio'])
    )

    fmap = fyear_map_df[['gvkey', 'datadate', 'fyear']].copy()
    fmap['datadate'] = pd.to_datetime(fmap['datadate'])
    fmap['gvkey'] = fmap['gvkey'].astype(str).str.zfill(6)
    df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
    fmap = fmap.drop_duplicates(subset=['gvkey', 'datadate'])
    df = df.merge(fmap, on=['gvkey', 'datadate'], how='left')

    # Keep oil & gas E&P SICs
    df = df[df['sic'].isin([1389])]

    negative_list = load_negative_list(flags_path)
    df['is_related'] = ~df['snms'].isin(negative_list)

    # Build firm-year wide and compute ENP share
    df_groupby = df.groupby(['gvkey', 'fyear', 'is_related'])['sales'].sum()
    df_wide = (
        df_groupby
          .unstack('is_related')
          .rename(columns={True: 'related', False: 'non_related'})
          .reset_index()
    )
    df_wide[['related', 'non_related']] = df_wide[['related', 'non_related']].fillna(0)
    df_wide['total_sales'] = df_wide['related'] + df_wide['non_related']
    share = np.where(df_wide['total_sales'] > 0, df_wide['related'] / df_wide['total_sales'], np.nan)
    df_wide['enp_share'] = pd.Series(share).clip(lower=0, upper=1)

    return df_wide[['gvkey', 'fyear', 'enp_share']]


def compute_domestic_share_by_firm_year(geo_df: pd.DataFrame, fyear_map_df: pd.DataFrame) -> pd.DataFrame:
    df = geo_df.copy()
    df['gvkey'] = df['gvkey'].astype(str).str.zfill(6)
    df['datadate'] = pd.to_datetime(df['datadate'])
    df['srcdate'] = pd.to_datetime(df['srcdate'])

    fmap = fyear_map_df[['gvkey', 'datadate', 'fyear', 'loc']].copy()
    fmap['gvkey'] = fmap['gvkey'].astype(str).str.zfill(6)
    fmap['datadate'] = pd.to_datetime(fmap['datadate'])
    df = df.merge(fmap, on=['gvkey', 'datadate'], how='left')

    df = (
        df.sort_values(['gvkey', 'datadate', 'geotp', 'srcdate'])
          .drop_duplicates(['gvkey', 'datadate', 'geotp'], keep='last')
    )

    df = df[df['sic'].isin([1311, 1321])]
    df = df.drop(columns=['srcdate', 'stype', 'sid'], errors='ignore')

    df['is_domestic_sale'] = np.where(
        df['loc'] == 'USA',
        df['geotp'] == 2,
        df['snms'].isin(US_LIST),
    )

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

    domestic_share_by_firm = firm_year_totals.merge(
        firm_year_domestic, on=['gvkey', 'fyear'], how='left'
    )
    domestic_share_by_firm['domestic_sales'] = domestic_share_by_firm['domestic_sales'].fillna(0)
    denominator = domestic_share_by_firm['total_sales']
    domestic_share_by_firm['domestic_sales_share'] = np.where(
        denominator > 0,
        domestic_share_by_firm['domestic_sales'] / denominator,
        np.nan,
    )

    return domestic_share_by_firm[['gvkey', 'fyear', 'domestic_sales_share']]

def build_quarterly_panel():
    # Load inputs
    buz = pd.read_csv('data/raw_data/compustat/buz_seg.csv')
    geo = pd.read_csv('data/raw_data/compustat/geo_seg.csv')
    fmap = pd.read_csv('data/raw_data/compustat/fyear_map.csv')
    qfund = pd.read_csv('data/raw_data/compustat/quarterly_fundamentals.csv')
    explore = pd.read_csv('data/raw_data/compustat/exploration_expense.csv')

    # Normalize keys and types needed for merges
    qfund['gvkey'] = qfund['gvkey'].astype(str).str.zfill(6)
    fmap['gvkey'] = fmap['gvkey'].astype(str).str.zfill(6)
    explore['gvkey'] = explore['gvkey'].astype(str).str.zfill(6)
    explore['datadate'] = pd.to_datetime(explore['datadate'])
    explore['ogxpxq'] = pd.to_numeric(explore['ogxpxq'], errors='coerce')
    explore = (
        explore[['gvkey', 'datadate', 'ogxpxq']]
            .drop_duplicates(subset=['gvkey', 'datadate'], keep='last')
    )

    # Compute firm-year shares
    enp_by_year = compute_enp_share_by_firm_year(
        buz_df=buz,
        fyear_map_df=fmap,
        flags_path='data/processed_data/segment_flags.csv',
    )
    dom_by_year = compute_domestic_share_by_firm_year(
        geo_df=geo,
        fyear_map_df=fmap,
    )

    # Map to quarterly by (gvkey, fyearq)
    panel = qfund.copy()
    panel['datadate'] = pd.to_datetime(panel['datadate'])
    panel['gvkey'] = panel['gvkey'].astype(str).str.zfill(6)

    # Ensure fyear for quarterly fundamentals exists (from file header: fyearq)
    if 'fyearq' not in panel.columns:
        raise ValueError("quarterly_fundamentals.csv must contain 'fyearq'")

    # Derive calendar quarter end date if fqtr is available
    if 'fqtr' in panel.columns:
        panel['date'] = _quarter_end_date(panel['fyearq'], panel['fqtr'])

    # Merge enp and domestic shares
    panel = panel.merge(
        enp_by_year.rename(columns={'fyear': 'fyearq'}),
        on=['gvkey', 'fyearq'],
        how='left',
    )
    panel = panel.merge(
        dom_by_year.rename(columns={'fyear': 'fyearq'}),
        on=['gvkey', 'fyearq'],
        how='left',
    )
    # Merge exploration expense (ogxpxq) by gvkey and datadate; fill missing with 0
    panel = panel.merge(
        explore,
        on=['gvkey', 'datadate'],
        how='left',
    )
    panel['ogxpxq'] = pd.to_numeric(panel['ogxpxq'], errors='coerce').fillna(0)
    # Compute ENP market share and HHI on the panel
    panel['revtq'] = pd.to_numeric(panel['revtq'], errors='coerce')
    panel['enp_share'] = pd.to_numeric(panel['enp_share'], errors='coerce').fillna(1.0)
    panel['domestic_sales_share'] = pd.to_numeric(panel['domestic_sales_share'], errors='coerce').fillna(1.0)

    # Scale fundamentals by enp_share when enp_share <= 0.9 (exclude ogxpxq)
    scale_factor = np.where(panel['enp_share'] > 0.9, 1.0, panel['enp_share'])
    fundamentals_to_scale = ['revtq', 'cogsq', 'oibdpq', 'oiadpq', 'atq', 'icaptq']
    for col in fundamentals_to_scale:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors='coerce') * scale_factor

    # Market share and HHI based on scaled revenue (revtq)
    panel = panel.sort_values(['gvkey', 'date'])
    revtq_lag1 = panel.groupby('gvkey', group_keys=False)['revtq'].shift(1)
    revtq_ttm_4 = (
        panel.groupby('gvkey', group_keys=False)['revtq']
             .apply(lambda s: s.shift(1).rolling(window=4, min_periods=4).sum())
    )
    # Fallback: if fewer than 4 prior quarters, use last quarter (lag 1)
    panel['revtq_ttm'] = np.where(revtq_ttm_4.notna(), revtq_ttm_4, revtq_lag1)
    panel['total_revtq_ttm'] = panel.groupby('date')['revtq_ttm'].transform('sum')
    panel['market_share'] = np.where(
        panel['total_revtq_ttm'] > 0,
        panel['revtq_ttm'] / panel['total_revtq_ttm'],
        np.nan,
    )
    panel['hhi'] = panel.groupby('date')['market_share'].transform(lambda s: np.square(s.fillna(0)).sum())

    # Normalise market_share cross-sectionally by date (z-score)
    mean_by_date = panel.groupby('date')['market_share'].transform('mean')
    std_by_date = panel.groupby('date')['market_share'].transform('std')
    panel['market_share_norm'] = np.where(
        (std_by_date > 0) & (~std_by_date.isna()),
        (panel['market_share'] - mean_by_date) / std_by_date,
        0,
    )

    # Normalise HHI across historical dates (z-score over unique dates)
    hhi_by_date = panel[['date', 'hhi']].drop_duplicates('date')
    valid_hhi = hhi_by_date['hhi'].notna()
    hhi_mean = hhi_by_date.loc[valid_hhi, 'hhi'].mean()
    hhi_std = hhi_by_date.loc[valid_hhi, 'hhi'].std()
    if pd.notna(hhi_std) and hhi_std > 0:
        hhi_by_date['hhi_norm'] = (hhi_by_date['hhi'] - hhi_mean) / hhi_std
    else:
        hhi_by_date['hhi_norm'] = 0
    panel = panel.merge(hhi_by_date[['date', 'hhi_norm']], on='date', how='left')

    # Calculate relevant margins
    panel['gross_margin'] = (panel['revtq'] + panel['ogxpxq'] - panel['cogsq']) / panel['revtq']
    panel['ebitdax_margin'] = (panel['oibdpq'] + panel['ogxpxq']) / panel['revtq']
    
    # Delta margins (QoQ change per firm)
    panel['delta_gross_margin'] = panel.groupby('gvkey')['gross_margin'].diff()
    panel['delta_ebitdax_margin'] = panel.groupby('gvkey')['ebitdax_margin'].diff()
    # Drop unneeded columns
    panel = panel.drop(columns=['costat', 'curcdq', 'datafmt', 'indfmt', 'consol', 'datadate'])

    # Reorder columns: place 'date' immediately after 'tic' if present
    if 'date' in panel.columns and 'tic' in panel.columns:
        cols = list(panel.columns)
        # Remove and reinsert 'date' after 'tic'
        cols.remove('date')
        tic_idx = cols.index('tic')
        cols.insert(tic_idx + 1, 'date')
        panel = panel[cols]

    # Optionally remove intermediate columns (keep only requested outputs)
    drop_cols = [c for c in ['revtq_ttm', 'total_revtq_ttm', 'sic', 'log_cogsq'] if c in panel.columns]
    panel = panel.drop(columns=drop_cols)

    # Save output
    panel.to_csv('data/processed_data/fundamentals_panel.csv', index=False)
    
if __name__ == '__main__':
    build_quarterly_panel()
