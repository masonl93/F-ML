import os
from pathlib import PurePath, Path

from financial_data import FinDataStruct


# Constants - Modify to fit your use case
# TODO: use config.ini file for these?
INPUT_FILE = 'data/raw/kibot/IVE_tickbidask.csv'
OUTPUT_FILE = 'data/clean/IVE_tick.parq'
TICK_THRESHOLD = 100
VOLUME_THRESHOLD = 10_000
DOLLAR_THRESHOLD = 1_000_000


def df_summary(df):
    print(df.describe())
    print(df.shape)


def main(in_fp, out_fp):
    print('hello mate')

    csv_data = Path(in_fp)
    out_fp = Path(out_fp)

    # Create financial data object
    fin_data = FinDataStruct()

    if not out_fp.is_file():
        print('Saving Input File to Parquet')

        # Load csv data to dataframe
        df = fin_data.load_ticks_CSV(csv_data)
        print(df.head())

        # Save dataframe as Parquet for quick loading
        fin_data.df_to_parquet(df, out_fp)
    else:
        print('Output file already exists, not using input file')

    # Load DataFrame from Parquet file
    df = fin_data.parquet_to_df(out_fp)
    df_summary(df)

    # Remove outliers
    df = fin_data.remove_outliers(df)
    df_summary(df)

    # TODO: add time bars? Close price

    # Tick bars
    df_tick = fin_data.get_tick_bars(df, TICK_THRESHOLD)
    df_summary(df_tick)

    # Volume bars
    df_vol = fin_data.get_volume_bars(df, VOLUME_THRESHOLD)
    df_summary(df_vol)

    # Dollar bars
    df_dol = fin_data.get_dollar_bars(df, DOLLAR_THRESHOLD)
    df_summary(df_dol)


if __name__ == '__main__':
    main(INPUT_FILE, OUTPUT_FILE)

# TODO: Instead of/in addition to main.py, create a jupyter notebook so we can plot?
