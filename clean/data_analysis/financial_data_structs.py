
import numpy as np
import pandas as pd
import pyarrow


def load_ticks_CSV(fp):
    """[summary]

    Arguments:
        fp {[type]} -- [description]

    Returns:
        DataFrame --
    """
    cols = list(
        map(str.lower, ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']))
    df = (pd.read_csv(fp, header=None)
            .rename(columns=dict(zip(range(len(cols)), cols)))
            .assign(timestamp=lambda df: (pd.to_datetime(df['date']+df['time'],
                                                        format='%m/%d/%Y%H:%M:%S')))
            .assign(dollar_volume=lambda df: df['price']*df['volume'])
            .drop(['date', 'time'], axis=1)
            .set_index('timestamp')
            .drop_duplicates())
    # ? Remove outliers here or own function
    return df

def remove_outliers(df):
    """Remove Price Outliers from dataframe

    Finds outliers by looking for prices that are greater than 3 std from mean.
    This is a tricky issue to solve for all financial products since there are time gaps
    along with huge price jumps, so this is not neccessarily a perfect solution but it works
    for our example with $IVE tick data. See link:
    https://stats.stackexchange.com/questions/1223/robust-outlier-detection-in-financial-timeseries

    Arguments:
        df {DataFrame} -- Tick data df with 'price' column

    Returns:
        DataFrame -- dataframe with outliers removed
    """
    return df[np.abs(df.price-df.price.mean()) <= (3*df.price.std())]

def df_to_parquet(df, out_fp):
    df.to_parquet(out_fp)

def parquet_to_df(out_fp):
    return pd.read_parquet(out_fp)

def get_tick_bars(df, tick_threshold):
    return df[::tick_threshold]

def get_volume_bars(df, volume_threshold):
    idxs = []
    vol_counter = 0
    # for i, j in enumerate(df['self.volume_col'])
    for idx, vol in enumerate(df['volume']):
        vol_counter += vol
        if vol >= volume_threshold:
            idxs.append(idx)
            vol_counter = 0
    return df.iloc[idxs].drop_duplicates()

def get_dollar_bars(df, dollar_threshold):
    idxs = []
    dol_counter = 0
    # for i, j in enumerate(df['self.volume_col'])
    for idx, dol in enumerate(df['dollar_volume']):
        dol_counter += dol
        if dol >= dollar_threshold:
            idxs.append(idx)
            dol_counter = 0
    return df.iloc[idxs].drop_duplicates()


def get_tick_imbalance_bars(df, exp_tick_imbal=100):
    raise NotImplementedError('To be implemented')


'''
TODO:
- Allow collumns to be specified i.e. for a csv not from Kibot
- set class variables for volume, dollar volume column names
  so they can be used in get_dollar_bars and get_volume_bars instead of
  hard-coded
'''