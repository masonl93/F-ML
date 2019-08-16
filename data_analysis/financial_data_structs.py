
import numpy as np
import pandas as pd
import pyarrow


def load_ticks_CSV(fp):
    """[summary]

    Arguments:
        fp {[type]} -- [description]

    Returns:
        DataFrame --

    # TODO: Allow collumns to be specified i.e. for a csv not from Kibot
    """
    cols = list(
        map(str.lower, ['Date', 'Time', 'Price', 'Bid', 'Ask', 'Volume']))
    df = (pd.read_csv(fp, header=None)
            .rename(columns=dict(zip(range(len(cols)), cols)))
            .assign(timestamp=lambda df: (pd.to_datetime(df['date'] + df['time'],
                                                         format='%m/%d/%Y%H:%M:%S')))
            .assign(dollar_volume=lambda df: df['price'] * df['volume'])
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
    return df[np.abs(df.price - df.price.mean()) <= (3 * df.price.std())]


def df_to_parquet(df, out_fp):
    df.to_parquet(out_fp)


def parquet_to_df(out_fp):
    return pd.read_parquet(out_fp)


def get_tick_bars(df, tick_threshold):
    return df[::tick_threshold]


def get_volume_bars(df, volume_threshold):
    # TODO: Allow volume column name to be specified
    idxs = []
    vol_counter = 0
    # for i, j in enumerate(df[volume_col])
    for idx, vol in enumerate(df['volume']):
        vol_counter += vol
        if vol_counter >= volume_threshold:
            idxs.append(idx)
            vol_counter = 0
    return df.iloc[idxs].drop_duplicates()


def get_dollar_bars(df, dollar_threshold):
    # TODO: Allow dollar column name to be specified
    idxs = []
    dol_counter = 0
    # for i, j in enumerate(df[dollar_col])
    for idx, dol in enumerate(df['dollar_volume']):
        dol_counter += dol
        if dol_counter >= dollar_threshold:
            idxs.append(idx)
            dol_counter = 0
    return df.iloc[idxs].drop_duplicates()


def get_tick_imbalance_bars(df, exp_tick_imbal=100):
    raise NotImplementedError('To be implemented')


def calc_returns(s):
    rets = np.diff(np.log(s))
    return (pd.Series(rets, index=s.index[1:]))


class MovingAvgCross():
    """ Moving Average Cross

    Arguments:
        price_series {pandas.Series} -- index is time series with prices as values
        ma_long {int} -- long moving average window
        ma_short {int} -- short moving average window
    """

    def __init__(self, price_series, ma_long, ma_short):
        self.price_series = price_series
        self.ma_long = ma_long
        self.ma_short = ma_short
        self.ma_df = pd.DataFrame().assign(price=price_series).assign(ma_l=price_series.rolling(
            window=ma_long).mean()).assign(ma_s=price_series.rolling(window=ma_short).mean())

    def buy_signal(self):
        cross = self.ma_df.ma_s > self.ma_df.ma_l
        # Check if this is first time crossing
        first_cross = self.ma_df.ma_s.shift(1) < self.ma_df.ma_l.shift(1)
        return self.ma_df.ma_s[cross & first_cross]

    def sell_signal(self):
        cross = self.ma_df.ma_s < self.ma_df.ma_l
        # Check if this is first time crossing
        first_cross = self.ma_df.ma_s.shift(1) > self.ma_df.ma_l.shift(1)
        return self.ma_df.ma_s[cross & first_cross]


class BollingerBands():
    """ Bollinger Bands

    Arguments:
        price_series {pandas.Series} -- index is time series with prices as values
        window {int} -- moving average window
        num_std {int} -- standard deviations of the bollinger bands
    """

    def __init__(self, price_series, window, num_std):
        self.price_series = price_series
        self.window = window
        self.num_std = num_std
        self.bol_df = pd.DataFrame().assign(price=price_series).assign(
            ma=price_series.rolling(window=window).mean())
        std = price_series.rolling(window=window).std(ddof=0)
        self.bol_df['high'] = self.bol_df.ma + num_std * std
        self.bol_df['low'] = self.bol_df.ma - num_std * std

    def buy_signal(self):
        """ Price series crosses bottom band
            Predicting mean reversion
        """
        cross = self.bol_df.price < self.bol_df.low
        # Check if this is first time crossing
        first_cross = self.bol_df.price.shift(1) > self.bol_df.low.shift(1)
        return self.bol_df.price[cross & first_cross]

    def sell_signal(self):
        """ Price series crosses top band
            Predicting mean reversion
        """
        cross = self.bol_df.price > self.bol_df.high
        # Check if this is first time crossing
        first_cross = self.bol_df.price.shift(1) < self.bol_df.high.shift(1)
        return self.bol_df.price[cross & first_cross]
