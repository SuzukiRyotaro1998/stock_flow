import pandas as pd
import numpy as np


def hilo(df: pd.DataFrame):
    return (df["high"] + df["low"]) / 2


def hilo_ratio(df: pd.DataFrame):
    return df["high"] / df["low"]


def sma(df: pd.DataFrame, column_name: str, period: int = 10) -> np.ndarray:
    return df[column_name].rolling(period, min_periods=1).mean()


def ema(df: pd.DataFrame, column_name: str, period: int = 10) -> np.ndarray:
    return df[column_name].ewm(span=period, min_periods=1).mean()


def ema_ratio(df: pd.DataFrame, period: int = 10, shift: int = 1) -> np.ndarray:
    return ema(df, period) / ema(df, period).shift(shift)


def highest(df: pd.DataFrame, column_name: str, period: int):
    df = df.copy()
    return df[column_name].rolling(period).max()


def lowest(df: pd.DataFrame, column_name: str, period: int):
    return df[column_name].rolling(period).min()


def close_ratio(df: pd.DataFrame, period: int = 10):
    return df["close"] / df["close"].shift(period)


def std(df: pd.DataFrame, column_name: str, period: int = 10) -> np.ndarray:
    return df[column_name].rolling(period).std()


def adosc(df: pd.DataFrame) -> np.ndarray:
    return (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"]) * df["volume"]


def spread(df: pd.DataFrame) -> np.ndarray:
    return (df["high"] - df["low"]) / ((df["high"] + df["low"]) / 2)


def realized_volatility(series_log_return: pd.Series):
    return np.sqrt(np.sum(series_log_return ** 2))


def rsi(x: pd.Series):
    up, down = [i for i in x if i > 0], [i for i in x if i <= 0]
    if len(down) == 0:
        return 100
    elif len(up) == 0:
        return 0
    else:
        up_average = sum(up) / len(up)
        down_average = -sum(down) / len(down)
        return 100 * up_average / (up_average + down_average)
