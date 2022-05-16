import pandas as pd
import sys

sys.path.append("../../")
from common.custom_logger import CustomLogger
from features import hilo, hilo_ratio, sma, ema, high, low, std, adosc, spread, realized_volatility, rsi
import numpy as np

logger = CustomLogger("Preprocess_Logger")


def agg_feature(df: pd.DataFrame, execution_type: str, fee_percent: float) -> None:
    # Add close for feature_columns to use close in backtest
    feature_columns = []
    feature_columns += ["close"]

    # input features
    for i in [5, 10, 15, 30, 45, 60, 90, 120, 150, 200, 300]:
        df[f"sma{i}"] = (sma(df, "close", i) / sma(df, "close", i).shift(1) - 1) * 100
        # MTM 10
        # df[f"mtm{i}"] = df["close"] - df["close"].shift(i)
        # df[f"mtm{i}"] = np.where(df[f"mtm{i}"] > 0, 1, -1)
        # feature_columns += [f"sma{i}", f"mtm{i}"]

    # for i in [5, 10, 15, 30, 45, 60, 90, 120, 150, 200, 300]:
    #     df[f"ema{i}"] = (ema(df, "close", 10) / ema(df, "close", 10).shift(1) - 1) * 100
    #     feature_columns += [f"ema{i}"]

    # for i in [5, 10, 15, 30, 45, 60, 90, 120, 150, 200, 300]:
    #     df[f"close_sma{i}_ratio"] = df["close"] / sma(df, "close", i)
    #     feature_columns += [f"close_sma{i}_ratio"]

    # for i in [5, 10, 15, 30, 45, 60, 120, 150, 300]:
    #     for k in [5, 10, 15, 30, 45, 60, 120, 150, 300]:
    #         df[f"sma{i}_sma{k}_ratio"] = (sma(df, "close", i) / sma(df, "close", k) - 1) * 100
    #         feature_columns += [f"sma{i}_sma{k}_ratio"]

    for i in [5, 10, 15, 30, 45, 60, 90, 120, 150, 200, 300]:
        df[f"close_lowest{i}_ratio"] = df["close"] / low(df, "close", i)
        df[f"close_lowest{i}_ratio_shift"] = df["close"] / low(df, "close", i).shift(1)

        feature_columns += [f"close_lowest{i}_ratio", f"close_lowest{i}_ratio_shift"]

    # for i in [5, 10, 15, 30, 45, 60, 120, 150, 300]:
    #     for k in [5, 10, 15, 30, 45, 60, 120, 150, 300]:
    #         df[f"close_{i}daysAgoHighest{k}_ratio"] = df["close"] / high(df, "close", i).shift(1 + k)
    #         feature_columns += [f"close_{i}daysAgoHighest{k}_ratio"]

    # for i in [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300]:
    #     df[f"close_{i}daysAgoclose_ratio"] = df["close"] / df["close"].shift(i)
    #     feature_columns += [f"close_{i}daysAgoclose_ratio"]

    df["high_low_ratio"] = hilo_ratio(df)
    for i in [5, 10, 15, 30, 45, 60, 90, 120, 150, 200, 300]:
        df[f"high_low_ratio_{i}sma"] = (sma(df, "high_low_ratio", i) / sma(df, "high_low_ratio", i).shift(1) - 1) * 100
        df[f"close_high_low_ratio_{i}sma_ratio"] = df["close"] / sma(df, "high_low_ratio", i)
        feature_columns += [f"high_low_ratio_{i}sma", f"close_high_low_ratio_{i}sma_ratio"]

    for i in [5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 200, 300]:
        df[f"Volume_{i}daysago_ratio"] = df["volume"] / df["volume"].shift(i)
        feature_columns += [f"Volume_{i}daysago_ratio"]

    # amount
    for i in [5, 15, 30, 45, 90, 120, 200, 300]:
        df["amount"] = df["close"] * df["volume"]
        df[f"amount_change_{i}"] = np.log(df["amount"] / df["amount"].shift(i))
        feature_columns += [f"amount_change_{i}"]

    # Relative Strength Index in 5 days
    df["price_change"] = df["close"] - df["close"].shift(1)
    df["rsi"] = df["price_change"].rolling(5).apply(rsi) / 100
    df["rsi_FP"] = df["rsi"] - df["rsi"].shift(1)
    feature_columns += ["rsi", "rsi_FP"]

    # ADOSC
    df["adosc_1"] = adosc(df)
    df["adosc"] = df["adosc_1"].cumsum()
    df["adosc_ratio"] = df["adosc"] / df["adosc"].shift(1)
    df["adosc_ema3"] = ema(df, "adosc", 3)
    df["adosc_ema10"] = ema(df, "adosc", 10)
    df["adosc_ema3_ratio"] = df["adosc_ema3"] / df["adosc_1"]
    df["adosc_ema10_ratio_ratio"] = df["adosc_ema10"] / df["adosc_1"]
    df["adosc_SG"] = np.where((df["adosc_ema3"] - df["adosc_ema10"]) > 0, 1, -1)
    feature_columns += ["adosc_ratio", "adosc_SG", "adosc_ema3_ratio", "adosc_ema10_ratio_ratio"]

    # Moving Average Convergence Divergence
    df["macd"] = sma(df, "close", 12) - sma(df, "close", 26)
    df["macd_SG"] = sma(df, "macd", 26)
    df["macd_histogram"] = df["macd"] - df["macd_SG"]
    df["macd_histogram"] = np.where(df["macd_histogram"] > 0, 1, -1)
    df["macd_SG"] = np.where(df["macd_SG"] > 0, 1, -1)
    df["macd"] = np.where(df["macd"] > 0, 1, -1)
    feature_columns += ["macd", "macd_SG", "macd_histogram"]

    # Commodity Channel Index in 24 days
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["sma_cci"] = sma(df, "typical_price", 26)
    df["mean_deviation"] = np.abs(df["typical_price"] / df["sma_cci"])
    df["mean_deviation"] = sma(df, "mean_deviation", 24)
    df["cci"] = (df["typical_price"] - df["sma_cci"]) / (0.015 * df["mean_deviation"])
    df["cci_SG"] = np.where(df["cci"] > 0, 1, -1)
    feature_columns += ["mean_deviation", "cci", "cci_SG"]

    # Calculate log returns
    df["hilo"] = hilo(df)
    df["log_return1"] = df["hilo"].diff().fillna(0)
    df["log_return2"] = df["close"].diff().fillna(0)
    df["price_spread"] = spread(df)
    feature_columns += ["log_return1", "log_return2", "price_spread"]

    create_feature = [
        "log_return1",
        "log_return2",
        "price_spread",
        "sma5",
        "sma30",
        "sma60",
        "close_highest5_ratio",
        "close_highest15_ratio",
        "close_highest60_ratio",
    ]
    # for i in [5, 10, 15, 30, 45, 60, 120, 150, 300]:
    #     for k in create_feature:
    #         df[f"{k}_mean{i}"] = sma(df, k, i)
    #         df[f"{k}_std{i}"] = std(df, k, i)
    #         df[f"{k}_max{i}"] = high(df, k, i)
    #         df[f"{k}_realized_volatility{i}"] = df[k].rolling(i).agg([realized_volatility])
    #         feature_columns += [f"{k}_mean{i}", f"{k}_std{i}", f"{k}_max{i}", f"{k}_realized_volatility{i}"]

    # buy price and sell price
    # df["buy_price"], df["sell_price"] = position_price_by_atr(df, pips=1, atr_width=0.5)
    # feature_columns += ["buy_price", "sell_price"]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df, feature_columns
