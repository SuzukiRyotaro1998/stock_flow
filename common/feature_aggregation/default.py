import pandas as pd
import sys

sys.path.append("../../")
from common.feature_aggregation.features import sma, ema, highest, lowest, hilo_ratio, adosc, hilo, spread
import numpy as np



class agg_feature_default:
    def agg_feature_default(df: pd.DataFrame) -> None:

        # Add close for feature_columns to use close in backtest
        feature_columns = []
        feature_columns += ["close"]

        # input features
        df["sma10"] = (sma(df, "close", 10) / sma(df, "close", 10).shift(1) - 1) * 100
        df["sma45"] = (sma(df, "close", 45) / sma(df, "close", 45).shift(1) - 1) * 100
        feature_columns += [f"sma{i}" for i in [10, 45]]

        df["ema45"] = (ema(df, "close", 10) / ema(df, "close", 10).shift(1) - 1) * 100
        feature_columns += [f"ema{i}" for i in [45]]

        df["close_sma5_ratio"] = df["close"] / sma(df, "close", 5)
        df["close_sma15_ratio"] = df["close"] / sma(df, "close", 15)
        df["close_sma30_ratio"] = df["close"] / sma(df, "close", 30)
        df["close_sma45_ratio"] = df["close"] / sma(df, "close", 45)
        feature_columns += [f"close_sma{i}_ratio" for i in [5, 15, 30, 45]]

        df["sma45_sma15_ratio"] = (sma(df, "close", 45) / sma(df, "close", 15) - 1) * 100
        df["sma30_sma10_ratio"] = (sma(df, "close", 30) / sma(df, "close", 10) - 1) * 100
        df["sma5_sma45_ratio"] = (sma(df, "close", 5) / sma(df, "close", 45) - 1) * 100
        df["sma5_sma15_ratio"] = (sma(df, "close", 5) / sma(df, "close", 15) - 1) * 100
        df["sma10_sma45_ratio"] = (sma(df, "close", 10) / sma(df, "close", 45) - 1) * 100
        feature_columns += [f"sma{i}_sma{j}_ratio" for i, j in zip([45, 30, 5, 5, 10], [15, 10, 45, 15, 45])]

        df["close_highest30_ratio"] = df["close"] / highest(df, "close", 30)
        df["close_highest45_ratio"] = df["close"] / highest(df, "close", 45)
        feature_columns += [f"close_highest{i}_ratio" for i in [30, 45]]

        df["close_10daysAgoHighest45_ratio"] = df["close"] / highest(df, "close", 45).shift(1 + 10)
        feature_columns += [f"close_{i}daysAgoHighest{j}_ratio" for i, j in zip([10], [45])]

        df["close_5daysAgoclose_ratio"] = df["close"] / df["close"].shift(5)
        df["close_10daysAgoclose_ratio"] = df["close"] / df["close"].shift(10)
        df["close_15daysAgoclose_ratio"] = df["close"] / df["close"].shift(15)
        df["close_20daysAgoclose_ratio"] = df["close"] / df["close"].shift(20)
        feature_columns += [f"close_{i}daysAgoclose_ratio" for i in [5, 10, 15, 20]]

        # # buy price and sell price
        # df["buy_price"], df["sell_price"] = position_price_by_atr(df, pips=1, atr_width=0.5)
        # feature_columns += ["buy_price", "sell_price"]

        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        return df, feature_columns

    def sma_1(df: pd.DataFrame) -> None:

        # Add close for feature_columns to use close in backtest
        feature_columns = []
        feature_columns += ["close"]

        for i in [10, 45]:
            # input features
            df[f"sma{i}"] = (sma(df, "close", i) / sma(df, "close", i).shift(1) - 1) * 100
            # MTM 10
            df[f"mtm{i}"] = df["close"] - df["close"].shift(i)
            df[f"mtm{i}"] = np.where(df[f"mtm{i}"] > 0, 1, -1)
            feature_columns += [f"sma{i}", f"mtm{i}"]

        df["ema45"] = (ema(df, "close", 10) / ema(df, "close", 10).shift(1) - 1) * 100
        feature_columns += [f"ema{i}" for i in [45]]

        df["close_sma5_ratio"] = df["close"] / sma(df, "close", 5)
        df["close_sma15_ratio"] = df["close"] / sma(df, "close", 15)
        df["close_sma30_ratio"] = df["close"] / sma(df, "close", 30)
        df["close_sma45_ratio"] = df["close"] / sma(df, "close", 45)
        feature_columns += [f"close_sma{i}_ratio" for i in [5, 15, 30, 45]]

        return df, feature_columns

    def sma_2(df: pd.DataFrame) -> None:
        feature_columns = []

        df["sma45_sma15_ratio"] = (sma(df, "close", 45) / sma(df, "close", 15) - 1) * 100
        df["sma30_sma10_ratio"] = (sma(df, "close", 30) / sma(df, "close", 10) - 1) * 100
        df["sma5_sma45_ratio"] = (sma(df, "close", 5) / sma(df, "close", 45) - 1) * 100
        df["sma5_sma15_ratio"] = (sma(df, "close", 5) / sma(df, "close", 15) - 1) * 100
        df["sma10_sma45_ratio"] = (sma(df, "close", 10) / sma(df, "close", 45) - 1) * 100
        feature_columns += [f"sma{i}_sma{j}_ratio" for i, j in zip([45, 30, 5, 5, 10], [15, 10, 45, 15, 45])]

        return df, feature_columns

    def highest_1(df: pd.DataFrame) -> None:
        feature_columns = []
        for i in [5, 15, 45, 60, 120]:
            df[f"close_highest{i}_ratio"] = df["close"] / highest(df, "close", i)
            df[f"close_highest{i}_ratio_shift"] = df["close"] / highest(df, "close", i).shift(1)

            feature_columns += [f"close_highest{i}_ratio", f"close_highest{i}_ratio_shift"]

        return df, feature_columns

    def highest_2(df: pd.DataFrame) -> None:
        feature_columns = []
        df["close_5daysAgoclose_ratio"] = df["close"] / df["close"].shift(5)
        df["close_10daysAgoclose_ratio"] = df["close"] / df["close"].shift(10)
        df["close_15daysAgoclose_ratio"] = df["close"] / df["close"].shift(15)
        df["close_20daysAgoclose_ratio"] = df["close"] / df["close"].shift(20)
        feature_columns += [f"close_{i}daysAgoclose_ratio" for i in [5, 10, 15, 20]]

        return df, feature_columns

    def low_1(df: pd.DataFrame) -> None:
        feature_columns = []

        for i in [5, 30, 120]:
            df[f"close_lowest{i}_ratio"] = df["close"] / lowest(df, "close", i)
            df[f"close_lowest{i}_ratio_shift"] = df["close"] / lowest(df, "close", i).shift(1)
            feature_columns += [f"close_lowest{i}_ratio"]
            feature_columns += [f"close_lowest{i}_ratio_shift"]

        return df, feature_columns

    def hilo_ratio_1(df: pd.DataFrame) -> None:
        feature_columns = []
        df["high_low_ratio"] = hilo_ratio(df)
        for i in [5, 15, 60, 120]:
            df[f"high_low_ratio_{i}sma"] = (sma(df, "high_low_ratio", i) / sma(df, "high_low_ratio", i).shift(1) - 1) * 100
            df[f"close_high_low_ratio_{i}sma_ratio"] = df["close"] / sma(df, "high_low_ratio", i)
            feature_columns += [f"high_low_ratio_{i}sma", f"close_high_low_ratio_{i}sma_ratio"]
        return df, feature_columns

    def amount(df: pd.DataFrame) -> None:
        feature_columns = []
        for i in [5, 15, 60, 120]:
            df[f"Volume_{i}daysago_ratio"] = df["volume"] / df["volume"].shift(i)
            feature_columns += [f"Volume_{i}daysago_ratio"]

            df["amount"] = df["close"] * df["volume"]
            df[f"amount_change_{i}"] = np.log(df["amount"] / df["amount"].shift(i))
            feature_columns += [f"amount_change_{i}"]

        return df, feature_columns

    def rsi(df: pd.DataFrame) -> None:
        feature_columns = []
        # Relative Strength Index in 5 days
        df["price_change"] = df["close"] - df["close"].shift(1)
        df["rsi"] = df["price_change"].rolling(5).apply(rsi) / 100
        df["rsi_FP"] = df["rsi"] - df["rsi"].shift(1)
        feature_columns += ["rsi", "rsi_FP"]

        return df, feature_columns

    def adosc(df: pd.DataFrame) -> None:
        feature_columns = []
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

        return df, feature_columns

    def typical(df: pd.DataFrame) -> None:
        feature_columns = []
        # Commodity Channel Index in 24 days
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["sma_cci"] = sma(df, "typical_price", 26)
        df["mean_deviation"] = np.abs(df["typical_price"] / df["sma_cci"])
        df["mean_deviation"] = sma(df, "mean_deviation", 24)
        df["cci"] = (df["typical_price"] - df["sma_cci"]) / (0.015 * df["mean_deviation"])
        df["cci_SG"] = np.where(df["cci"] > 0, 1, -1)
        feature_columns += ["typical_price", "sma_cci", "mean_deviation", "cci", "cci_SG"]
        return df, feature_columns

    def log_return(df: pd.DataFrame) -> None:
        feature_columns = []
        # Calculate log returns
        df["hilo"] = hilo(df)
        df["log_return1"] = df["hilo"].diff().fillna(0)
        df["log_return2"] = df["close"].diff().fillna(0)
        df["price_spread"] = spread(df)
        feature_columns += ["log_return1", "log_return2", "price_spread"]
        return df, feature_columns

    def pin_bar(df: pd.DataFrame) -> None:
        feature_columns = []
        # Calculate log returns

        df["upper"] = np.where(df["close"] >= df["open"], df["close"], df["open"])
        df["lower"] = np.where(df["close"] >= df["open"], df["open"], df["close"])
        df["pin_bar"] = (df["high"] - df["upper"]) / (df["upper"] - df["lower"])
        df["pin_bar2"] = (df["lower"] - df["low"]) / (df["upper"] - df["lower"])
        df["pin_bar3"] = (df["high"] - df["upper"]) / (df["lower"] - df["low"])
        df["in_upper"] = df["close"] / df["upper"].shift()
        df["in_lower"] = df["close"] / df["lower"].shift()

        df["inner_candle_sell"] = np.where(
            (df["high"] >= df["upper"].shift())
            & (df["low"] <= df["lower"].shift())
            & (df["close"].shift() / df["open"].shift() > 1)
            & (df["close"] / df["open"] < 1),
            1,
            -1,
        )

        df["inner_candle_buy"] = np.where(
            (df["high"] >= df["upper"].shift())
            & (df["low"] <= df["lower"].shift())
            & (df["close"].shift() / df["open"].shift() < 1)
            & (df["close"] / df["open"] > 1),
            1,
            -1,
        )

        feature_columns += ["upper", "lower", "pin_bar", "pin_bar2", "pin_bar3", "in_upper", "in_lower", "inner_candle_sell", "inner_candle_buy"]
        return df, feature_columns

