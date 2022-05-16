import pandas as pd
import sys

sys.path.append("../../")
from common.custom_logger import CustomLogger
import numpy as np
from scipy.stats import linregress

logger = CustomLogger("Preprocess_Logger")


class trend_line:
    def trend_line(df: pd.DataFrame) -> None:

        # Add close for feature_columns to use close in bacwindotest
        feature_columns = []
        for window in [30, 45, 60, 120]:
            high_trend = []
            low_trend = []
            high_trend_tilt = []
            low_trend_tilt = []

            for i in range(len(df)):
                if i < window:
                    high_trend.append(np.nan)
                    high_trend_tilt.append(np.nan)
                    low_trend.append(np.nan)
                    low_trend_tilt.append(np.nan)
                    continue

                before = i - window
                data = df[before:i].reset_index()
                data["time_id"] = data.index + 1
                df_fin = data.copy()
                df_high = data.copy()
                df_low = data.copy()

                while len(df_high) > 3:
                    reg_1 = linregress(x=df_high["time_id"], y=df_high["high"],)
                    df_high = df_high.loc[df_high["high"] > reg_1[0] * df_high["time_id"] + reg_1[1]]

                if len(df_high) != 0:
                    reg_1 = linregress(x=df_high["time_id"], y=df_high["high"],)
                df_fin["high_trend"] = reg_1[0] * df_fin["time_id"] + reg_1[1]
                high_trend.append(df_fin.iloc[-1]["high_trend"])
                high_trend_tilt.append(reg_1[0])

                # 安値のトレンドライン
                while len(df_low) > 3:
                    reg_2 = linregress(x=df_low["time_id"], y=df_low["low"],)
                    df_low = df_low.loc[df_low["low"] < reg_2[0] * df_low["time_id"] + reg_2[1]]

                # if len(df_low) == 0:
                #     print(df_low)
                #     reg_2 = linregress(x=df_low["time_id"], y=df_low["low"],)
                df_fin["low_trend"] = reg_2[0] * df_fin["time_id"] + reg_2[1]
                low_trend.append(df_fin.iloc[-1]["low_trend"])
                low_trend_tilt.append(reg_2[0])

            df[f"high_trend{window}"] = high_trend
            df[f"high_trend_tilt{window}"] = high_trend_tilt
            df[f"low_trend{window}"] = low_trend
            df[f"low_trend_tilt{window}"] = low_trend_tilt
            df[f"close_high_trend_ratio{window}"] = df["close"] / df[f"high_trend{window}"]
            df[f"close_low_trend_ratio{window}"] = df["close"] / df[f"low_trend{window}"]
            df[f"high_trend_shift_ratio{window}"] = df[f"high_trend{window}"] / df[f"high_trend{window}"].shift()
            df[f"low_trend_shift_ratio{window}"] = df[f"low_trend{window}"] / df[f"low_trend{window}"].shift()

            feature_columns += [f"high_trend{window}", f"low_trend{window}", f"high_trend_tilt{window}", f"low_trend_tilt{window}"]
            feature_columns += [
                f"close_high_trend_ratio{window}",
                f"close_low_trend_ratio{window}",
                f"high_trend_shift_ratio{window}",
                f"low_trend_shift_ratio{window}",
            ]

        return df, feature_columns

    def supres(low, high, n=28, min_touches=2, stat_likeness_percent=1.5, bounce_percent=5):

        df = pd.concat([high, low], keys=["high", "low"], axis=1)
        df["sup"] = pd.Series(np.zeros(len(low)))
        df["res"] = pd.Series(np.zeros(len(low)))
        df["sup_break"] = pd.Series(np.zeros(len(low)))
        df["sup_break"] = 0
        df["res_break"] = pd.Series(np.zeros(len(high)))
        df["res_break"] = 0

        for x in range((n - 1) + n, len(df)):
            tempdf = df[x - n : x + 1].copy()
            sup = None
            res = None
            maxima = tempdf.high.max()
            minima = tempdf.low.min()
            move_range = maxima - minima
            move_allowance = move_range * (stat_likeness_percent / 100)
            bounce_distance = move_range * (bounce_percent / 100)
            touchdown = 0
            awaiting_bounce = False
            for y in range(0, len(tempdf)):
                if abs(maxima - tempdf.high.iloc[y]) < move_allowance and not awaiting_bounce:
                    touchdown = touchdown + 1
                    awaiting_bounce = True
                elif abs(maxima - tempdf.high.iloc[y]) > bounce_distance:
                    awaiting_bounce = False
            if touchdown >= min_touches:
                res = maxima

            touchdown = 0
            awaiting_bounce = False
            for y in range(0, len(tempdf)):
                if abs(tempdf.low.iloc[y] - minima) < move_allowance and not awaiting_bounce:
                    touchdown = touchdown + 1
                    awaiting_bounce = True
                elif abs(tempdf.low.iloc[y] - minima) > bounce_distance:
                    awaiting_bounce = False
            if touchdown >= min_touches:
                sup = minima
            if sup:
                df["sup"].iloc[x] = sup
            if res:
                df["res"].iloc[x] = res
        res_break_indices = list(df[(np.isnan(df["res"]) & ~np.isnan(df.shift(1)["res"])) & (df["high"] > df.shift(1)["res"])].index)
        for index in res_break_indices:
            df["res_break"].at[index] = 1
        sup_break_indices = list(df[(np.isnan(df["sup"]) & ~np.isnan(df.shift(1)["sup"])) & (df["low"] < df.shift(1)["sup"])].index)
        for index in sup_break_indices:
            df["sup_break"].at[index] = 1
        ret_df = pd.concat([df["sup"], df["res"], df["sup_break"], df["res_break"]], keys=["sup", "res", "sup_break", "res_break"], axis=1)
        return ret_df
