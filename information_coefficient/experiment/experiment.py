import os
import sys
import logging
import numpy as np
import pandas as pd

# import mlflow

sys.path.append("../../")
from common.custom_logger import CustomLogger
from common.feature_aggregation.talib import agg_feature

from common.constants import DATAFOLDER
from common.feature_aggregation.features import highest, sma, std
from ml_base.preprocess.src.label_features import position_label

logging.basicConfig(level=logging.INFO)
logger = CustomLogger("Preprocess_Logger")


def plot_scatter(x, returns, normalize=False):
    """
    :param np.ndarray x: 指標
    :param np.ndarray returns: リターン
    :param bool normalize: x をスケーリングするかどうか
    """
    assert len(x) == len(returns)
    # 正規化
    x = (x - x.mean()) / x.std() if normalize else x
    # 相関係数（情報係数）
    ic = np.corrcoef(x, returns)[0, 1]
    return ic


def target_calucuration(df):
    target_features = []
    # simple
    df.loc[:, ["target_simple"]] = df["close"].shift(-1) / df["open"].shift(-1)
    target_features.append("target_simple")

    for i in [2, 4, 6]:
        df.loc[:, [f"target_{i}"]] = df["close"].shift(-i) / df["close"].shift(-1)
        target_features.append(f"target_{i}")

    (y_buy, y_sell, buy_executed, sell_executed, buy_price, sell_price,) = position_label(
        df, execution_type="LIMIT", pips=1.0, fee_percent=-0.025, horizon_barrier=1, atr_width=0.1
    )
    df.loc[:, ["y_buy"]] = y_buy
    df.loc[:, ["y_sell"]] = y_sell
    target_features.append("y_buy")
    target_features.append("y_sell")

    return df, target_features


def IC(df, feature_columns):
    features = feature_columns.copy()
    for feature in features:
        # print(feature)
        df.loc[:, [f"{feature}_close_ratio"]] = df["close"] / df[feature]
        df.loc[:, [f"{feature}_tilt"]] = df[feature].copy() / df[feature].shift(1)
        df.loc[:, [f"{feature}_acceleration"]] = df[f"{feature}_tilt"] / df[f"{feature}_tilt"].shift(1)

        feature_columns += [f"{feature}_close_ratio", f"{feature}_tilt", f"{feature}_acceleration"]

        i = 5
        df.loc[:, [f"{feature}_highest_{i}"]] = highest(df.copy(), feature, i)
        df.loc[:, [f"close_{feature}_highest_{i}_ratio"]] = df["close"] / df[f"{feature}_highest_{i}"]
        feature_columns += [f"close_{feature}_highest_{i}_ratio"]

        df.loc[:, [f"{feature}_mean{i}"]] = sma(df, feature, i)
        df.loc[:, [f"close_{feature}_mean_{i}_ratio"]] = df["close"] / df[f"{feature}_mean{i}"]
        feature_columns += [f"close_{feature}_mean_{i}_ratio"]

        df.loc[:, [f"{feature}_std{i}"]] = std(df, feature, i)
        df.loc[:, [f"close_{feature}_std_{i}_ratio"]] = df["close"] / df[f"{feature}_std{i}"]
        feature_columns += [f"close_{feature}_std_{i}_ratio"]

        # df.loc[:, [f"{feature}_realized_volatility{i}"]] = df[feature].rolling(i).agg([realized_volatility])
        # df.loc[:, [f"close_{feature}_realized_volatility{i}_ratio"]] = df["close"] / df[f"{feature}_realized_volatility{i}"]
        # feature_columns += [f"{feature}_realized_volatility{i}", f"close_{feature}_realized_volatility{i}_ratio"]

    df, target_features = target_calucuration(df)
    df = df.dropna()

    df_result = pd.DataFrame()
    for i in feature_columns:
        scores = []
        for target in target_features:
            ic = plot_scatter(df[i].values, df[target].values)
            scores.append(ic)
        df_result_fragment = pd.DataFrame([scores], index=[i], columns=target_features)
        df_result = df_result.append(df_result_fragment)
    df_result["IC_abs"] = df_result["target_simple"].abs()

    return df_result


def experiment(downstream, exchange_name: str, trading_type: str, pair_name: str, time_bar: str):

    print("==========================")
    print(f"{exchange_name}_{trading_type}_{pair_name}_{time_bar}")
    print("==========================")

    os.makedirs(downstream, exist_ok=True)
    # source_path = (os.path.join(DATAFOLDER.ohlc_data_folder, time_bar, f"{exchange_name}_{trading_type}_{pair_name}.parquet.gzip"),)
    df = pd.read_parquet(os.path.join(DATAFOLDER.ohlc_data_folder, time_bar, f"{exchange_name}_{trading_type}_{pair_name}.parquet.gzip"), engine="pyarrow",)
    df = df.dropna()
    df = df[-1000000:]
    # df = df[-1000:]
    df_result_total = pd.DataFrame()

    # ===================
    # ryotaro made
    # ===================
    # highest
    data, feature_columns = agg_feature.highest(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    # Overlap Studies
    # ===================

    # BBANDS
    data, feature_columns = agg_feature.BBANDS(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # DEMA
    data, feature_columns = agg_feature.DEMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # HT_TRENDLINE
    data, feature_columns = agg_feature.HT_TRENDLINE(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # KAMA
    data, feature_columns = agg_feature.KAMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MAMA
    data, feature_columns = agg_feature.MAMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MIDPRICE
    data, feature_columns = agg_feature.MIDPRICE(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # SAR
    # data, feature_columns = agg_feature.SAR(df.copy())
    # df_result = IC(data, feature_columns)
    # df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # # SAREXT
    # data, feature_columns = agg_feature.SAREXT(df.copy())
    # df_result = IC(data, feature_columns)
    # df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TEMA
    data, feature_columns = agg_feature.TEMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TRIMA
    data, feature_columns = agg_feature.TRIMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # WMA
    data, feature_columns = agg_feature.WMA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Momentum Indicator Functions
    # ===================

    # ADX
    data, feature_columns = agg_feature.ADX(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ADXR
    data, feature_columns = agg_feature.ADXR(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # APO
    data, feature_columns = agg_feature.APO(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # AROON
    data, feature_columns = agg_feature.AROON(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # AROONOSC
    data, feature_columns = agg_feature.AROONOSC(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # BOP
    data, feature_columns = agg_feature.BOP(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # CCI
    data, feature_columns = agg_feature.CCI(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # CMO
    data, feature_columns = agg_feature.CMO(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MAC
    data, feature_columns = agg_feature.MAC(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MACDEXT
    data, feature_columns = agg_feature.MACDEXT(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MACDFIX
    data, feature_columns = agg_feature.MACDFIX(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # MI
    data, feature_columns = agg_feature.MI(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # RO
    data, feature_columns = agg_feature.RO(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ROCR
    data, feature_columns = agg_feature.ROCR(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # STOCH1
    data, feature_columns = agg_feature.STOCH1(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # STOCH2
    data, feature_columns = agg_feature.STOCH2(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # STOCH3
    data, feature_columns = agg_feature.STOCH3(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TRIX
    data, feature_columns = agg_feature.TRIX(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Volume Indicators
    # ===================
    # AD
    data, feature_columns = agg_feature.AD(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Volatility Indicator Functions
    # ===================
    # ATR
    data, feature_columns = agg_feature.ATR(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Cycle Indicator Functions
    # ===================
    # HT_DCPERIOD
    data, feature_columns = agg_feature.HT_DCPERIOD(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # sine
    data, feature_columns = agg_feature.sine(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Price Transform Functions
    # ===================
    # PRICE
    data, feature_columns = agg_feature.PRICE(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Pattern Recognition Functions
    # ===================

    # pattern_recognition
    data, feature_columns = agg_feature.pattern_recognition(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition2
    data, feature_columns = agg_feature.pattern_recognition2(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition3
    data, feature_columns = agg_feature.pattern_recognition3(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition4
    data, feature_columns = agg_feature.pattern_recognition4(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition5
    data, feature_columns = agg_feature.pattern_recognition5(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition6
    data, feature_columns = agg_feature.pattern_recognition6(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition7
    data, feature_columns = agg_feature.pattern_recognition7(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition8
    data, feature_columns = agg_feature.pattern_recognition8(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition9
    data, feature_columns = agg_feature.pattern_recognition9(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition10
    data, feature_columns = agg_feature.pattern_recognition10(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition11
    # data, feature_columns = agg_feature.pattern_recognition11(df.copy())
    # df_result = IC(data, feature_columns)
    # df_result = df_result.sort_values("IC_abs", ascending=False)
    # # display(df_result)
    # df_result_total = df_result_total.append(df_result)

    # pattern_recognition12
    data, feature_columns = agg_feature.pattern_recognition12(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition13
    data, feature_columns = agg_feature.pattern_recognition13(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition14
    data, feature_columns = agg_feature.pattern_recognition14(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition15
    data, feature_columns = agg_feature.pattern_recognition15(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # pattern_recognition16
    data, feature_columns = agg_feature.pattern_recognition16(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Statistic Functions
    # ===================
    # LINEARREG
    data, feature_columns = agg_feature.LINEARREG(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # LINEARREG_SLOPE
    data, feature_columns = agg_feature.LINEARREG_SLOPE(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # VAR
    data, feature_columns = agg_feature.VAR(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TSF
    data, feature_columns = agg_feature.TSF(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # STDDEV
    data, feature_columns = agg_feature.STDDEV(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # LINEARREG_ANGLE
    data, feature_columns = agg_feature.LINEARREG_ANGLE(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # BETA
    data, feature_columns = agg_feature.BETA(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # CORREL
    data, feature_columns = agg_feature.CORREL(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Statistic Functions
    # ===================
    # ASIN
    data, feature_columns = agg_feature.ASIN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # ATAN
    data, feature_columns = agg_feature.ATAN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # CEIL
    data, feature_columns = agg_feature.CEIL(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # COS
    data, feature_columns = agg_feature.COS(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # COSH
    data, feature_columns = agg_feature.COSH(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # EXP
    data, feature_columns = agg_feature.EXP(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # FLOOR
    data, feature_columns = agg_feature.FLOOR(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # LN
    data, feature_columns = agg_feature.LN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # LOG10
    data, feature_columns = agg_feature.LOG10(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # SIN
    data, feature_columns = agg_feature.SIN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # SINH
    data, feature_columns = agg_feature.SINH(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TAN
    data, feature_columns = agg_feature.TAN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # TANH
    data, feature_columns = agg_feature.TANH(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # ===================
    #  Math Operator Functions
    # ===================
    # # MINMAX
    # data, feature_columns = agg_feature.MINMAX(df.copy())
    # df_result = IC(data, feature_columns)
    # df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # ADD
    data, feature_columns = agg_feature.ADD(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # SUM
    # data, feature_columns = agg_feature.SUM(df.copy())
    # df_result = IC(data, feature_columns)
    # df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # SUB
    data, feature_columns = agg_feature.SUB(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # MULT
    data, feature_columns = agg_feature.MULT(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # DIV
    data, feature_columns = agg_feature.DIV(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # MAX
    data, feature_columns = agg_feature.MAX(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)
    # MIN
    data, feature_columns = agg_feature.MIN(df.copy())
    df_result = IC(data, feature_columns)
    df_result = df_result.sort_values("IC_abs", ascending=False)
    # display(df_result)
    df_result_total = df_result_total.append(df_result)

    # 結果のまとめ
    df_result_total = df_result_total.sort_values("IC_abs", ascending=False)
    print(df_result_total[:15])
    df_result_total = df_result_total[:15]

    # result_dict = dict(df_result_total[["feature", "IC_abs"]].values)
    # mlflow.log_metrics(result_dict)
    # mlflow.log_artifacts(downstream)

    # logger.info(f"result info files have been saved in {downstream}")
