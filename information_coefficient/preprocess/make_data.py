import sys

sys.path.append("../../../")
from common.feature_aggregation.talib import agg_feature
from common.feature_aggregation.default import agg_feature_default
from information_coefficient.preprocess.corr_cols import IC, corr_column
import pandas as pd
import numpy as np


def feature_calcuration(df, IC_scores_list, IC_scores_df, target_dict):
    # ===================
    # ryotaro made
    # ===================
    df_result = pd.DataFrame()

    # sma_1
    data, feature_columns = agg_feature_default.sma_1(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # sma_2
    data, feature_columns = agg_feature_default.sma_2(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # highest_1
    data, feature_columns = agg_feature_default.highest_1(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # highest_2
    data, feature_columns = agg_feature_default.highest_2(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # low_1
    data, feature_columns = agg_feature_default.low_1(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # hilo_ratio_1
    data, feature_columns = agg_feature_default.hilo_ratio_1(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # amount
    data, feature_columns = agg_feature_default.amount(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # adosc
    data, feature_columns = agg_feature_default.adosc(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # typical
    data, feature_columns = agg_feature_default.typical(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # log_return
    data, feature_columns = agg_feature_default.log_return(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pin_bar
    data, feature_columns = agg_feature_default.pin_bar(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    # Overlap Studies
    # ===================
    # MIDPRICE
    data, feature_columns = agg_feature.MIDPRICE(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # BBANDS
    data, feature_columns = agg_feature.BBANDS(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # DEMA
    data, feature_columns = agg_feature.DEMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # HT_TRENDLINE
    data, feature_columns = agg_feature.HT_TRENDLINE(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # KAMA
    data, feature_columns = agg_feature.KAMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # MAMA
    data, feature_columns = agg_feature.MAMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TEMA
    data, feature_columns = agg_feature.TEMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TRIMA
    data, feature_columns = agg_feature.TRIMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # WMA
    data, feature_columns = agg_feature.WMA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Momentum Indicator Functions
    # ===================

    # ADX
    data, feature_columns = agg_feature.ADX(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ADXR
    data, feature_columns = agg_feature.ADXR(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # APO
    data, feature_columns = agg_feature.APO(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # AROON
    data, feature_columns = agg_feature.AROON(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # AROONOSC
    data, feature_columns = agg_feature.AROONOSC(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # BOP
    data, feature_columns = agg_feature.BOP(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # CCI
    data, feature_columns = agg_feature.CCI(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # CMO
    data, feature_columns = agg_feature.CMO(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # MI
    data, feature_columns = agg_feature.MI(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # RO
    data, feature_columns = agg_feature.RO(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ROCR
    data, feature_columns = agg_feature.ROCR(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # # STOCH1
    # data, feature_columns = agg_feature.STOCH1(df.copy())
    # data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    # df_result = pd.concat([df_result, data], axis=1)

    # STOCH2
    data, feature_columns = agg_feature.STOCH2(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # STOCH3
    data, feature_columns = agg_feature.STOCH3(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TRIX
    data, feature_columns = agg_feature.TRIX(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Volatility Indicator Functions
    # ===================
    # ATR
    data, feature_columns = agg_feature.ATR(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Cycle Indicator Functions
    # ===================
    # HT_DCPERIOD
    data, feature_columns = agg_feature.HT_DCPERIOD(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # HT_DCPERIOD2
    data, feature_columns = agg_feature.HT_DCPERIOD2(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # sine
    data, feature_columns = agg_feature.sine(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Price Transform Functions
    # ===================
    # PRICE
    data, feature_columns = agg_feature.PRICE(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Pattern Recognition Functions
    # ===================

    # pattern_recognition2
    data, feature_columns = agg_feature.pattern_recognition2(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition3
    data, feature_columns = agg_feature.pattern_recognition3(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition4
    data, feature_columns = agg_feature.pattern_recognition4(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition5
    data, feature_columns = agg_feature.pattern_recognition5(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition6
    data, feature_columns = agg_feature.pattern_recognition6(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition7
    data, feature_columns = agg_feature.pattern_recognition7(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition8
    data, feature_columns = agg_feature.pattern_recognition8(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition9
    data, feature_columns = agg_feature.pattern_recognition9(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition12
    data, feature_columns = agg_feature.pattern_recognition12(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition13
    data, feature_columns = agg_feature.pattern_recognition13(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition14
    data, feature_columns = agg_feature.pattern_recognition14(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition15
    data, feature_columns = agg_feature.pattern_recognition15(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # pattern_recognition16
    data, feature_columns = agg_feature.pattern_recognition16(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Statistic Functions
    # ===================
    # LINEARREG
    data, feature_columns = agg_feature.LINEARREG(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # LINEARREG_SLOPE
    data, feature_columns = agg_feature.LINEARREG_SLOPE(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # VAR
    data, feature_columns = agg_feature.VAR(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TSF
    data, feature_columns = agg_feature.TSF(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # STDDEV
    data, feature_columns = agg_feature.STDDEV(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # LINEARREG_ANGLE
    data, feature_columns = agg_feature.LINEARREG_ANGLE(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # BETA
    data, feature_columns = agg_feature.BETA(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # CORREL
    data, feature_columns = agg_feature.CORREL(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Statistic Functions
    # ===================
    # ASIN
    data, feature_columns = agg_feature.ASIN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # ATAN
    data, feature_columns = agg_feature.ATAN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # CEIL
    data, feature_columns = agg_feature.CEIL(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # COS
    data, feature_columns = agg_feature.COS(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # COSH
    data, feature_columns = agg_feature.COSH(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # EXP
    data, feature_columns = agg_feature.EXP(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # FLOOR
    data, feature_columns = agg_feature.FLOOR(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # LN
    data, feature_columns = agg_feature.LN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # LOG10
    data, feature_columns = agg_feature.LOG10(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # SIN
    data, feature_columns = agg_feature.SIN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # SINH
    data, feature_columns = agg_feature.SINH(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TAN
    data, feature_columns = agg_feature.TAN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # TANH
    data, feature_columns = agg_feature.TANH(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    # ===================
    #  Math Operator Functions
    # ===================

    # SUB
    data, feature_columns = agg_feature.SUB(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # MULT
    data, feature_columns = agg_feature.MULT(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # DIV
    data, feature_columns = agg_feature.DIV(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # MAX
    data, feature_columns = agg_feature.MAX(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)
    # MIN
    data, feature_columns = agg_feature.MIN(df.copy())
    data = IC(data, feature_columns, IC_scores_list, IC_scores_df, target_dict)
    df_result = pd.concat([df_result, data], axis=1)

    return df_result


def calc_force_entry_price(entry_price=None, lo=None, pips=None):
    y = entry_price.copy()
    y[:] = np.nan
    force_entry_time = entry_price.copy()
    force_entry_time[:] = np.nan
    for i in range(entry_price.size):
        for j in range(i + 1, entry_price.size):
            if round(lo[j] / pips) < round(entry_price[j - 1] / pips):
                y[i] = entry_price[j - 1]
                force_entry_time[i] = j - i
                break
    return y, force_entry_time


def make_data(df, IC_scores_list, IC_scores_df, target_dict):

    df_result = feature_calcuration(df, IC_scores_list, IC_scores_df, target_dict)
    before_len = len(df_result)

    # 相関関係のある特徴量を削除
    features = corr_column(df_result, 0.8, target_dict["target_feature"], IC_scores_df)

    # target
    # df, target_list = target_calucuration(df, target_dict)
    df = pd.concat([df, df_result[features]], axis=1)
    df = df.round(12)
    df.dropna(inplace=True)

    embargo = before_len - len(df)
    print(f"embargo: {embargo}")

    return df, embargo, features
