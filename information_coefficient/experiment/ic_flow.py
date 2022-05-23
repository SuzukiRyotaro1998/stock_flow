import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append("../../")
from common.feature_aggregation.talib import agg_feature

from common.constants import DATAFOLDER
from common.feature_aggregation.features import highest, sma, std
from common.feature_aggregation.default import agg_feature_default
from information_coefficient.preprocess.corr_cols import target_calucuration

logging.basicConfig(level=logging.INFO)

def load_data():
    topix_code = pd.read_csv('../../data/stock_code/TOPIX_weight_jp.csv',index_col=0)
    topix_code = topix_code.sort_values('TOPIXに占める個別銘柄のウェイト',ascending=False)

    topix_code = topix_code['銘柄名'].values.tolist()

    candle_data_path = "../../data/raw_data/candle_data/TOPIX/"
    df = pd.DataFrame()
    for code in topix_code[1:1000]:
        try:
            data = pd.read_csv(candle_data_path+code+'.csv',index_col=0, engine='python')
            data = data.rename(columns={'Open': 'open', 'High': 'high','Low': 'low','Close': 'cl','Adj Close':'close','Volume': 'volume'})
            data, target_list = target_calucuration(data)
            df = pd.concat([df,data],axis=0)
        except:
            continue

    print(df)

    return df,target_list


def calcurate_ic(x, returns, normalize=False):
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


def IC(df, feature_columns, target_list):
    features = feature_columns.copy()
    for feature in features:
        # print(feature)
        df[f"{feature}_close_ratio"] = df["close"] / df[feature]
        df[f"{feature}_tilt"] = df[feature].copy() / df[feature].shift(1)
        df[f"{feature}_acceleration"] = df[f"{feature}_tilt"] / df[f"{feature}_tilt"].shift(1)

        feature_columns += [f"{feature}_close_ratio", f"{feature}_tilt", f"{feature}_acceleration"]

        for i in [5, 15, 30]:
            df[f"{feature}_highest_{i}"] = highest(df.copy(), feature, i)
            df[f"close_{feature}_highest_{i}_ratio"] = df["close"] / df[f"{feature}_highest_{i}"]
            feature_columns += [f"close_{feature}_highest_{i}_ratio"]

            df[f"{feature}_mean{i}"] = sma(df, feature, i)
            df[f"close_{feature}_mean_{i}_ratio"] = df["close"] / df[f"{feature}_mean{i}"]
            feature_columns += [f"close_{feature}_mean_{i}_ratio"]

            df[f"{feature}_std{i}"] = std(df, feature, i)
            df[f"close_{feature}_std_{i}_ratio"] = df["close"] / df[f"{feature}_std{i}"]
            feature_columns += [f"close_{feature}_std_{i}_ratio"]
            feature_columns += [f"{feature}_std{i}"]


    df_result = pd.DataFrame()
    for i in feature_columns:
        target_feature_list = []
        scores = []

        if df[i].isna().sum() > 300:
            # print(f"{i} too many Nan values")
            continue

        # infの数が多い特徴量は削除
        number_inf = df[i][(df[i] == np.inf) | (df[i] == -np.inf)].count()
        if number_inf > 0:
            # print(f"{i} has inf values")
            continue

        for tar in target_list:
            for target_feature in [tar["y_buy"], tar["y_sell"]]:
                df_ic = df.replace([np.inf, -np.inf], np.nan)
                df_ic = df_ic[[i, target_feature]]
                df_ic = df_ic.dropna()
                ic = calcurate_ic(df_ic[i].values, df_ic[target_feature].values)
                scores.append(ic)
                target_feature_list.append(target_feature)

        df_result_fragment = pd.DataFrame([scores], index=[i], columns=target_feature_list)
        df_result = pd.concat([df_result, df_result_fragment])

    print(df_result)
    return df_result


def main():

    print("==========================")

    df,target_list = load_data()
    df_result_total = pd.DataFrame()

    # ===================
    # ryotaro made
    # ===================
    # sma_1
    data, feature_columns = agg_feature_default.sma_1(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # sma_2
    data, feature_columns = agg_feature_default.sma_2(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # highest_1
    data, feature_columns = agg_feature_default.highest_1(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # highest_2
    data, feature_columns = agg_feature_default.highest_2(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # low_1
    data, feature_columns = agg_feature_default.low_1(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # hilo_ratio_1
    data, feature_columns = agg_feature_default.hilo_ratio_1(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # amount
    data, feature_columns = agg_feature_default.amount(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # adosc
    data, feature_columns = agg_feature_default.adosc(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # typical
    data, feature_columns = agg_feature_default.typical(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # log_return
    data, feature_columns = agg_feature_default.log_return(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pin_bar
    data, feature_columns = agg_feature_default.pin_bar(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    # Overlap Studies
    # ===================

    # BBANDS
    data, feature_columns = agg_feature.BBANDS(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # DEMA
    data, feature_columns = agg_feature.DEMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # HT_TRENDLINE
    data, feature_columns = agg_feature.HT_TRENDLINE(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # KAMA
    data, feature_columns = agg_feature.KAMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MAMA
    data, feature_columns = agg_feature.MAMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MIDPRICE 分布が歪
    data, feature_columns = agg_feature.MIDPRICE(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TEMA
    data, feature_columns = agg_feature.TEMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TRIMA
    data, feature_columns = agg_feature.TRIMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # WMA
    data, feature_columns = agg_feature.WMA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Momentum Indicator Functions
    # ===================

    # ADX
    data, feature_columns = agg_feature.ADX(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ADXR
    data, feature_columns = agg_feature.ADXR(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # APO
    data, feature_columns = agg_feature.APO(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # AROON
    data, feature_columns = agg_feature.AROON(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # AROONOSC
    data, feature_columns = agg_feature.AROONOSC(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # BOP
    data, feature_columns = agg_feature.BOP(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # CCI
    data, feature_columns = agg_feature.CCI(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # CMO
    data, feature_columns = agg_feature.CMO(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MAC
    data, feature_columns = agg_feature.MAC(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MACDEXT
    data, feature_columns = agg_feature.MACDEXT(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MACDFIX
    data, feature_columns = agg_feature.MACDFIX(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # MI
    data, feature_columns = agg_feature.MI(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # RO
    data, feature_columns = agg_feature.RO(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ROCR
    data, feature_columns = agg_feature.ROCR(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # STOCH1
    data, feature_columns = agg_feature.STOCH1(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # STOCH2
    data, feature_columns = agg_feature.STOCH2(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # STOCH3
    data, feature_columns = agg_feature.STOCH3(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TRIX
    data, feature_columns = agg_feature.TRIX(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Volatility Indicator Functions
    # ===================
    # ATR
    data, feature_columns = agg_feature.ATR(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Cycle Indicator Functions
    # ===================
    # HT_DCPERIOD
    data, feature_columns = agg_feature.HT_DCPERIOD(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # HT_DCPERIOD2
    data, feature_columns = agg_feature.HT_DCPERIOD2(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # sine
    data, feature_columns = agg_feature.sine(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Price Transform Functions
    # ===================
    # PRICE
    data, feature_columns = agg_feature.PRICE(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Pattern Recognition Functions
    # ===================

    # pattern_recognition2
    data, feature_columns = agg_feature.pattern_recognition2(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition3
    data, feature_columns = agg_feature.pattern_recognition3(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition4
    data, feature_columns = agg_feature.pattern_recognition4(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition5
    data, feature_columns = agg_feature.pattern_recognition5(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition6
    data, feature_columns = agg_feature.pattern_recognition6(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition7
    data, feature_columns = agg_feature.pattern_recognition7(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition8
    data, feature_columns = agg_feature.pattern_recognition8(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition9
    data, feature_columns = agg_feature.pattern_recognition9(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition10
    # data, feature_columns = agg_feature.pattern_recognition10(df.copy())
    # df_result = IC(data, feature_columns, target_list)

    # # display(df_result)
    # df_result_total = pd.concat([df_result_total,df_result],ignore_index=True)

    # pattern_recognition11
    # data, feature_columns = agg_feature.pattern_recognition11(df.copy())
    # df_result = IC(data, feature_columns, target_list)

    # # display(df_result)
    # df_result_total = pd.concat([df_result_total,df_result],ignore_index=True)

    # pattern_recognition12
    data, feature_columns = agg_feature.pattern_recognition12(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition13
    data, feature_columns = agg_feature.pattern_recognition13(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition14
    data, feature_columns = agg_feature.pattern_recognition14(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition15
    data, feature_columns = agg_feature.pattern_recognition15(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # pattern_recognition16
    data, feature_columns = agg_feature.pattern_recognition16(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Statistic Functions
    # ===================
    # LINEARREG
    data, feature_columns = agg_feature.LINEARREG(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # LINEARREG_SLOPE
    data, feature_columns = agg_feature.LINEARREG_SLOPE(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # VAR
    data, feature_columns = agg_feature.VAR(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TSF
    data, feature_columns = agg_feature.TSF(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # STDDEV
    data, feature_columns = agg_feature.STDDEV(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # LINEARREG_ANGLE
    data, feature_columns = agg_feature.LINEARREG_ANGLE(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # BETA
    data, feature_columns = agg_feature.BETA(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # CORREL
    data, feature_columns = agg_feature.CORREL(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Statistic Functions
    # ===================
    # ASIN
    data, feature_columns = agg_feature.ASIN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # ATAN
    data, feature_columns = agg_feature.ATAN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # CEIL
    data, feature_columns = agg_feature.CEIL(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # COS
    data, feature_columns = agg_feature.COS(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # COSH
    data, feature_columns = agg_feature.COSH(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # EXP
    data, feature_columns = agg_feature.EXP(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # FLOOR
    data, feature_columns = agg_feature.FLOOR(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # LN
    data, feature_columns = agg_feature.LN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # LOG10
    data, feature_columns = agg_feature.LOG10(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # SIN
    data, feature_columns = agg_feature.SIN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # SINH
    data, feature_columns = agg_feature.SINH(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TAN
    data, feature_columns = agg_feature.TAN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # TANH
    data, feature_columns = agg_feature.TANH(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # ===================
    #  Math Operator Functions
    # ===================
    # # MINMAX
    # data, feature_columns = agg_feature.MINMAX(df.copy())
    # df_result = IC(data, feature_columns, target_list)

    # ADD
    data, feature_columns = agg_feature.ADD(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # SUM
    # data, feature_columns = agg_feature.SUM(df.copy())
    # df_result = IC(data, feature_columns, target_list)

    # SUB
    data, feature_columns = agg_feature.SUB(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # MULT
    data, feature_columns = agg_feature.MULT(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # DIV
    data, feature_columns = agg_feature.DIV(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # MAX
    data, feature_columns = agg_feature.MAX(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)
    # MIN
    data, feature_columns = agg_feature.MIN(df.copy())
    df_result = IC(data, feature_columns, target_list)
    df_result_total = pd.concat([df_result_total, df_result], ignore_index=False)

    # 結果のまとめ
    df_result_total.to_csv(f"IC_TOPIX.csv")
    print(df_result_total)

    # result_dict = dict(df_result_total[["feature", "IC_abs"].values)
    # mlflow.log_metrics(result_dict)
    # mlflow.log_artifacts(downstream)
    # logger.info(f"result info files have been saved in {downstream}")


if __name__ == "__main__":
    main()