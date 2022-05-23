import pandas as pd
from common.feature_aggregation.features import highest, sma, std
import numpy as np
import sys

sys.path.append("../../../")

def corr_column(df, threshold, target_feature, IC_scores_df):
    df = df.dropna()

    df_corr = df.corr()
    df_corr = abs(df_corr)

    # 対角線の値を0にする
    for i in range(len(df_corr.index)):
        for j in range(len(df_corr.columns)):
            if i == j:
                df_corr.iloc[i, j] = 0

    while True:
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        # ヒートマップをプロット
        # plt.figure(figsize = (25,25))
        # sns.heatmap(df_corr, cmap='Blues')

        if max_corr < threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None

            query_ic = IC_scores_df.loc[query_column, target_feature]
            target_ic = IC_scores_df.loc[target_column, target_feature]

            # # その他との相関の絶対値が大きい方を除去
            # if sum(df_corr[query_column]) <= sum(df_corr[target_column]):
            #     delete_column = target_column
            #     saved_column = query_column

            # # 情報係数が低い方を削除
            if query_ic < target_ic:
                delete_column = query_column
            else:
                delete_column = target_column

            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)
    return sorted(df_corr.columns.to_list())


def IC(df, feature_columns, IC_scores_list, IC_scores_df, target_dict):
    df_result = pd.DataFrame()
    features = feature_columns.copy()
    df, target_list = target_calucuration(df)

    for feature in features:
        candidates = [feature]
        r = [5, 15, 30]
        candidates += [f"{feature}_close_ratio", f"{feature}_tilt", f"{feature}_acceleration"]
        candidates += [f"close_{feature}_highest_{term}_ratio" for term in range(5, 35, 5) if term in r]
        candidates += [f"close_{feature}_mean_{term}_ratio" for term in range(5, 35, 5) if term in r]
        candidates += [f"close_{feature}_std_{term}_ratio" for term in range(5, 35, 5) if term in r]

        for i in candidates:
            if i in IC_scores_list:
                if i == f"{feature}_close_ratio":
                    df[f"{feature}_close_ratio"] = df["close"] / df[feature]

                elif i == f"{feature}_tilt" or i == f"{feature}_acceleration":
                    df[f"{feature}_tilt"] = df[feature] / df[feature].shift(1)
                    df[f"{feature}_acceleration"] = df[f"{feature}_tilt"] / df[f"{feature}_tilt"].shift(1)

                elif i in [f"close_{feature}_highest_{term}_ratio" for term in range(5, 35, 5) if term in r] or i in [
                    f"close_{feature}_highest_{term}_ratio" for term in range(5, 35, 5) if term in r
                ]:
                    for term in [5, 15, 30]:
                        df[f"{feature}_highest_{term}"] = highest(df.copy(), feature, term)
                        df[f"close_{feature}_highest_{term}_ratio"] = df["close"] / df[f"{feature}_highest_{term}"]

                elif i in [f"{feature}_mean{term}" for term in range(5, 35, 5) if term in r] or i in [
                    f"close_{feature}_mean_{term}_ratio" for term in range(5, 35, 5) if term in r
                ]:
                    for term in [5, 15, 30]:
                        df[f"{feature}_mean{term}"] = sma(df, feature, term)
                        df[f"close_{feature}_mean_{term}_ratio"] = df["close"] / df[f"{feature}_mean{term}"]

                elif i in [f"{feature}_std{term}" for term in range(5, 35, 5) if term in r] or i in [
                    f"close_{feature}_std_{term}_ratio" for term in range(5, 35, 5) if term in r
                ]:
                    for term in [5, 15, 30]:
                        df[f"{feature}_std{term}"] = std(df, feature, term)
                        df[f"close_{feature}_std_{term}_ratio"] = df["close"] / df[f"{feature}_std{term}"]
                    # print('std の計算コストは高いため採用しない')
                    continue

                # nanの数が多い特徴量は削除
                if df[i].isna().sum() > 300:
                    print(f"{i} has too many Nan values")
                    continue

                # infの数が多い特徴量は削除
                number_inf = df[i][(df[i] == np.inf) | (df[i] == -np.inf)].count()
                if number_inf > 0:
                    print(f"{i} has inf values")
                    continue

                df_result = pd.concat([df_result, df[[i]]], axis=1)

    if len(df_result) != 0:
        columns = corr_column(df_result, 0.8, target_dict["target_feature"], IC_scores_df)
        df_result = df_result[columns]

    return df_result


def target_calucuration(df):
    target_list = []

    for i in range(10):
        df[f"y_buy_{i}"]= (df['open'].shift(-1-i)/ df['open'].shift(-1) - 1) * 100
        df[f"y_sell_{i}"]= (1 - df['open'].shift(-1-i)/ df['open'].shift(-1)) * 100
        df[f"buy_executed_{i}"]= 1
        df[f"sell_executed_{i}"]= 1
        df[f"buy_price_{i}"]= df['open'].shift(-1)
        df[f"sell_price_{i}"]= df['open'].shift(-1)

        target_dict = {
            "target_feature": f"y_buy_{i}",
            "y_buy": f"y_buy_{i}",
            "y_sell": f"y_sell_{i}",
            "buy_executed": f"buy_executed_{i}",
            "sell_executed": f"sell_executed_{i}",
            "buy_price": f"buy_price_{i}",
            "sell_price": f"sell_price_{i}",
        }
        target_list.append(target_dict)


    return df, target_list
