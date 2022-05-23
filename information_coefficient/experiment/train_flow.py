import pandas as pd
import sys

sys.path.append("../../")
from common.constants import DATAFOLDER
import os
from information_coefficient.preprocess.make_data import make_data
from information_coefficient.preprocess.corr_cols import target_calucuration

from information_coefficient.train.lightgbm import lighgbm

# from information_coefficient.train.lasso import lasso
# from information_coefficient.train.xgboost import xgboost


def get_IC_scores_list(exchange_name: str, trading_type: str, pair_name: str, time_bar: str, target_feature: str):
    IC_scores_df = pd.read_csv(f"information_coefficient/result/IC_feature/{exchange_name}_{trading_type}_{pair_name}_{time_bar}.csv")
    IC_scores_df = IC_scores_df.drop_duplicates(subset="Unnamed: 0")
    IC_scores_df = IC_scores_df.set_index("Unnamed: 0")

    IC_scores_df = IC_scores_df.abs()
    IC_scores_df = IC_scores_df.sort_values(target_feature, ascending=False)

    # lasso回帰の候補とする特長量の個数を指定
    num_feature = 1000
    IC_scores_list = IC_scores_df.index.values.tolist()[:num_feature]
    return IC_scores_list, IC_scores_df


def train_flow(downstream, exchange_name: str, trading_type: str, pair_name: str, time_bar: str):

    print("==========================")
    print(f"{exchange_name}_{trading_type}_{pair_name}_{time_bar}")
    print("==========================")
    # i = 5
    # target_dict = {
    #     "target_feature": f"y_buy_{i}",
    #     "y_buy": f"y_buy_{i}",
    #     "y_sell": f"y_sell_{i}",
    #     "buy_executed": f"buy_executed_{i}",
    #     "sell_executed": f"sell_executed_{i}",
    #     "buy_price": f"buy_price_{i}",
    #     "sell_price": f"sell_price_{i}",
    # }

    os.makedirs(downstream, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATAFOLDER.ohlc_data_folder, time_bar, f"{exchange_name}_{trading_type}_{pair_name}.parquet.gzip"), engine="pyarrow",)
    save_path = f"information_coefficient/result/{exchange_name}/{trading_type}_{pair_name}_{time_bar}"
    os.makedirs(save_path, exist_ok=True)

    df = df["2019/01/01 00:00:00":]
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")
    df = df[-1000000:]

    df, target_list = target_calucuration(df.copy())

    for target_dict in target_list:
        print("==========================")
        print(target_dict)
        IC_scores_list, IC_scores_df = get_IC_scores_list(exchange_name, trading_type, pair_name, time_bar, target_dict["target_feature"])
        data, embargo, features = make_data(df.copy(), IC_scores_list, IC_scores_df, target_dict)
        print(features)
        print(len(features))
        # lasso(df, features, embargo, target_dict, save_path)
        # try:
        lighgbm(data, features, embargo, target_dict, save_path)
        # except:
        #     print('error')
        #     continue
        # xgboost(df, features, embargo, target_dict, save_path)
