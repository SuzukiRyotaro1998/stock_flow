import pandas as pd
import sys
import random

sys.path.append("../../")
from common.constants import DATAFOLDER
import os
from information_coefficient.preprocess.make_data import feature_calcuration, corr_column
from information_coefficient.preprocess.corr_cols import target_calucuration
from information_coefficient.evaluate.evaluate import richman_backtest
from information_coefficient.train.lightgbm import train_and_save
import pickle


def lighgbm(df, features, embargo, target_dict, save_path, ex_num_return):
    train_parameters = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "rf",
        "lambda_l1": 0.1463894505800834,
        "lambda_l2": 3.8628956682033713,
        "num_leaves": 240,
        "feature_fraction": 0.803334859033497,
        "bagging_fraction": 0.9502934982338573,
        "bagging_freq": 6,
        "min_child_samples": 96,
        "learning_rate": 0.08362636819642999,
        "max_depth": 490,
    }

    buy_model, backtest_df = train_and_save(df.copy(), target_dict["y_buy"], features, train_parameters, embargo)
    sell_model, valid_sell_preds = train_and_save(df.copy(), target_dict["y_sell"], features, train_parameters, embargo)

    backtest_df["y_pred_sell"] = valid_sell_preds
    # validation backtest
    # 予測値の計算
    backtest_df["predict_buy_entry"] = backtest_df["y_pred_buy"] > 0
    backtest_df["predict_sell_entry"] = backtest_df["y_pred_sell"] > 0
    backtest_df["priority_buy_entry"] = backtest_df["y_pred_buy"] > backtest_df["y_pred_sell"]
    train_test_split_save_path = save_path + "/train_test_split"
    os.makedirs(train_test_split_save_path, exist_ok=True)
    split_result_dict = richman_backtest(backtest_df.copy(), target_dict, None)

    # # cross_validation
    # df["y_pred_buy"] = cross_validation(df.copy(), embargo, features, target_dict["y_buy"], train_parameters)
    # df["y_pred_sell"] = cross_validation(df.copy(), embargo, features, target_dict["y_sell"], train_parameters)
    # df = df.dropna()
    # df["predict_buy_entry"] = df["y_pred_buy"] > 0
    # df["predict_sell_entry"] = df["y_pred_sell"] > 0
    # df["priority_buy_entry"] = df["y_pred_buy"] > df["y_pred_sell"]
    # cross_save_path = save_path + "/cross_validation"
    # os.makedirs(cross_save_path, exist_ok=True)
    # result_dict = richman_backtest(df, target_dict, None)

    if split_result_dict["Total return percent"] > ex_num_return:
        split_result_dict = richman_backtest(backtest_df.copy(), target_dict, train_test_split_save_path)
        # result_dict = richman_backtest(df, target_dict, cross_save_path)
        print("バックテストのリターンが最高を更新")
        print(f'train_test_split return : {split_result_dict["Total return percent"]}%')
        # print(f'cross_validation return : {result_dict["Total return percent"]}%')

        ex_num_return = split_result_dict["Total return percent"]

        return ex_num_return, True

    else:
        return ex_num_return, False


def get_IC_scores_list(exchange_name: str, trading_type: str, pair_name: str, time_bar: str, target_feature: str):
    IC_scores_df = pd.read_csv(f"information_coefficient/result/IC_feature/{exchange_name}_{trading_type}_{pair_name}_{time_bar}.csv")
    IC_scores_df = IC_scores_df.drop_duplicates(subset="Unnamed: 0")
    IC_scores_df = IC_scores_df.set_index("Unnamed: 0")

    IC_scores_df = IC_scores_df.abs()
    IC_scores_df = IC_scores_df.sort_values(target_feature, ascending=False)

    # lasso回帰の候補とする特長量の個数を指定
    num_feature = 2000
    IC_scores_list = IC_scores_df.index.values.tolist()[:num_feature]
    return IC_scores_list, IC_scores_df


def make_data(df, IC_scores_list, IC_scores_df, target_dict):

    df_result = feature_calcuration(df, IC_scores_list, IC_scores_df, target_dict)
    # 相関関係のある特徴量を削除
    features = corr_column(df_result, 0.8, target_dict["target_feature"], IC_scores_df)

    return df_result, features


def swap_feature(downstream, exchange_name: str, trading_type: str, pair_name: str, time_bar: str):

    print("==========================")
    print(f"{exchange_name}_{trading_type}_{pair_name}_{time_bar}")
    print("==========================")
    i = 3

    target_dict = {
        "target_feature": f"y_buy_{i}",
        "y_buy": f"y_buy_{i}",
        "y_sell": f"y_sell_{i}",
        "buy_executed": f"buy_executed_{i}",
        "sell_executed": f"sell_executed_{i}",
        "buy_price": f"buy_price_{i}",
        "sell_price": f"sell_price_{i}",
    }

    os.makedirs(downstream, exist_ok=True)
    df = pd.read_parquet(os.path.join(DATAFOLDER.ohlc_data_folder, time_bar, f"{exchange_name}_{trading_type}_{pair_name}.parquet.gzip"), engine="pyarrow",)
    save_path = f"information_coefficient/result/{exchange_name}/{trading_type}_{pair_name}_{time_bar}"
    os.makedirs(save_path, exist_ok=True)

    df = df["2019/01/01 00:00:00":]
    df = df.drop_duplicates()
    df = df.fillna(method="ffill")
    df = df[-2000000:]

    IC_scores_list, IC_scores_df = get_IC_scores_list(exchange_name, trading_type, pair_name, time_bar, target_dict["target_feature"])
    df_result, features = make_data(df, IC_scores_list, IC_scores_df, target_dict)
    print(features)
    print(len(features))

    before_len = len(df)
    # target
    df, target_list = target_calucuration(df)

    ex_num_return = 0
    best_trial = 0
    for i in range(50000):
        new_columns = sorted(random.sample(features, random.randrange(60, 100)))

        print(f"=====trial{i}=====")
        data = pd.concat([df, df_result[new_columns]], axis=1)
        data = data.round(12)
        data.dropna(inplace=True)
        embargo = before_len - len(data)
        print(f"embargo: {embargo}")

        print(f"number of features :{len(new_columns)}")
        ex_num_return, is_best_trial = lighgbm(data, new_columns, embargo, target_dict, save_path, ex_num_return)
        if is_best_trial:
            best_trial = i
            with open("test", "wb") as fp:  # Pickling
                pickle.dump(new_columns, fp)
            with open("test", "rb") as fp:  # Unpickling
                new_columns = pickle.load(fp)
                print("特徴量")
                print(new_columns)

        print(f"best_trial is trial:{best_trial}, num_return:{ex_num_return}")

