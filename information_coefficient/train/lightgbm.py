from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sys

# import pandas as pd

sys.path.append("../../../")
from information_coefficient.evaluate.evaluate import richman_backtest
import numpy as np
from sklearn.model_selection import KFold
import os


def train_and_save(df, target_col, features, train_parameters, embargo):
    X = df[features]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, shuffle=False)
    X_train = X_train[:-embargo]
    y_train = y_train[:-embargo]

    train_data = lgb.Dataset(X_train, y_train)
    valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

    trained_model = lgb.train(train_parameters, train_data, num_boost_round=10000, valid_sets=[valid_data], early_stopping_rounds=100, verbose_eval=500,)

    # # model_save and load
    # trained_model.save_model(f"model/lgb_model_{target_col}.txt")
    # trained_model = lgb.Booster(model_file=f"model/lgb_model_{target_col}.txt")

    preds = trained_model.predict(X_valid)
    # ic = np.corrcoef(preds, y_valid)[0, 1]
    # plot_scatter(target_col, "feature", preds, y_valid, normalize=False)

    # backtest
    if "y_buy" in target_col:
        backtest_df = df[X_valid.index[0] : X_valid.index[-1]]
        backtest_df["y_pred_buy"] = preds
        return trained_model, backtest_df

    if "y_sell" in target_col:
        return trained_model, preds


def cross_validation(df, embargo, features, target_feature, train_parameters):
    y_pred = df[target_feature].values.copy()
    y_pred[:] = np.nan

    kf = KFold(n_splits=5)
    kf.get_n_splits(df)
    for train_index, test_index in kf.split(df):
        train_start = train_index[0]
        train_end = train_index[-1] - embargo
        test_start = test_index[0]
        test_end = test_index[-1]

        df_train, df_test = df.iloc[train_start:train_end, :], df.iloc[test_start:test_end, :]
        X_train = df_train[features]
        y_train = df_train[target_feature]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, shuffle=False)
        X_train = X_train[:-embargo]
        y_train = y_train[:-embargo]

        train_data = lgb.Dataset(X_train, y_train)
        valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

        trained_model = lgb.train(train_parameters, train_data, num_boost_round=10000, valid_sets=[valid_data], early_stopping_rounds=100, verbose_eval=500,)

        y_pred[test_start:test_end] = trained_model.predict(df_test[features])

    return y_pred


def lighgbm(df, features, embargo, target_dict, save_path):
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
    result_dict = richman_backtest(backtest_df, target_dict, train_test_split_save_path)

    # cross_validation
    df["y_pred_buy"] = cross_validation(df.copy(), embargo, features, target_dict["y_buy"], train_parameters)
    df["y_pred_sell"] = cross_validation(df.copy(), embargo, features, target_dict["y_sell"], train_parameters)
    df = df.dropna()
    df["predict_buy_entry"] = df["y_pred_buy"] > 0
    df["predict_sell_entry"] = df["y_pred_sell"] > 0
    df["priority_buy_entry"] = df["y_pred_buy"] > df["y_pred_sell"]
    cross_save_path = save_path + "/cross_validation"
    os.makedirs(cross_save_path, exist_ok=True)
    result_dict = richman_backtest(df, target_dict, cross_save_path)

    return result_dict
