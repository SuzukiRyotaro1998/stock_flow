from sklearn.model_selection import train_test_split
import sys
import pandas as pd

sys.path.append("../../../")
from information_coefficient.evaluate.evaluate import richman_backtest
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb


def train_and_save(df, target_col, features, train_parameters, embargo):
    X = df[features]
    y = df[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, shuffle=False)
    X_train = X_train[:-embargo]
    y_train = y_train[:-embargo]

    train_data = xgb.DMatrix(X_train, label=y_train)
    valid_data = xgb.DMatrix(X_valid, label=y_valid)

    results_dict = {}
    trained_model = xgb.train(
        train_parameters,
        train_data,  # 訓練データ
        num_boost_round=10000,  # 設定した学習回数
        early_stopping_rounds=100,
        evals=[(train_data, "train"), (valid_data, "valid")],
        evals_result=results_dict,
    )

    # 学習曲線の表示
    # lgb.plot_metric(trained_model)
    # plt.plot(results_dict["train"]["rmse"], color="red", label="train")
    # plt.plot(results_dict["valid"]["rmse"], color="blue", label="valid")
    # plt.legend()
    # plt.show()
    # plt.close()

    preds = trained_model.predict(valid_data)

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

        df_train, df_test = df.iloc[train_index[0] : train_index[-1] - embargo, :], df.iloc[test_index[0] : test_index[-1], :]
        X_train = df_train[features]
        y_train = df_train[target_feature]

        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, shuffle=False)
        X_train = X_train[:-embargo]
        y_train = y_train[:-embargo]

        train_data = xgb.DMatrix(X_train, label=y_train)
        valid_data = xgb.DMatrix(X_valid, label=y_valid)

        results_dict = {}
        trained_model = xgb.train(
            train_parameters,
            train_data,  # 訓練データ
            num_boost_round=100,  # 設定した学習回数
            early_stopping_rounds=10,
            evals=[(train_data, "train"), (valid_data, "valid")],
            evals_result=results_dict,
        )

        y_pred[test_index[0] : test_index[-1]] = trained_model.predict(xgb.DMatrix(df_test[features]))

    return y_pred


def xgboost(df, features, embargo, target_dict, save_path):
    train_parameters = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "boster": "rf",
        "random_state": 42,
    }

    df = pd.read_csv("ml_base/EDA/short_experiment/a.csv", index_col=0)
    df = df.rename(
        columns={
            "y_buy": "y_buy_5",
            "y_sell": "y_sell_5",
            "buy_executed": "buy_executed_5",
            "sell_executed": "sell_executed_5",
            "buy_price": "buy_price_5",
            "sell_price": "sell_price_5",
        }
    )
    print(df)

    buy_model, backtest_df = train_and_save(df.copy(), target_dict["y_buy"], features, train_parameters, embargo)
    sell_model, valid_sell_preds = train_and_save(df.copy(), target_dict["y_sell"], features, train_parameters, embargo)

    backtest_df["y_pred_sell"] = valid_sell_preds
    print(backtest_df)

    # validation backtest
    # 予測値の計算
    backtest_df["predict_buy_entry"] = backtest_df["y_pred_buy"] > 0
    backtest_df["predict_sell_entry"] = backtest_df["y_pred_sell"] > 0
    backtest_df["priority_buy_entry"] = backtest_df["y_pred_buy"] > backtest_df["y_pred_sell"]
    result_dict = richman_backtest(backtest_df, target_dict, save_path)

    # cross_validation
    df["y_pred_buy"] = cross_validation(df.copy(), embargo, features, target_dict["y_buy"], train_parameters)
    df["y_pred_sell"] = cross_validation(df.copy(), embargo, features, target_dict["y_sell"], train_parameters)
    df = df.dropna()
    df["predict_buy_entry"] = df["y_pred_buy"] > 0
    df["predict_sell_entry"] = df["y_pred_sell"] > 0
    df["priority_buy_entry"] = df["y_pred_buy"] > df["y_pred_sell"]

    result_dict = richman_backtest(df, target_dict, save_path)
    return result_dict

