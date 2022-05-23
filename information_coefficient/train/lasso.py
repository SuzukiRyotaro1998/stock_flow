import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

import pandas as pd

sys.path.append("../../../")
from information_coefficient.evaluate.evaluate import richman_backtest
import numpy as np
from sklearn.model_selection import KFold


def lasso_(X, y, visualize_type):
    # グリッドサーチを行うためのパラメーター
    param = [{"alpha": [0.001, 0.01, 0.1, 1, 10]}]

    gs_lasso = GridSearchCV(estimator=Lasso(random_state=1), param_grid=param, cv=10, n_jobs=-1)
    gs_lasso = gs_lasso.fit(X, y)

    # 裁量スコアとなるパラメータ値を出力
    print(gs_lasso.best_params_)
    return gs_lasso


def cross_validation(df, embargo, features, target_feature):
    y_pred = df[target_feature].values.copy()
    y_pred[:] = np.nan

    kf = KFold(n_splits=5)
    kf.get_n_splits(df)
    for train_index, test_index in kf.split(df):
        train_start = train_index[0]
        train_end = train_index[-1] - embargo
        test_start = test_index[0]
        test_end = test_index[-1]

        df_train, df_valid = df.iloc[train_start:train_end, :], df.iloc[test_start:test_end, :]

        # 標準化
        scaler = StandardScaler()
        scaler.fit(df_train)
        df_train = pd.DataFrame(scaler.transform(df_train), columns=df_train.columns)
        df_valid = pd.DataFrame(scaler.transform(df_valid), columns=df_valid.columns)

        X_train = df_train[features]
        y_train = df_train[target_feature]
        model = lasso_(X_train, y_train, False)

        X_valid = df_valid[features]
        start = test_index[0]
        end = test_index[-1]
        y_pred[start:end] = model.predict(X_valid)

    return y_pred


def lasso(df, features, embargo, target_dict, save_path):
    df["y_pred_buy"] = cross_validation(df.copy(), embargo, features, target_dict["y_buy"])
    df["y_pred_sell"] = cross_validation(df.copy(), embargo, features, target_dict["y_sell"])
    df = df.dropna()
    print(df)
    # ic = np.corrcoef(df["y_pred_buy"], df["y_buy_5"])[0, 1]
    # plot_result(df["y_pred_buy"], df["y_buy_5"], ic)

    # print("result of sell pred")
    # ic = np.corrcoef(df["y_pred_sell"], df["y_sell_5"])[0, 1]
    # plot_result(df["y_pred_sell"], df["y_sell_5"], ic)

    result_dict = richman_backtest(df, target_dict, save_path)

    return result_dict
