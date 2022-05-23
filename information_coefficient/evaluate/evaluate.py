import numpy as np
import pandas as pd
from common.evaluate.evaluate import evaluation_index
import matplotlib.pyplot as plt
import os


def backtest(
    close: np.ndarray,
    predict_buy_entry: np.ndarray,
    predict_sell_entry: np.ndarray,
    priority_buy_entry: np.ndarray,
    buy_executed: np.ndarray,
    sell_executed: np.ndarray,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
):
    n = close.size
    y = close.copy() * 0.0
    poss = close.copy() * 0.0
    ret = 0.0
    pos = 0.0
    buy_entry_price = 1.0
    sell_entry_price = 1.0
    buy_entry_prices = np.full((n), np.nan)
    sell_entry_prices = np.full((n), np.nan)
    buy_exit_prices = np.full((n), np.nan)
    sell_exit_prices = np.full((n), np.nan)
    for i in range(n):
        prev_pos = pos

        # exit
        # Exit of short
        if buy_executed[i]:
            vol = np.maximum(0, -prev_pos)
            if vol == 1:
                buy_exit_price = buy_price[i]
                buy_exit_prices[i] = buy_price[i]
                ret -= (buy_exit_price / sell_entry_price - 1) * vol
                pos += vol

        # Exit of long
        if sell_executed[i]:
            vol = np.maximum(0, prev_pos)
            if vol == 1:
                sell_exit_price = sell_price[i]
                sell_exit_prices[i] = sell_price[i]
                ret += (sell_exit_price / buy_entry_price - 1) * vol
                pos -= vol

        # entry
        if priority_buy_entry[i] and predict_buy_entry[i] and buy_executed[i]:
            vol = np.minimum(1.0, 1 - prev_pos) * predict_buy_entry[i]
            # ret -= buy_cost[i] * vol
            pos += vol
            if vol == 1:
                buy_entry_price = buy_price[i]
                buy_entry_prices[i] = buy_price[i]

        if not priority_buy_entry[i] and predict_sell_entry[i] and sell_executed[i]:
            vol = np.minimum(1.0, prev_pos + 1) * predict_sell_entry[i]
            pos -= vol
            if vol == 1:
                sell_entry_price = sell_price[i]
                sell_entry_prices[i] = sell_price[i]

        y[i] = ret
        poss[i] = pos

    return y, poss, buy_entry_prices, sell_entry_prices, buy_exit_prices, sell_exit_prices


def richman_backtest(df, target_dict, save_path=None):
    # 予測値の計算
    df["predict_buy_entry"] = df["y_pred_buy"] > 0
    df["predict_sell_entry"] = df["y_pred_sell"] > 0
    # df["predict_sell_entry"] = False

    df["priority_buy_entry"] = df["y_pred_buy"] > df["y_pred_sell"]

    cumulative_return, possition, buy_entry_prices, sell_entry_prices, buy_exit_prices, sell_exit_prices = backtest(
        close=df["close"].to_numpy(),
        predict_buy_entry=df["predict_buy_entry"].to_numpy(),
        predict_sell_entry=df["predict_sell_entry"].to_numpy(),
        priority_buy_entry=df["priority_buy_entry"].to_numpy(),
        buy_executed=df[target_dict["buy_executed"]].to_numpy(),
        sell_executed=df[target_dict["sell_executed"]].to_numpy(),
        buy_price=df[target_dict["buy_price"]].to_numpy(),
        sell_price=df[target_dict["sell_price"]].to_numpy(),
    )

    df_result = pd.DataFrame(
        {
            "cumulative_return": cumulative_return,
            "position": possition,
            "buy_entry_price": buy_entry_prices,
            "sell_entry_price": sell_entry_prices,
            "buy_exit_price": buy_exit_prices,
            "sell_exit_price": sell_exit_prices,
        },
        index=df.index,
    )

    result_dict = evaluation_index(df_result)
    total_return = result_dict["Total return percent"]

    df_result["cumulative_return"].plot()
    plt.title(f"cumulative_return {total_return}%")
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{total_return}%.png"))
    plt.close()

    # print("ポジション推移です。変動が細かすぎて青色一色になっていると思います。")
    # print("ちゃんと全ての期間でトレードが発生しているので、正常です。")
    # df_result["position"].plot()
    # plt.title("position")
    # plt.show()

    # print("ポジションの平均の推移です。どちらかに偏りすぎていないかなどを確認できます。")
    # df_result["position"].rolling(1000).mean().plot()
    # plt.title("position_average")
    # plt.show()

    # print("取引量(ポジション差分の絶対値)の累積です。")
    # print("期間によらず傾きがだいたい同じなので、全ての期間でちゃんとトレードが行われていることがわかります。")
    # df_result["position"].diff(1).abs().dropna().cumsum().plot()
    # plt.title("cumulative_position")

    return result_dict
