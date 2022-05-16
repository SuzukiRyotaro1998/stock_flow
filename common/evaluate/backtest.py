import logging
import numba
import numpy as np

logger = logging.getLogger("Evaluate_Logger")


@numba.njit
def backtest(
    close: np.ndarray,
    predict_buy_entry: np.ndarray,
    predict_sell_entry: np.ndarray,
    priority_buy_entry: np.ndarray,
    buy_executed: np.ndarray,
    sell_executed: np.ndarray,
    buy_price: np.ndarray,
    sell_price: np.ndarray,
    losscut_rate: float,
    fee_percent: float,
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
        # if (100 + losscut_rate) / 100 * sell_entry_price < close[i] and np.maximum(0, -prev_pos) == 1:
        #     buy_exit_price = close[i]
        #     buy_exit_prices[i] = close[i]
        #     ret -= (buy_exit_price / sell_entry_price - 1) * np.maximum(0, -prev_pos)
        #     pos += np.maximum(0, -prev_pos)
        #     y[i] = ret
        #     poss[i] = pos

        #     # print("損切り")
        #     continue

        # if (100 - losscut_rate) / 100 * buy_entry_price > close[i] and np.maximum(0, prev_pos) == 1:
        #     sell_exit_price = close[i]
        #     sell_exit_prices[i] = close[i]
        #     ret += (sell_exit_price / buy_entry_price - 1) * np.maximum(0, prev_pos)
        #     pos -= np.maximum(0, prev_pos)
        #     y[i] = ret
        #     poss[i] = pos

        #     # print("損切り")
        #     continue

        # exit
        # Exit of short
        if buy_executed[i] and predict_buy_entry[i]:
            vol = np.maximum(0, -prev_pos)
            if vol == 1:
                buy_exit_price = buy_price[i]
                buy_exit_prices[i] = buy_price[i]
                ret -= (buy_exit_price / sell_entry_price - 1) * vol - fee_percent * 2
                pos += vol

        # Exit of long
        if sell_executed[i] and predict_sell_entry[i]:
            vol = np.maximum(0, prev_pos)
            if vol == 1:
                sell_exit_price = sell_price[i]
                sell_exit_prices[i] = sell_price[i]
                ret += (sell_exit_price / buy_entry_price - 1) * vol - fee_percent * 2
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


@numba.njit
# エントリーとexitの判断が別の指標で行われる場合
def backtest_ver2(
    close: np.ndarray,
    predict_buy_entry: np.ndarray,
    predict_sell_entry: np.ndarray,
    priority_buy_entry: np.ndarray,
    buy_executed_entry: np.ndarray,
    sell_executed_entry: np.ndarray,
    buy_price_entry: np.ndarray,
    sell_price_entry: np.ndarray,
    predict_short_exit: np.ndarray,
    predict_long_exit: np.ndarray,
    short_executed_exit: np.ndarray,
    long_executed_exit: np.ndarray,
    buy_price_exit: np.ndarray,
    sell_price_exit: np.ndarray,
    losscut_rate: float,
    horizon_barrier: int,
    fee_percent: float,
):
    n = close.size
    y = close.copy() * 0.0
    poss = close.copy() * 0.0
    ret = 0.0
    pos = 0.0
    buy_entry_price = 1.0
    sell_entry_price = 1.0
    buy_entry_date = 0
    sell_entry_date = 0
    buy_exit_date = 0
    sell_exit_date = 0
    buy_entry_prices = np.full((n), np.nan)
    sell_entry_prices = np.full((n), np.nan)
    buy_exit_prices = np.full((n), np.nan)
    sell_exit_prices = np.full((n), np.nan)
    for i in range(n):
        prev_pos = pos
        # if (100 + losscut_rate) / 100 * sell_entry_price < close[i] and np.maximum(0, -prev_pos) == 1:
        #     buy_exit_price = close[i]
        #     buy_exit_prices[i] = close[i]
        #     ret -= (buy_exit_price / sell_entry_price - 1) * np.maximum(0, -prev_pos)
        #     pos += np.maximum(0, -prev_pos)
        #     y[i] = ret
        #     poss[i] = pos

        #     # print("損切り")
        #     continue

        # if (100 - losscut_rate) / 100 * buy_entry_price > close[i] and np.maximum(0, prev_pos) == 1:
        #     sell_exit_price = close[i]
        #     sell_exit_prices[i] = close[i]
        #     ret += (sell_exit_price / buy_entry_price - 1) * np.maximum(0, prev_pos)
        #     pos -= np.maximum(0, prev_pos)
        #     y[i] = ret
        #     poss[i] = pos

        #     # print("損切り")
        #     continue

        # exit
        # Exit of short
        if short_executed_exit[i] and predict_short_exit[i] and i - sell_entry_date >= horizon_barrier:
            vol = np.maximum(0, -prev_pos)
            if vol == 1:
                buy_exit_price = buy_price_exit[i]
                buy_exit_prices[i] = buy_price_exit[i]
                ret -= (buy_exit_price / sell_entry_price - 1) * vol - fee_percent * 2
                pos += vol

        # Exit of long
        if long_executed_exit[i] and predict_long_exit[i] and i - buy_entry_date >= horizon_barrier:
            vol = np.maximum(0, prev_pos)
            if vol == 1:
                sell_exit_price = sell_price_exit[i]
                sell_exit_prices[i] = sell_price_exit[i]
                ret += (sell_exit_price / buy_entry_price - 1) * vol - fee_percent * 2
                pos -= vol

        # entry
        if priority_buy_entry[i] and predict_buy_entry[i] and buy_executed_entry[i]:
            vol = np.minimum(1.0, 1 - prev_pos) * predict_buy_entry[i]
            # ret -= buy_cost[i] * vol
            pos += vol
            if vol == 1:
                buy_entry_price = buy_price_entry[i]
                buy_entry_prices[i] = buy_price_entry[i]
                buy_entry_date = i

        if not priority_buy_entry[i] and predict_sell_entry[i] and sell_executed_entry[i]:
            vol = np.minimum(1.0, prev_pos + 1) * predict_sell_entry[i]
            pos -= vol
            if vol == 1:
                sell_entry_price = sell_price_entry[i]
                sell_entry_prices[i] = sell_price_entry[i]
                sell_entry_date = i

        y[i] = ret
        poss[i] = pos

    return y, poss, buy_entry_prices, sell_entry_prices, buy_exit_prices, sell_exit_prices


# richmanbtc backtest
@numba.njit
def backtest_richman(
    close: np.ndarray,
    predict_buy_entry: np.ndarray,
    predict_sell_entry: np.ndarray,
    priority_buy_entry: np.ndarray,
    buy_executed: np.ndarray,
    sell_executed: np.ndarray,
    buy_cost: np.ndarray,
    sell_cost: np.ndarray,
):
    n = close.size
    y = close.copy() * 0.0
    poss = close.copy() * 0.0
    ret = 0.0
    pos = 0.0
    for i in range(n):
        prev_pos = pos

        # exit
        if buy_executed[i]:
            vol = np.maximum(0, -prev_pos)
            ret -= buy_cost[i] * vol
            pos += vol

        if sell_executed[i]:
            vol = np.maximum(0, prev_pos)
            ret -= sell_cost[i] * vol
            pos -= vol

        # entry
        if priority_buy_entry[i] and predict_buy_entry[i] and buy_executed[i]:
            vol = np.minimum(1.0, 1 - prev_pos) * predict_buy_entry[i]
            ret -= buy_cost[i] * vol
            pos += vol

        if not priority_buy_entry[i] and predict_sell_entry[i] and sell_executed[i]:
            vol = np.minimum(1.0, prev_pos + 1) * predict_sell_entry[i]
            ret -= sell_cost[i] * vol
            pos -= vol

        if i + 1 < n:
            ret += pos * (close[i + 1] / close[i] - 1)

        y[i] = ret
        poss[i] = pos

    return y, poss
