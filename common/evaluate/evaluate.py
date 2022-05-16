import sys
import pandas as pd

sys.path.append("../../")
from common.evaluate.visualize import return_histgram
from prettytable import PrettyTable


def evaluation_index(df, downstream=None):
    df.index = pd.to_datetime(df.index)
    df["return"] = df["cumulative_return"].diff()
    # df = df.dropna()
    trading_period = float((df.index[-1] - df.index[0]).days)

    num_gains = (df["return"] > 0).sum()
    num_losses = (df["return"] < 0).sum()
    num_trades = num_gains + num_losses
    gains = df["return"][df["return"] > 0].sum()
    losses = df["return"][df["return"] < 0].sum()
    total_return = 100 * (gains + losses)
    average_gain = gains / num_gains
    average_loss = losses / num_losses
    max_return = 100 * df["return"].max()
    max_loss = 100 * df["return"].min()
    win_rate = 100 * num_gains / (num_gains + num_losses)
    lose_rate = 100 * num_losses / (num_gains + num_losses)

    drawdowns = df["cumulative_return"].cummax().subtract(df["cumulative_return"])
    drawdown = 100 * drawdowns.max()

    day_returns = 100 * df["return"].resample("D").sum()
    day_return = day_returns.mean()
    num_win_day = sum([i > 0 for i in day_returns])
    num_lose_day = sum([i < 0 for i in day_returns])
    win_rate_perday = 100 * num_win_day / (num_lose_day + num_win_day)
    df.loc[df["return"] != 0, "trade"] = 1
    df.loc[df["return"] == 0, "trade"] = 0
    day_num_trades = df["trade"].resample("D").sum()
    day_num_trade = day_num_trades.mean()

    month_returns = 100 * df["return"].resample("M").sum()
    month_return = month_returns.mean()
    num_win_month = sum([i > 0 for i in month_returns])
    num_lose_month = sum([i < 0 for i in month_returns])
    win_rate_permonth = 100 * num_win_month / (num_lose_month + num_win_month)

    # print("--------------------------------------")
    # print("The period is from {} up to {}".format(df.index[0], df.index[-1]))
    # print(f"Total return: {round(total_return,2)}%")
    # print("total days:{}".format(trading_period))
    # print(f"1day average return:{round(day_return,2)}%")
    # print(f"1day average num trades:{round(day_num_trade)}回")
    # print(f"1日の通算損益がプラスになる確率:{round(win_rate_perday,2)}%")
    # print(f"1month average return:{round(month_return,2)}%")
    # print(f"1ヶ月の通算損益がプラスになる確率:{round(win_rate_permonth,2)}%")
    # print(f"num Trades: {num_trades}")
    # print(f"Average Gain: {round(average_gain,4)}%")
    # print(f"Average Loss: {round(average_loss,4)}%")
    # print(f"Max Return: {round(max_return,2)}%")
    # print(f"Max Loss: {round(max_loss,2)}%")
    # print(f"最大ドローダウン: {round(drawdown,2)}%")
    # print("--------------------------------------")

    result_table = PrettyTable()
    result_table.field_names = ["evaluate index", "value"]
    result_table.add_row(["start", df.index[0]])
    result_table.add_row(["end", df.index[-1]])
    result_table.add_row(["Total return percent", round(total_return, 2)])
    result_table.add_row(["num trades", round(num_trades, 2)])
    result_table.add_row(["win rate percent", round(win_rate, 2)])
    result_table.add_row(["lose rate percent", round(lose_rate, 2)])
    result_table.add_row(["average gain percent", round(average_gain, 4)])
    result_table.add_row(["average loss percent", round(average_loss, 4)])
    result_table.add_row(["total days", trading_period])
    result_table.add_row(["1day average return percent", round(day_return, 2)])
    result_table.add_row(["probavility to win in 1 day", round(win_rate_perday, 2)])
    result_table.add_row(["1day average num trades", round(day_num_trade, 2)])
    result_table.add_row(["probavility to win in 1 month", round(win_rate_permonth, 2)])
    result_table.add_row(["1month average return percent", round(num_trades, 2)])
    result_table.add_row(["max draw down percent", round(drawdown, 2)])
    result_table.align["Field 1"] = "l"
    print(result_table)

    start = int(df.index[0].strftime("%Y%m%d%H"))
    end = int(df.index[-1].strftime("%Y%m%d%H"))

    result_dict = {
        "start": start,
        "end": end,
        "Total return percent": round(total_return, 2),
        "total days": trading_period,
        "win rate percent": win_rate,
        "lose rate percent": lose_rate,
        "1day average return percent": round(day_return, 2),
        "1day average num trades": round(day_num_trade, 2),
        "probavility to win in 1 day": round(win_rate_perday, 2),
        "1month average return ": round(month_return, 2),
        "probavility to win in 1 month": round(win_rate_permonth, 2),
        "1month average return percent": round(num_trades, 2),
        "Average Gain percent": round(average_gain, 4),
        "Average Loss percent": round(average_loss, 4),
        "Max Return percent": round(max_return, 2),
        "Max Loss percent": round(max_loss, 2),
        "max draw down percent": round(drawdown, 2),
    }

    each_month_return = PrettyTable()
    each_month_return.title = "each month return"
    each_month_return.field_names = ["month", "return"]
    for i, v in month_returns.iteritems():
        each_month_return.add_row([i.strftime("%Y/%m"), v])
    print(each_month_return)

    if downstream is not None:
        return_histgram(day_returns, downstream, "1day")
        return_histgram(month_returns, downstream, "1month")

    return result_dict
