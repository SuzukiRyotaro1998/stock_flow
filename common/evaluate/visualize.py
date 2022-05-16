import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


def return_histgram(list, downstream_directory, time_span):
    plt.hist(list, bins=100)
    plt.xlabel("return [%]")
    plt.ylabel("Frequency")
    plt.title(f"{time_span} return histgram ")

    plt.savefig(os.path.join(downstream_directory, f"{time_span}_return_histgram.png"))
    plt.close()


def prediction_histgram(downstream_directory, df):
    df["buy_predict"].hist(bins=100)
    plt.savefig(os.path.join(downstream_directory, "prediction_y_buyt.png"))
    plt.close()

    df["sell_predict"].hist(bins=100)
    plt.savefig(os.path.join(downstream_directory, "prediction_y_sell.png"))
    plt.close()


def cumulative_return_plot(downstream_directory, df):
    plot_df = df.resample("1D").apply(lambda x: x.iloc[-1] if len(x) > 0 else None)
    plt.figure(figsize=(15, 10))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.2, top=0.95)
    sns.lineplot(data=plot_df, y="cumulative_return", x=plot_df.index.strftime("%Y/%m/%d"))
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(downstream_directory, "backtest_result.png"))
    plt.close()


def position_change_plot(downstream_directory, df):
    ax = sns.lineplot(data=df, x=df.index, y=df["position"])
    ax.set_title("Position Change")
    plt.savefig(os.path.join(downstream_directory, "position_change.png"))
    plt.close()


def position_change_average_plot(downstream_directory, df):
    ax = sns.lineplot(data=df, x=df.index, y=df["position"].rolling(1000).mean())
    ax.set_title("Position Change Average")
    plt.savefig(os.path.join(downstream_directory, "position_change_average.png"))
    plt.close()


def predict_bar_plot(downstream_directory, df):
    buy_positive_count = (df["buy_predict"] > 0).sum()
    buy_negative_count = (df["buy_predict"] < 0).sum()
    sell_positive_count = (df["sell_predict"] > 0).sum()
    sell_negative_count = (df["sell_predict"] < 0).sum()
    data = {
        "y": [buy_positive_count, buy_negative_count, sell_positive_count, sell_negative_count],
        "x": ["buy_predict > 0", "buy_predict < 0", "sell_predict > 0", "sell_predict < 0"],
        "hue": ["buy", "buy", "sell", "sell"],
    }
    bar_data_df = pd.DataFrame(data)
    ax = sns.barplot(data=bar_data_df, x="x", y="y", hue="hue")
    ax.set_title("Buy-Sell prediction balance.")
    plt.savefig(os.path.join(downstream_directory, "predict_balance.png"))
    plt.close()
