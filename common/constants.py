import os
from pathlib import Path


class DATAFOLDER:
    common_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = Path(common_dir).parent.absolute()
    raw_data_root_path = os.path.join(root_dir, "raw_data/")
    cleaned_data_root_path = os.path.join(root_dir, "data/")

    ohlc_data_folder = os.path.join(cleaned_data_root_path, "ohlc")


class ASSET_INFO:
    asset_info = {
        "BTCUSDT": {},
    }


class COLUMNS:
    raw_data_columns = [
        "OpenTime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "CloseTime",
        "QuoteAssetVolume",
        "NumberOfTrades",
        "TakerBuyBaseAssetVolume",
        "TakerBuyQuoteAssetVolume",
        "Ignore",
    ]

    target_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
