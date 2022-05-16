from logging import Logger
from typing import List, MutableSequence, Optional
from omegaconf import DictConfig


# Get meta info from mlflow experiment name that given
# at the 'mlflow run . --experiment_name {?} ...'
class BaseRunner:
    def __init__(
        self,
        cfg: DictConfig,
        exchange_name: Optional[str],
        trading_types: Optional[MutableSequence], 
        pair_names: Optional[MutableSequence],
        time_bars: Optional[MutableSequence],
        execution_types: Optional[MutableSequence],
        logger: Logger,
    ) -> None:
        self.cfg = cfg

        if not isinstance(exchange_name, str) and exchange_name is not None:
            raise ValueError("Invalid flag for exchange_names. Use '++mlflow.data_loader.exchange_name=`str`'")
        self.exchange_name = exchange_name

        if not isinstance(trading_types, MutableSequence) and trading_types is not None:
            raise ValueError("Invalid flag for trading_types. Use '++mlflow.data_loader.trading_types=`List`'")
        self.trading_types = trading_types

        if not isinstance(pair_names, MutableSequence) and pair_names is not None:
            raise ValueError("Invalid flag for pair_names. Use '++mlflow.data_loader.pair_names=`List`'")
        self.pair_names = pair_names

        if not isinstance(time_bars, MutableSequence) and time_bars is not None:
            raise ValueError("Invalid flag for time_bars. Use '++mlflow.data_loader.time_bars=`List`'")
        self.time_bars = time_bars

        if not isinstance(execution_types, MutableSequence) and execution_types is not None:
            raise ValueError("Invalid flag for time_bars. Use '++mlflow.data_loader.time_bars=`List`'")
        self.execution_types = execution_types 

        self.cfg_validator = HydraCfgValidator(self.cfg)

        if exchange_name is None and trading_types is None and pair_names is None:
            logger.warning("Load all exchanges, trading_types and pair_names because exchange_name, trading_types and pair_names are None.")
            logger.warning(
                "Please be sure that you use correct flags. '++mlflow.data_loader.exchange_name=`str`',"
                " '++mlflow.data_loader.trading_types=`List`', '++mlflow.data_loader.pair_names=`List`'"
                " '++mlflow.data_loader.time_bars=`List`'"
            )

        self.logger = logger


class HydraCfgValidator:
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        if not isinstance(cfg, DictConfig):
            raise ValueError("Invalid type for cfg (HydraCfgValidator)")
        self.cfg = cfg

    # Methods for hydra.conf.exchange
    def exchange_names(self) -> List:
        return list(self.cfg["exchange"].keys())

    def trading_types(self, exchange_name: str) -> List:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange name. Use a exchange name of {self.exchange_names()}")

        return list(self.cfg["exchange"][exchange_name]["trading_type"].keys())

    def pair_names(self, exchange_name: str, trading_type: str) -> List:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange name. Use one of {self.exchange_names()}")

        if not self.is_trading_type_valid(exchange_name, trading_type):
            raise ValueError(f"Invalid trading_type for {exchange_name}. Use one of {self.trading_types(exchange_name)}")

        return list(self.cfg["exchange"][exchange_name]["trading_type"][trading_type]["pair_name"].values())

    def time_bars(self) -> List:
        return list(self.cfg["mlflow"]["time_bar"].values())

    def execution_types(self) -> List:
        return list(self.cfg["mlflow"]["execution_type"].values())

    def fee_percent(self, exchange_name: str, trading_type: str, execution_type: str) -> float:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange name. Should be in {self.exchange_names()}")

        if not self.is_trading_type_valid(exchange_name, trading_type):
            raise ValueError(f"Invalid trading type for {exchange_name}. Should be in {self.trading_types(exchange_name)}")

        if not self.is_execution_type_valid(execution_type):
            raise ValueError(f"Invalid trading type for {execution_type}. Should be in {self.execution_types()}")

        _fee_percent = self.cfg["exchange"][exchange_name]["trading_type"][trading_type]["fee_percent"][execution_type]
        _fee_percent = 0.0 if _fee_percent is None else _fee_percent
        return _fee_percent

    def is_exchange_name_valid(self, exchange_name: str) -> bool:
        if not isinstance(exchange_name, str):
            raise ValueError("exchange_name should be string type for `is_exchange_names_valid`.")

        return exchange_name in self.exchange_names()

    def is_exchange_names_valid(self, exchange_names: MutableSequence) -> bool:
        if not isinstance(exchange_names, MutableSequence):
            raise ValueError("exchange_names should be list for `is_exchange_names_valid` method.")

        is_valid: bool = True
        for ex_name in exchange_names:
            is_valid = is_valid and self.is_exchange_name_valid(ex_name)

        return is_valid

    def is_trading_type_valid(self, exchange_name: str, trading_type: str) -> bool:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange_name: {exchange_name}. Should be in {self.exchange_names()}")

        if not isinstance(trading_type, str):
            raise ValueError("trading_type should be string type for `is_trading_types_valid`.")

        return trading_type in self.trading_types(exchange_name)

    def is_trading_types_valid(self, exchange_name: str, trading_types: MutableSequence) -> bool:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange_name: {exchange_name}. Should be in {self.exchange_names()}")

        if not isinstance(trading_types, MutableSequence):
            raise ValueError("trading_types should be List type for `is_trading_types_valid`")

        is_valid: bool = True
        for tr_type in trading_types:
            is_valid = is_valid and self.is_trading_type_valid(exchange_name, tr_type)

        return is_valid

    def is_pair_name_valid(self, exchange_name: str, trading_type: str, pair_name: str) -> bool:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange_name: {exchange_name}. Should be in {self.exchange_names()}")

        if not self.is_trading_type_valid(exchange_name, trading_type):
            raise ValueError(f"Invalid trading_type: {trading_type}. Should be in {self.trading_types()}")

        if not isinstance(pair_name, str):
            raise ValueError("You have to set pair_name: str to use `is_pair_names_valid`")

        return pair_name in self.pair_names(exchange_name, trading_type)

    def is_pair_names_valid(self, exchange_name: str, trading_type: str, pair_names: MutableSequence) -> bool:
        if not self.is_exchange_name_valid(exchange_name):
            raise ValueError(f"Invalid exchange_name: {exchange_name}. Should be in {self.exchange_names()}")

        if not self.is_trading_type_valid(exchange_name, trading_type):
            raise ValueError(f"Invalid trading_type: {trading_type}. Should be in {self.trading_types(exchange_name)}")

        if not isinstance(pair_names, MutableSequence):
            raise ValueError("pair_names should be List type for `is_pair_names_valid`")

        is_valid: bool = True
        for pair_name in pair_names:
            is_valid = is_valid and self.is_pair_name_valid(exchange_name, trading_type, pair_name)

        return is_valid

    def is_time_bar_valid(self, time_bar: str):
        if not isinstance(time_bar, str):
            raise ValueError("time_bar should be string for `is_time_bar_valid`")

        return time_bar in self.time_bars()

    def is_time_bars_valid(self, time_bars: MutableSequence):
        if not isinstance(time_bars, MutableSequence):
            raise ValueError("time_bars should be List or MutableSequence for `is_time_bars_valid`")

        is_valid: bool = True
        for time_bar in time_bars:
            is_valid = is_valid and self.is_time_bar_valid(time_bar)

        return is_valid

    def is_execution_type_valid(self, execution_type: str):
        if not isinstance(execution_type, str):
            raise ValueError("execution_type should be string for `is_execution_type_valid`")

        return execution_type in self.execution_types()

    def is_execution_types_valid(self, execution_types: MutableSequence):
        if not isinstance(execution_types, MutableSequence):
            raise ValueError("execution_types should be List or MutableSequence for `is_time_bars_valid`")

        is_valid: bool = True
        for execution_type in execution_types:
            is_valid = is_valid and self.is_execution_type_valid(execution_type)

        return is_valid

    # Methods for hydra.conf.train
    def train_model_names(self) -> List:
        return list(self.cfg["train"].keys())

    def train_types(self, train_model_name: str) -> List:
        if not isinstance(train_model_name, str):
            raise ValueError("train_model_name should be string for `train_types`")

        _train_types: List = list(self.cfg["train"][train_model_name].keys())

        # Check spell check of yaml file of hydra.
        is_valid: bool = True
        for _train_type in _train_types:
            is_valid = is_valid and (_train_type in ["classification", "regression"])
        if not is_valid:
            raise ValueError(f"Invalid train_type: {_train_types}. Should be in ['classification', 'regression']")

        return _train_types

    def train_parameter_options(self, train_model_name: str, train_type: str) -> List:
        if not self.is_train_model_name_valid(train_model_name):
            raise ValueError(f"Invalid train model name: {train_model_name}. Should be in {self.train_model_names()}")

        if not self.is_train_type_valid(train_model_name, train_type):
            raise ValueError(f"Invalid train type: {train_type}. Should be in {self.train_types(train_model_name)}")

        return list(self.cfg["train"][train_model_name][train_type].keys())

    def train_parameters(self, train_model_name: str, train_type: str, train_parameter_option: str) -> DictConfig:
        if not self.is_train_model_name_valid(train_model_name):
            raise ValueError(f"Invalid train model name: {train_model_name}. Should be in {self.train_model_names()}")

        if not self.is_train_type_valid(train_model_name, train_type):
            raise ValueError(f"Invalid train type: {train_type}. Should be in {self.train_types(train_model_name)}")

        if not self.is_train_parameter_option_valid(train_model_name, train_type, train_parameter_option):
            raise ValueError(
                f"Invalid train_parameter_options: {train_parameter_option} for {train_model_name} - {train_type}."
                f" Should be in {self.train_parameter_options(train_model_name, train_type)}"
            )

        return self.cfg["train"][train_model_name][train_type][train_parameter_option]

    def is_train_model_name_valid(self, train_model_name: str) -> bool:
        return train_model_name in self.train_model_names()

    def is_train_model_names_valid(self, train_model_names: MutableSequence) -> bool:
        if not isinstance(train_model_names, MutableSequence):
            raise ValueError("model_names should be MutableSequence for `is_train_model_names_valid`")

        is_valid: bool = True
        for train_model_name in train_model_names:
            is_valid = is_valid and self.is_train_model_name_valid(train_model_name)
        return is_valid

    def is_train_type_valid(self, train_model_name: str, train_type: str) -> bool:
        if not self.is_train_model_name_valid(train_model_name):
            raise ValueError(f"Invalid train model name: {train_model_name}. Should be in {self.train_model_names()}")

        return train_type in self.train_types(train_model_name)

    def is_train_types_valid(self, train_model_name: str, train_types: MutableSequence) -> bool:
        if not isinstance(train_types, MutableSequence):
            raise ValueError("train_types should be MutableSequence for `is_train_types_valid`.")

        is_valid: bool = True
        for train_type in train_types:
            is_valid = is_valid and self.is_train_type_valid(train_model_name, train_type)
        return is_valid

    def is_train_parameter_option_valid(self, train_model_name: str, train_type: str, train_parameter_option: str):
        if not self.is_train_model_name_valid(train_model_name):
            raise ValueError(f"Invalid train model name: {train_model_name}. Should be in {self.train_model_names()}")

        if not self.is_train_type_valid(train_model_name, train_type):
            raise ValueError(f"Invalid train type: {train_type}. Should be in {self.train_types(train_model_name)}")

        return train_parameter_option in self.train_parameter_options(train_model_name, train_type)

    def is_train_parameter_options_valid(self, train_model_name: str, train_type: str, train_parameter_options: MutableSequence):
        if not isinstance(train_parameter_options, MutableSequence):
            raise ValueError("train_parameters options shoud be MutableSequence for `is_train_parameters_options_valid`.")

        is_valid: bool = True
        for train_parameter_option in train_parameter_options:
            is_valid = is_valid and self.is_train_parameter_option_valid(train_model_name, train_type, train_parameter_option)
        return is_valid

    # Methods for hydra.conf.mlflow
    def mlflow_parameters(self) -> DictConfig:
        return self.cfg["mlflow"]["parameters"]
