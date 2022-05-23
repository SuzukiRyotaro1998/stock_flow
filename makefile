

# ====================
# ML FLOW
# ====================
ml_base_train: ml_base/main.py
	poetry run python ml_base/main.py \
		++mlflow.experiment_target.train_model_name=lightgbm \
		'++mlflow.experiment_target.train_types=[regression]' \
        '++mlflow.experiment_target.train_parameter_options=[ryotaro_numerai]'
# '++mlflow.experiment_target.exchange_name=bybit' \
# '++mlflow.experiment_target.trading_types=[usdt_perpetual]' \
# '++mlflow.experiment_target.pair_names=[BTCUSDT]' \
# '++mlflow.experiment_target.time_bars=[15T]' \
.PHONY: ml_base_ui
ml_base_ui:
	poetry run mlflow ui --backend-store-uri=./ml_base/mlruns


# ====================
# RULE FLOW
# ====================
rule_base_train: rule_base/main.py
	poetry run python rule_base/main.py \
		++mlflow.experiment_target.exchange_name=gmo \
		'++mlflow.experiment_target.trading_types=[margin]' \
		'++mlflow.experiment_target.pair_names=[BTC_JPY]' \
		'++mlflow.experiment_target.time_bars=[1T]' 
# '++mlflow.experiment_target.execution_type=[STOP]' 
.PHONY: rule_base_ui
rule_base_ui:
	poetry run mlflow ui --backend-store-uri=./rule_base/mlruns


# ====================
# information coefficient
# ====================
ic: information_coefficient/main.py
	python information_coefficient/main.py \
		+mlflow.experiment_target.exchange_name=bybit \
		'++mlflow.experiment_target.trading_types=[usdt_perpetual]' \
		'++mlflow.experiment_target.time_bars=[15T]' 
# '++mlflow.experiment_target.pair_names=[BTC_JPY]' \
# '++mlflow.experiment_target.execution_type=[STOP]' 



# ====================
# make load clean test
# ====================
.PHONY: load_clean_test
load_clean_test:
	make load_data && make clean_data && make test_data

# ====================
# Sub command
# ====================
DATA_PROCESS_EXCHANGE_NAME='++mlflow.data_process.exchange_name=gmo'
DATA_PROCESS_TRADING_TYPES='++mlflow.data_process.trading_types=[margin]'
DATA_PROCESS_PAIR_NAMES='++mlflow.data_process.pair_names=[BTC_JPY]'
DATA_PROCESS_TIME_BARS='++mlflow.data_process.time_bars=[D]'

load_data: data_loader/main.py
	poetry run python data_loader/main.py \
		$(DATA_PROCESS_EXCHANGE_NAME) $(DATA_PROCESS_TRADING_TYPES) $(DATA_PROCESS_PAIR_NAMES) $(DATA_PROCESS_TIME_BARS)

clean_data: data_cleaning/main.py
	poetry run python data_cleaning/main.py \
		$(DATA_PROCESS_EXCHANGE_NAME) $(DATA_PROCESS_TRADING_TYPES) $(DATA_PROCESS_PAIR_NAMES) $(DATA_PROCESS_TIME_BARS)

test_data: data_testing/main.py
	poetry run python data_testing/main.py \
		$(DATA_PROCESS_EXCHANGE_NAME) $(DATA_PROCESS_TRADING_TYPES) $(DATA_PROCESS_PAIR_NAMES) $(DATA_PROCESS_TIME_BARS)


.PHONY: test
test:
	poetry run python -m unittest