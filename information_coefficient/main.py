import os
from typing import MutableSequence, Optional
# import sys
# sys.path.append("../..")
from information_coefficient.experiment.ic_flow import ic_flow

def main():
    ic_flow()

    # train_flow(
    #     downstream=mlflow_parameters.evaluate_downstream,
    #     exchange_name=exchange_name,
    #     trading_type=trading_type,
    #     pair_name=pair_name,
    #     time_bar=time_bar,
    # )

if __name__ == "__main__":
    main()
