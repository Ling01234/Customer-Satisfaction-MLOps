import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataDivideStrategy, DataPreprocessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test'],
]:
    try:
        process_strategy = DataPreprocessStrategy()
        data_clean = DataCleaning(df, process_strategy)
        processed_data = data_clean.handle_data()

        divide_strategy = DataDivideStrategy()
        data_clean = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_clean.handle_data()

        logging.info('Data cleaning completed')

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f'Error in data cleaning: {e}')
        raise e
