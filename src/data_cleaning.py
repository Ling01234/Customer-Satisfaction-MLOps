import logging
from typing import Union
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data

    Args:
        ABC (_type_): _description_
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(
                data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(
                data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(
                data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(
                data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)

            return data
        except Exception as e:
            logging.error(f'Error in preprocessing data: {e}')
            raise e


class DataDivideStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame | pd.Series:
        try:
            X = data.drop('review_score', axis=1)
            y = data['review_score']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error in dividing data: {e}')
            raise e


class DataCleaning:
    """
    Process data and divides data
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> pd.DataFrame | pd.Series:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f'Error in data cleaning: {e}')
            raise e
