import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series,
                config: ModelNameConfig) -> RegressorMixin:

    model = None

    try:
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train(X_train=X_train, y_train=y_train)
            logging.info('Completed Model training')
            return trained_model
        else:
            raise ValueError(f'Model {config.model_name} not supported.')
    except Exception as e:
        logging.error(f'Error in training model: {e}')
        raise e
