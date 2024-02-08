import logging
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from abc import ABC, abstractmethod


class Evaluation(ABC):
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class MSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating MSE:')
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f'MSE calculation done')
            return mse
        except Exception as e:
            logging.error(f'Error during MSE calculation: {e}')
            raise e


class R2(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating R2 score')
            r2 = r2_score(y_true, y_pred)
            logging.info('R2 score calculation done')
            return r2
        except Exception as e:
            logging.error(f'Error during R2 calculation: {e}')
            raise e


class RMSE(Evaluation):
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info('Calculating RMSE:')
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f'RMSE calculation done')
            return rmse
        except Exception as e:
            logging.error(f'Error during RMSE calculation: {e}')
            raise e
