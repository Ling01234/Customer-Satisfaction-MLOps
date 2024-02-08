import logging
from sklearn.linear_model import LinearRegression
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass


class LinearRegressionModel(Model):

    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)

            reg.fit(X_train, y_train)
            logging.info('Model Training Completed')
            return reg
        except Exception as e:
            logging.error(f'Error during model training: {e}')
            raise e
