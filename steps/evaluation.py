import logging
import pandas as pd
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin

from typing_extensions import Annotated
from typing import Tuple
import mlflow
from zenml.client import Client
from zenml import step

experiment_tracker = Client().active_stack.experiment_tracker
# # Get the client
# client = Client()

# # Check if there is an active stack
# if client.active_stack is None:
#     print("There is no active stack.")
# else:
#     print(f"The active stack is: {client.active_stack.name}")

#     # Check if the active stack has an experiment tracker
#     if client.active_stack.experiment_tracker is None:
#         print("The active stack does not have an experiment tracker.")
#     else:
#         print(
#             f"The experiment tracker is: {client.active_stack.experiment_tracker.name}")
#         experiment_tracker = client.active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> Tuple[
                       Annotated[float, 'r2_score'],
                       Annotated[float, 'rmse']]:

    try:
        prediction = model.predict(X_test)

        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric('mse', mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric('r2', r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric('rmse', rmse)

        return r2, rmse

    except Exception as e:
        logging.error(f'Error during model evaluation: {e}')
        raise e
