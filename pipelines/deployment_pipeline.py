import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
# from materializer.customer_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float = 0


@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig):
    return accuracy >= config.min_accuracy


@pipeline(enable_cache=False, settings={'docker': docker_settings})
def continuous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rsme = evaluate_model(model, X_test, y_test)
    # print(f'R2 Score: {r2_score}')

    # criteria for deployment
    deployment_decision = deployment_trigger(
        r2_score)

    mlflow_model_deployer_step(model=model,
                               deploy_decision=deployment_decision,
                               workers=workers,
                               timeout=timeout)
