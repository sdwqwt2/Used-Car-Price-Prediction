"""
Airflow DAG: used_car_price_pipeline
=====================================
Runs the full MLOps pipeline end-to-end every 5 minutes:
  1. Data engineering  (code/datasets/data_pipeline.py)
  2. Model engineering (code/models/train.py)
  3. Deployment        (docker compose build + up for API + app)

PROJECT_ROOT must point at the repository root so the pipeline
modules and docker-compose file can be found. Set it via the
PROJECT_ROOT env var (defaults to a path relative to this file
assuming the standard repo layout).
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def run_data_engineering():
    os.chdir(PROJECT_ROOT)
    from code.datasets.data_pipeline import main as data_main
    data_main()


def run_model_engineering():
    os.chdir(PROJECT_ROOT)
    from code.models.train import main as train_main
    train_main()


with DAG(
    dag_id="used_car_price_pipeline",
    description="Data engineering -> model engineering -> deployment for used car price prediction",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "used-car-price"],
) as dag:

    data_engineering = PythonOperator(
        task_id="data_engineering",
        python_callable=run_data_engineering,
    )

    model_engineering = PythonOperator(
        task_id="model_engineering",
        python_callable=run_model_engineering,
    )

    deploy = BashOperator(
        task_id="deployment",
        bash_command=(
            f"cd {PROJECT_ROOT}/code/deployment && "
            "docker compose build && docker compose up -d"
        ),
    )

    data_engineering >> model_engineering >> deploy
