
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_DIR = "/opt/project"

with DAG(
    dag_id="recommendation_model_retraining",
    description="Automated Two-Tower recommendation model retraining",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["recommendation", "machine-learning", "retraining"],
) as dag:

    check_dataset = BashOperator(
        task_id="check_training_dataset",
        bash_command=(
            f"test -f {PROJECT_DIR}/data/processed/train.csv "
            "&& echo 'Training dataset found'"
        ),
    )

    retrain_two_tower = BashOperator(
        task_id="retrain_two_tower_model",
        bash_command=(
            f"python {PROJECT_DIR}/src/two_tower_model.py"
        ),
    )

    check_dataset >> retrain_two_tower
