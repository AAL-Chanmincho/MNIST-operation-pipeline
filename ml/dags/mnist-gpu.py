from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# DAG의 기본 인자 설정
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['your_email@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'start_date': datetime(2023, 11, 30),
    'retry_delay': timedelta(minutes=5),
}

with DAG(
        'mnist_pytorch_training',
        default_args=default_args,
        description='MNIST training with PyTorch and MLflow',
        schedule_interval=timedelta(days=1),
) as dag:
    train_and_save_model_job = BashOperator(
        task_id='train_save_model',
        depends_on_past=False,
        bash_command='python3 /opt/airflow/dags/scripts/training-and-save.py ',
        retries=1,
        dag=dag,
    )

