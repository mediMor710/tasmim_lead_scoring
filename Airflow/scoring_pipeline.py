from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline_tasks import(
    task_extract_leads,
    task_build_features,
    task_score_and_save
)

from data.sync_from_sheets import sync_new_leads

##### PART A: Assigning arguments #####
default_args = {
    'owner': 'tasmim_web',
    'retries': 1,   # Try one more time after a failure
    'retry_delay': timedelta(minutes=5),    # Waiting 5 minutes before retrying
    'email_on_failure': False,
}


##### PART B: DAG Definition #####
with DAG(
    dag_id = 'tasmim_lead_scoring_pipeline',
    description = 'Daily pipeline: extract unscored leads -> build features -> score and save',
    default_args = default_args,
    start_date = datetime(2026, 1, 1),
    schedule = '@daily',
    catchup = False,    # Forget about the old leads and start from the new ones
    tags = ['tasmim', 'lead_scoring', 'ML', 'Intership Project']
) as dag:
    
    ##### PART C: Task definitions #####
    sync = PythonOperator(
        task_id = 'sync_from_sheets',
        python_callable = sync_new_leads
    )
    extract = PythonOperator(
        task_id = 'extract_leads',
        python_callable = task_extract_leads,
    )

    build = PythonOperator(
        task_id = 'build_features',
        python_callable = task_build_features,
    )

    score = PythonOperator(
        task_id = 'score_and_save',
        python_callable = task_score_and_save,
    )

    ##### PART D: Defining Task order #####
    sync >> extract >> build >> score