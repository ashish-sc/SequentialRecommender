import json
import os
from datetime import datetime, timedelta
from functools import partial
from typing import Dict, List

from airflow import DAG
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator

from config.config import configuration
from driver.kubeDriver import *
from operators.gcp_container_operator_xcom_fix import GKEPodOperator
from util.ads_ds.slackAlerts import task_fail_slack_alert
from config.ads_ds.cold_start.config import configuration as cs_configuration
from config.ads_ds.cold_start import constants as cs_constants


def get_sql_query(sql_file_path, patterns=None):
    fsql = open(sql_file_path, "r")
    sql = fsql.read()
    fsql.close()
    sql = replace_patterns(sql, patterns)
    return sql


def replace_patterns(raw_str: str, patterns: dict = None):
    if patterns != None: 
        for k, v in patterns.items():
            raw_str = raw_str.replace(k, v)
    return raw_str


def get_dataset_config_value(dataset_name: str, keys: List):
    dataset_config = cs_configuration[dataset_name]
    if len(keys) > 0:
        for key in keys:
            dataset_config = dataset_config[key]
    return dataset_config


def prepare_training_tasks(dag: DAG,
                           dataset_name: str,
                           step_name: str,
                           bq_table: str,
                           query_path: str = None,
                           query_replace_pattern: dict = None,
                           dst_gcs_path: str = None,
                           query_operator_write_disposition: str = "WRITE_TRUNCATE",
                           query_operator_create_disposition: str = "CREATE_IF_NEEDED",
                           bq_export_format: str = "PARQUET"):

    tasks = []
    if query_path:
        bq_task = BigQueryOperator(
            task_id=dataset_name + '_' + step_name + '_bq',
            use_legacy_sql=False,
            write_disposition=query_operator_write_disposition,
            allow_large_results=True,
            priority='BATCH',
            bigquery_conn_id="moj-prod", #configuration['connections']['project'],
            sql=get_sql_query(query_path,
                              query_replace_pattern),
            create_disposition=query_operator_create_disposition,
            destination_dataset_table=bq_table,
            retries=1,
            dag=dag)
        tasks.append(bq_task)

    if dst_gcs_path:
        bq_to_gcs = BigQueryToCloudStorageOperator(
            task_id=dataset_name + '_' + step_name + '_bq_to_gcs',
            export_format=bq_export_format,
            source_project_dataset_table=bq_table,
            destination_cloud_storage_uris=[
                dst_gcs_path + f'.{bq_export_format.lower()}'],
            bigquery_conn_id="moj-prod", #configuration['connections']['project'],
            print_header=True,
            dag=dag
        )
        tasks.append(bq_to_gcs)
    
    return tasks


def get_gke_pod_operator(dag,
                         variant_name,
                         model_script_args,
                         docker_img_path):
    tolerations = [{
        'key': "nvidia.com/gpu",
        'operator': 'Equal',
        'value': 'present',
        'effect': 'NoSchedule'
    }, {
        'key': "content-affinity",
        'operator': 'Equal',
        'value': 'true',
        'effect': 'NoSchedule'
    }]

    n1_highmen_32_gpu_node_selectors = {
        'cloud.google.com/gke-nodepool': 'content-affinity-v2'
    }
    pod_resources_60G_32C = {'request_memory': "32Gi", 'request_cpu': '8000m',
                             'limit_memory': "32Gi", 'limit_cpu': '8000m', 'limit_gpu': 1}

    gke_pod_operator = GKEPodOperator(
        task_id=variant_name + "_scratch",
        cmds=['bash'],
        arguments=model_script_args,
        dag=dag,
        project_id=configuration["gcp_projects"]["production"],
        location="us-central1-a",
        cluster_name='data-science-gpu-production',
        name=variant_name,
        namespace="default",
        image=docker_img_path,
        image_pull_policy="Always",
        resources=pod_resources_60G_32C,
        volumes=[docker_volume_tag_clusters_update,
                 secret_volume_tag_clusters_update],
        volume_mounts=[docker_volume_mount_tag_clusters_update,
                       secret_volume_mount_tag_clusters_update],
        env_vars={'ACTIVE_ENV': os.environ['ACTIVE_ENV'],
                  'GOOGLE_APPLICATION_CREDENTIALS': '/root/.gcp/tag-cluster-service-sa.json'},
        labels={
            'pod-label': variant_name + "_scratch",
        },
        startup_timeout_seconds=60*30,
        execution_timeout=timedelta(hours=48),
        is_delete_operator_pod=True,
        get_logs=True,
        tolerations=tolerations,
        node_selectors=n1_highmen_32_gpu_node_selectors)
    return gke_pod_operator


def get_dummy_operator(dag: DAG, elem: str):
    Dummy = DummyOperator(
        task_id=elem,
        dag=dag,
    )
    return Dummy


def _start(**context):
    execution_date = context['next_execution_date'].strftime(
        '%Y_%m_%d__%H_%M_%S')
    return execution_date

def hour_setter(**context):
    hour_delta = '2'
    execution_hour = context['next_execution_date'].strftime('%H')
    if execution_hour == '23':
        hour_delta = '6'
    return hour_delta