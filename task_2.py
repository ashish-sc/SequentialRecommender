from datetime import datetime, timedelta
import os
import json

from airflow import DAG
from airflow.models import Variable

from airflow.utils.task_group import TaskGroup
from airflow.kubernetes.secret import Secret
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator, BigQueryDeleteTableOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from operators.gcs_operatorV2 import ContentToGoogleCloudStorageOperator
from airflow.providers.http.operators.http import SimpleHttpOperator

from util import slackUtil, helper
from config.config import configuration

from functools import partial
task_fail_slack_alert = partial(slackUtil.task_fail_slack_alert, mentions=['<@U01QG21KQNP>'])
task_success_slack_alert = partial(slackUtil.task_success_slack_alert, mentions=[])

# Check -  reflects on DAG

DAG_NAME = "sequential_isp_training"
TPU_PREFIX = "sequential_isp"
LARGE_LANGS = ["Hindi", "Tamil", "Telugu", "Marathi", "Kannada"]
LANGS = [
    "Hindi",
    "Tamil",
    "Telugu",
    "Kannada",
    "Punjabi",
    "Odia",
    "Bengali",
    "Marathi",
    "Malayalam",
    "Gujarati",
]
NUM_QUERIES = 5
TRAINING_DAYS = 8
REDIS_PUSH_IMAGE = "asia.gcr.io/sharechat-production/sequential_isp_training_push_to_redis:latest"


n1_16_pool_affinity = {
    'nodeAffinity': {
        'requiredDuringSchedulingIgnoredDuringExecution': {
            'nodeSelectorTerms': [{
                'matchExpressions': [{
                    'key': 'cloud.google.com/gke-nodepool',
                    'operator': 'In',
                    'values': [
                        'n1-standard-16'
                    ]
                }]
            }]
        }
    }
}

tolerations = [{
    'key': "ffm-train",
    'operator': 'Equal',
    'value': 'true',
    'effect': 'NoSchedule'
}]


def set_table_name_prefix(**kwargs):
    table_prefix = helper.create_file_name(
        configuration['big_query_tables']['sequential_isp_tpu_temp_results'], 5)
    kwargs['ti'].xcom_push(key='bigquery_table_prefix', value=table_prefix)


def set_gcs_file_name_prefix(**kwargs):
    final_path = helper.create_file_name(
        configuration['gcs_filenames']['sequential_isp_model_data'], 5)
    destination_storage_path = helper.create_GCS_path(
        configuration['gcs_buckets']['sequential_isp_tpu_data'], 'sequential_isp_model', final_path)
    bucket, file_path = destination_storage_path[len("gs://"):].split("/", 1)
    kwargs['ti'].xcom_push(key='gcs_path', value=destination_storage_path)
    kwargs['ti'].xcom_push(key='gcs_file_path', value=file_path)
    kwargs['ti'].xcom_push(key='gcs_bucket', value=bucket)


def readSqlFile(file_name, dtype, lang, table_prefix, start_time, end_time, q0table):
    with open (os.path.dirname(__file__) + "/queries" + "/" + file_name + ".sql", "r") as file:
        sql_command=file.read()
        sql_command = sql_command.format(
            start_time=start_time,
            end_time=end_time,
            language=lang,
            q0table=q0table,
            q1table_post_index=table_prefix+'post_index',
            q1table_ugc=table_prefix+"ugc",
            q1table_post=table_prefix+"post",
            q1table_user=table_prefix+"user",
            q1table_views=table_prefix+"views",
            q1table_sampling=table_prefix+"sample",
            q2table_popular_posts=table_prefix+"popular_posts",
            q2table_uhistory=table_prefix+"uhistory",
            q1table_user_index=table_prefix+"user_index",
            q2table=table_prefix+"2",
            q3table=table_prefix+"3",
            q4table=table_prefix+"4",
            q5table=table_prefix+"5",
            q6table=table_prefix+"6",
            dtype=dtype
        )
    return sql_command


def update_airflow_variable(**context):
    print(json.dumps(context['templates_dict']))
    Variable.set("sequential_isp_last_success_run",
                 # Need to copy to two_tower_candidate_generator_last_incremental_success_run
                 json.dumps(context['templates_dict']))


tpu_secret_volume = Secret(
    deploy_type='volume',
    deploy_target='/root/.gcp/',
    secret='ds-tpu-sa',
    key='ds-tpu-sa.json'
)
secret_volume = Secret(
    deploy_type='volume',
    deploy_target='/root/.gcp/',
    secret='feed-relevance-service-sa',
    key='feed-relevance-service-sa.json'
)

default_args = {
    'owner': 'Data Science Team',
    'depends_on_past': False,
    'catchup': False,
    'start_date': datetime(2022, 11, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'on_failure_callback': task_fail_slack_alert,
}

dag = DAG(
    DAG_NAME,
    default_args=default_args,
    description='The job for training Sequential ISP model',
    schedule_interval="@once",
    user_defined_macros={
        'project_id': configuration['gcp_projects']['production']
    },
    catchup=False,
    concurrency=100,
)

set_destination_table = PythonOperator(
    task_id='set_destination_tablename',
    provide_context=True,
    python_callable=set_table_name_prefix,
    dag=dag
)

create_gcs_file = PythonOperator(
    task_id='set_gcs_bucket_name',
    provide_context=True,
    python_callable=set_gcs_file_name_prefix,
    dag=dag
)

# write_ffm_version_gcs = ContentToGoogleCloudStorageOperator(
#     task_id='write_tpu_version_gcs',
#     content="{{ ti.xcom_pull(task_ids='set_gcs_bucket_name', key='gcs_path', dag_id=dag.dag_id) }}",
#     dst='sequential_isp_24hr_v1/version.txt',
#     bucket='sharechat-ml-models-prod-v2',
#     gcp_conn_id=configuration['connections']['project'],
#     dag=dag,
# )

done = DummyOperator(
    task_id="sequential_isp_finished",
    dag=dag,
    on_success_callback=task_success_slack_alert
)

start = DummyOperator(
    task_id="start-dag",
    dag=dag,
)

data_end_time = '2022-11-07T16:12:36'
end_time = datetime.strptime(data_end_time, "%Y-%m-%dT%H:%M:%S")
start_time = end_time - timedelta(days=TRAINING_DAYS)
end_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
start_time = start_time.strftime('%Y-%m-%d %H:%M:%S')

runtime = data_end_time

var_val = {"previous_run_end_time": end_time, "model_info": {}}

update_airflow_variable_success_run = PythonOperator(
    task_id='update_airflow_variable_success_run',
    provide_context=True,
    python_callable=update_airflow_variable,
    templates_dict=var_val,
    dag=dag,
)

metric_name = "ndcg"

for lang in LANGS:
    var_val["model_info"][lang] = {}

    if lang in LARGE_LANGS:
        batch_size = 1024
        tpu_type = "v2-32"
        tpu_priority_weight = 2
    else:
        batch_size = 512
        tpu_type = "v3-8"
        tpu_priority_weight = 1

    dtype = "video"
    with TaskGroup(group_id=f"data_preperation_{lang.lower()}", dag=dag) as lang_tg:
        var_val["model_info"][lang][dtype] = {}
        model_prefix = lang.lower() + '_' + dtype
        gcs_pref2 = "{{ ti.xcom_pull(task_ids='set_gcs_bucket_name', key='gcs_path') }}" + "_" + model_prefix

        q0table = "{{ ti.xcom_pull(task_ids='set_destination_tablename', key='bigquery_table_prefix') }}_" + model_prefix + "_0"

        base_query = BigQueryExecuteQueryOperator(
            task_id='bq_write_candidates_' + model_prefix + "_0",
            use_legacy_sql=False,
            write_disposition='WRITE_TRUNCATE',
            allow_large_results=True,
            priority='BATCH',
            pool='isp_train_base_bq',
            api_resource_configs={
                'query': {
                    'timeoutMs': 1000 * 60 * 80
                }
            },
            priority_weight=tpu_priority_weight,
            sql=readSqlFile('query0', dtype, lang, '', start_time, end_time, q0table),
            bigquery_conn_id='datascience-sc-production',
            create_disposition='CREATE_IF_NEEDED',
            dag=dag
        )
        # delete_base_table = BigQueryDeleteTableOperator(
        #     deletion_dataset_table=q0table,
        #     task_id='bq_delete_table_' + model_prefix + "_0",
        #     bigquery_conn_id=configuration['connections']['project'],
        #     dag=dag
        # )
        start >> [create_gcs_file, set_destination_table] >> base_query

        query_ops = []
        delete_ops = []
        gcs_ops = []

        model_name = lang.lower() + '_' + dtype
        table_pref = "{{ ti.xcom_pull(task_ids='set_destination_tablename', key='bigquery_table_prefix') }}" + \
            "_" + model_name + "_"
        gcs_pref = "{{ ti.xcom_pull(task_ids='set_gcs_bucket_name', key='gcs_path') }}" + \
            "_" + model_name + "_"

        var_val["model_info"][lang][dtype] = {
            "model_path": f"{gcs_pref}out/tfserve_two_tower",
            "model_weights": f"{gcs_pref}out/"
        }

        with TaskGroup(group_id=f"user_counters", dag=dag) as user_counters_tg:
            user_counter = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_user',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='ffm_feed_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_user', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # user_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'user',
            #     task_id='bq_delete_table_user',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            base_query >> user_counter

            user_index = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_user_index',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_user_index', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # user_count_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'user_count',
            #     task_id='bq_delete_table_user',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            base_query >> user_index

            user_tasks = [user_counter, user_index]

        with TaskGroup(group_id=f"post_counters", dag=dag) as post_counters_tg:
            post_index = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_post_index',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_post_index', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            bq_to_gcs_post_index = BigQueryToGCSOperator(
                task_id='bq_to_gcs_post_index',
                export_format='CSV',
                source_project_dataset_table=table_pref+"post_index",
                destination_cloud_storage_uris=[gcs_pref+'post_index/*.csv'],
                # bigquery_conn_id=configuration['connections']['project'],
                bigquery_conn_id="datascience-sc-production",
                print_header=False,
                dag=dag
            )
            # post_index_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'post_index',
            #     task_id='bq_delete_table_post_index',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            base_query >> post_index >> bq_to_gcs_post_index

            post_ugc = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_ugc',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_ugc', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # ugc_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'ugc',
            #     task_id='bq_delete_table_ugc',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )

            [create_gcs_file, set_destination_table] >> post_ugc >> post_index

            post_counter = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_post',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_post', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # post_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'post',
            #     task_id='bq_delete_table_post',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            base_query >> post_counter
            post_tasks = [post_ugc, post_counter]

        with TaskGroup(group_id=f"base_dataset", dag=dag) as base_dataset_tg:
            views_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_views',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_views', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # views_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'views',
            #     task_id='bq_delete_table_views',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )

            sampling_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_sampling',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query1_sampling', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # sampling_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'sampling',
            #     task_id='bq_delete_table_sampling',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )

            post_index >> views_table >> sampling_table
            user_index >> views_table
            
        with TaskGroup(group_id=f"user_history_aggregation", dag=dag) as user_history_aggregation_tg:
            uhistory_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_uhistory',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query2_uhistory', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # uhistory_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + 'uhistory',
            #     task_id='bq_delete_table_uhistory',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            views_table >> uhistory_table

            # user_history_tasks = [popular_table, uhistory_table]
            user_history_tasks = [uhistory_table]

        with TaskGroup(group_id=f"dataset_merge", dag=dag) as dataset_merge_tg:
            counters_merge_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_counters_merge',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query2', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # counters_merge_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + '2',
            #     task_id='bq_delete_table_counters_merge',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            [sampling_table] + post_tasks + user_tasks >> counters_merge_table

            uhistory_merge_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_uhistory_merge',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query3', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # uhistory_merge_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + '3',
            #     task_id='bq_delete_table_uhistory_merge',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            [counters_merge_table] + user_history_tasks >> uhistory_merge_table

        with TaskGroup(group_id=f"final_dataset", dag=dag) as final_dataset_tg:
            final_dataset_table = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_final_dataset',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query4_onetime', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            # final_dataset_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref + '4',
            #     task_id='bq_delete_table_final_table',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )

            uhistory_merge_table >> final_dataset_table

        with TaskGroup(group_id="data_split", dag=dag) as data_split_tg:
            train_split = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_train_split',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query5_train', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            bq_to_gcs_train = BigQueryToGCSOperator(
                task_id='bq_to_gcs_train_split',
                export_format='CSV',
                source_project_dataset_table=table_pref+"5_train",
                destination_cloud_storage_uris=[gcs_pref+'train/*.csv'],
                # bigquery_conn_id=configuration['connections']['project'],
                bigquery_conn_id="datascience-sc-production",
                print_header=False,
                dag=dag
            )
            # train_split_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref+'5_train',
            #     task_id='bq_delete_table_train_split',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            train_split >> bq_to_gcs_train

            valid_split = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_valid_split',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query5_valid', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            bq_to_gcs_valid = BigQueryToGCSOperator(
                task_id='bq_to_gcs_valid_split',
                export_format='CSV',
                source_project_dataset_table=table_pref+"5_valid",
                destination_cloud_storage_uris=[gcs_pref+'valid/*.csv'],
                # bigquery_conn_id=configuration['connections']['project'],
                bigquery_conn_id="datascience-sc-production",
                print_header=False,
                dag=dag
            )
            # valid_split_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref+'5_valid',
            #     task_id='bq_delete_table_valid_split',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            valid_split >> bq_to_gcs_valid

            data_meta = BigQueryExecuteQueryOperator(
                task_id='bq_write_candidates_data_meta',
                use_legacy_sql=False,
                write_disposition='WRITE_TRUNCATE',
                allow_large_results=True,
                priority='BATCH',
                pool='isp_train_base_bq',
                api_resource_configs={'query': {'timeoutMs': 1000 * 60 * 60}},
                sql=readSqlFile('query6', dtype, lang, table_pref, start_time, end_time, q0table),
                bigquery_conn_id='datascience-sc-production',
                create_disposition='CREATE_IF_NEEDED',
                dag=dag
            )
            
            bq_to_gcs_meta = BigQueryToGCSOperator(
                task_id='bq_to_gcs_data_meta',
                export_format='CSV',
                source_project_dataset_table=table_pref+"6",
                destination_cloud_storage_uris=[gcs_pref+'meta/*.csv'],
                # bigquery_conn_id=configuration['connections']['project'],
                bigquery_conn_id="datascience-sc-production",
                print_header=False,
                dag=dag
            )
            # data_meta_delete = BigQueryDeleteTableOperator(
            #     deletion_dataset_table=table_pref+'6',
            #     task_id='bq_delete_table_data_meta',
            #     bigquery_conn_id=configuration['connections']['project'],
            #     dag=dag
            # )
            [data_meta, bq_to_gcs_post_index] >> bq_to_gcs_meta

        final_dataset_table >> [train_split, valid_split] >> data_meta
        [bq_to_gcs_train, bq_to_gcs_valid] >> bq_to_gcs_meta
        
    with TaskGroup(group_id=f"training_{lang.lower()}", dag=dag) as lang_tg:
        train = GKEStartPodOperator(
            task_id='sequential-isp-training-pod-task-' + model_name,
            project_id=configuration["gcp_projects"]["production"],
            location="us-central1",
            cluster_name='ds-feed-tpu-sc-prod',
            name='sequential-isp-training-pod-' + model_name,
            namespace="default",
            cmds=['sh'],
            arguments=[
                "run_all.sh", runtime, gcs_pref+"post_index/", "NONE", "NONE", "NONE",
                gcs_pref+"train/", gcs_pref+"valid/", gcs_pref+"meta/", gcs_pref+"out/",
                metric_name, "false", lang, "us-central1-a", "false", data_end_time, dtype, str(batch_size),
                tpu_type
            ], # Integer parameters but need to provided as string to pod operator
            image=configuration['gcr_images']['two_tower_candidate_generator_training'],
            image_pull_policy="Always",
            is_delete_operator_pod=True,
            resources={'request_memory': "20Gi", 'request_cpu': 4},
            secrets=[tpu_secret_volume],
            env_vars={'ACTIVE_ENV': os.environ['ACTIVE_ENV'],
                        'GOOGLE_APPLICATION_CREDENTIALS': '/root/.gcp/ds-tpu-sa.json'},
            labels={
                'pod-label': 'ffm-tf',
            },
            startup_timeout_seconds=60 * 20,
            retries=3,
            retry_delay = timedelta(minutes=10),
            execution_timeout=timedelta(hours=72),
            get_logs=True,
            # tolerations=tolerations,
            node_selectors={'cloud.google.com/gke-nodepool': 'n1-highmem-32-tpu'},
            pool="tpu_ffm_train",
            priority_weight=tpu_priority_weight,
            dag=dag
        )

        bq_to_gcs_meta >> train >> update_airflow_variable_success_run

    with TaskGroup(group_id=f"push_embs_{lang.lower()}", dag=dag) as lang_tg:
        post_emb_to_redis = KubernetesPodOperator(
            name="post-emb-to-redis-" + model_name,
            task_id="sequential-isp-post-emb-to-redis-task-" + model_name,
            namespace="default",
            image="gcr.io/sharechat-production/redis-operators:latest",
            arguments=[gcs_pref + "out/post_embeddings_redis/0000.json", "embeddingsStore", "hmset", str(int(timedelta(days=90).total_seconds())), "0.005"],
            image_pull_policy="Always",
            resources={ 'request_memory': "2Gi", 'request_cpu': "800m", 'limit_memory': "4Gi", 'limit_cpu': "1000m" },
            is_delete_operator_pod=True,
            affinity=n1_16_pool_affinity,
            secrets=[secret_volume],
            env_vars={"ACTIVE_ENV": os.environ["ACTIVE_ENV"]},
            startup_timeout_seconds=600,
            execution_timeout=timedelta(minutes=90),
            get_logs=True,
            dag=dag,
            )
        
        train >> post_emb_to_redis >> update_airflow_variable_success_run
        # train >> update_airflow_variable_success_run

# update_airflow_variable_success_run >> write_ffm_version_gcs >> done
update_airflow_variable_success_run >> done
