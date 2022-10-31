from airflow.hooks.base_hook import BaseHook
import requests
from config.config import configuration
import os

def task_fail_slack_alert(context, mentions = []):
    if (os.environ['ACTIVE_ENV'] == 'PRODUCTION'):
        slack_webhook_token = BaseHook.get_connection(configuration['connections']['slack']).password
        slack_webhook_base_url = BaseHook.get_connection(configuration['connections']['slack']).host
        slack_msg = """
                :red_circle: Task Failed.
                *Task*: {task}
                *Dag*: {dag}
                *Execution Time*: {exec_date}
                *Log Url*: {log_url}
                """.format(
                task=context.get('task_instance').task_id,
                dag=context.get('task_instance').dag_id,
                ti=context.get('task_instance'),
                exec_date=context.get('execution_date'),
                log_url=context.get('task_instance').log_url)

        curl_msg = '{{\"text\":\"{message}\"}}'.format(message=" ".join(mentions) + slack_msg)

        url = slack_webhook_base_url + slack_webhook_token
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=curl_msg, headers=headers)

def task_success_slack_alert(context, mentions = []):
    if (os.environ['ACTIVE_ENV'] == 'PRODUCTION'):
        slack_webhook_token = BaseHook.get_connection(configuration['connections']['slack']).password
        slack_webhook_base_url = BaseHook.get_connection(configuration['connections']['slack']).host
        slack_msg = """
                :success_kid: Task succeeded.
                *Task*: {task}
                *Dag*: {dag}
                *Execution Time*: {exec_date}
                *Log Url*: {log_url}
                """.format(
                task=context.get('task_instance').task_id,
                dag=context.get('task_instance').dag_id,
                ti=context.get('task_instance'),
                exec_date=context.get('execution_date'),
                log_url=context.get('task_instance').log_url)

        curl_msg = '{{\"text\":\"{message}\"}}'.format(message=" ".join(mentions) + slack_msg)

        url = slack_webhook_base_url + slack_webhook_token
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        r = requests.post(url, data=curl_msg, headers=headers)