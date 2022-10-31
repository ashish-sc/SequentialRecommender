gpu_resources_150G = {
    'request_memory': '150Gi',
    'request_cpu': 30,
    'limit_memory': '180Gi',
    'limit_cpu': 32,
    'limit_gpu': 4,
}


n1_highmen_32_gpu_tolerations = [
    {
        'key': "nvidia.com/gpu",
        'operator': 'Equal',
        'value': 'present',
        'effect': 'NoSchedule'
    },
    {
        'key': "highmem-4x-t4",
        'operator': 'Equal',
        'value': 'true',
        'effect': 'NoSchedule'
    }
]

n1_highmen_32_gpu_node_selectors = {
    'cloud.google.com/gke-nodepool': 'n1-highmem-32-4x-t4-gpu'
}

n1_highmen_300g_gpu_node_selectors = {
            'cloud.google.com/gke-nodepool': 'cpu-32-300-gb-4t4'
}

n1_highmen_400g_gpu_node_selectors = {
    'cloud.google.com/gke-nodepool': 'cpu-32-400-gb-4t4'
}

gpu_resources_200G = {
    'request_memory': '200Gi',
    'request_cpu': 32,
    'limit_memory': '250Gi',
    'limit_cpu': 32,
    'limit_gpu': 4,
}


n1_highmen_64_test_gpu_tolerations = [
    {
        'key': "nvidia.com/gpu",
        'operator': 'Equal',
        'value': 'present',
        'effect': 'NoSchedule'
    },
    {
        'key': "test-highmem-4x-t4",
        'operator': 'Equal',
        'value': 'true',
        'effect': 'NoSchedule'
    },
    {
        'key': "test-nvidia.com/gpu",
        'operator': 'Equal',
        'value': 'present',
        'effect': 'NoSchedule'
    }
]

n1_highmen_64_test_gpu_node_selectors = {
    'cloud.google.com/gke-nodepool': 'n1-highmem-32-4x-t4-gpu-test'
}


def get_sql_query(sql_file_path, patterns):
    fsql = open(sql_file_path, "r")
    sql = fsql.read()
    fsql.close()
    for k, v in patterns.items():
        sql = sql.replace(k, v)
    return sql


# docker_image = "asia.gcr.io/sharechat-production/knowledge-graph-embedding:latest"

from airflow.contrib.kubernetes.volume import Volume
from airflow.contrib.kubernetes.volume_mount import VolumeMount

shm_volume_mount = VolumeMount('dshm',
                                  mount_path='/dev/shm',
                                  sub_path=None,
                                  read_only=False)

shm_volume_config = {
    'emptyDir': {
        'medium': 'Memory',
        'sizeLimit': "200Gi"
    }
}



shm_volume = Volume(
    name='dshm',
    configs=shm_volume_config)


def replace_hyphen(inp: str):
    return inp.replace("-", "_")


def replace_colon(inp: str):
    return inp.replace(":", "_")


bucket = "ds-compose-jobs"

project_path = "BGE/prod/"
stg_project_path = "BGE/test/"
export_path = "/export_to_BT"

mlt_file_name = ["video_post", "image_post"]

feed_docker_image = "gcr.io/sharechat-production/knowledge-graph-embedding-feed:latest"
predictionPostfix = "/prediction"
feed_file_name = ["link_engagement", "link_activity", "transformed_link_post_author"] # link_author
