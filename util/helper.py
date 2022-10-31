import random
import math
from datetime import datetime

def get_random_number(length):
    if length > 15:
        return -1
    return random.randint(math.pow(10, length), math.pow(10, length + 1) - 1)

def create_file_name(file, length):
    return file + datetime.utcnow().strftime('%Y_%m_%dT%H_%M_%SZ_') + str(get_random_number(length))

def create_GCS_path(bucket, folder, filename):
    return bucket + folder + '/' + datetime.utcnow().strftime('%Y/%m/%d/') + filename
