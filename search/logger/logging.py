
import logging
import os
import time
from datetime import datetime

# 初始化保存文件路径的函数,可以输入想要保存的文件名，否则创建默认文件名
def make_path(fn=''):
    dt = datetime.now()
    if fn == '':
        dt = str(dt).split('.')[0][2:]
        filename = 'job_' + dt
    else:
        dt = str(dt).split('.')[0][2:-9]
        filename = fn + '_' + dt

    path = './Job/' + filename
    os.makedirs(path, exist_ok=True)
    return path + '/'

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger