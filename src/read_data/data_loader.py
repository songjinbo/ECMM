import os
import sys
import math
import datetime
from conf import params
import tensorflow.compat.v1 as tf
import numpy as np

class DataLoader(object):
    def __init__(self, file_dir, file_type, logger, begin_day=None, end_day=None):
        self.file_dir = file_dir
        self.file_type = file_type
        self.begin_day = begin_day
        self.end_day = end_day
        self.file_list = []
        self.logger = logger

    def _get_data_recursive_hdfs(self, hdfs_dir):
            p = os.popen("hadoop fs -ls %s | awk '{print $8}'"%(hdfs_dir))
            file_list = p.readlines()[:]
            for ff in file_list:
                ff = ff.strip().split('\n')[0]
                if ff == '':
                    continue
                ret = os.system('hdfs dfs -test -f %s'%(ff))
                if int(ret) == 0:
                    if "_SUCCESS" in ff:
                        continue
                    self.file_list.append(ff)
                else:
                    day=ff.strip().split('/')[-1]
                    if self.begin_day != None and self.end_day != None:
                        if int(day) < int(self.begin_day) or int(day) > int(self.end_day):
                            continue
                    self._get_data_recursive_hdfs(ff)

    def get_data_recursive_local(self, local_dir):
        file_list = os.popen("ls -l %s | awk '{print $9}'"%(local_dir)).readlines()[:]
        for ff in file_list:
            day = ff = ff.strip()
            if ff == '':
                continue
            if ff.startswith('.'):
                continue
            ff = local_dir + '/' + ff
            if os.path.isdir(ff):
                if self.begin_day != None and self.end_day != None:
                    if int(day) < int(self.begin_day) or int(day) > int(self.end_day):
                        continue
                self.get_data_recursive_local(ff)
            else:
                if "_SUCCESS" in ff:
                    continue
                self.file_list.append(ff)

    def _format_data_dir(self):
        if self.file_type == "hdfs":
            self._get_data_recursive_hdfs(self.file_dir)
        elif self.file_type == "local":
            self.get_data_recursive_local(self.file_dir)
        else:
            self.logger.info("wrong file_type, shuld be {}, or {}".format("hdfs", "local"))

    def dataset_get(self, batch_size, parallel, data_split_func):
        self._format_data_dir()
        if len(self.file_list) == 0:
            return []
        self.logger.info('*'*20+'file list'+'*'*20)
        self.logger.info(self.file_list)
        dataset = tf.data.TextLineDataset(self.file_list)
        if 'use_keras' in params and params['use_keras']:
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(buffer_size = 2 * batch_size)
            dataset = dataset.map(data_split_func,num_parallel_calls=parallel)
        else:
            dataset = dataset.map(data_split_func,num_parallel_calls=parallel)
            dataset = dataset.prefetch(buffer_size = 2 * batch_size)
            dataset = dataset.batch(batch_size)
        return dataset
