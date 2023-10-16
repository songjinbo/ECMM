#coding=utf-8
import sys,os
from datetime import datetime,timedelta
from multiprocessing import Process

def DownloadParallel(hdfs_src, local_dst):
    cmd = """
        hdfs dfs -get {src} {dst}
    """.format(src=hdfs_src, dst=local_dst)
    if os.system(cmd) != 0:
        print('download hdfs fail! hdfs:'+hdfs_src)

def DownLoad(begin_day, end_day, hdfs_root, local_dst):
    begin_day_fm = datetime.strptime(begin_day, '%Y%m%d')
    end_day_fm = datetime.strptime(end_day, '%Y%m%d')

    day=begin_day_fm
    pool = list()
    while day < end_day_fm:
        p = Process(target=DownloadParallel, args=(hdfs_root+'/'+day.strftime('%Y%m%d'), local_dst))
        p.start()
        pool.append(p)
        day = day+timedelta(days=1)
    for pidx in pool:
        pidx.join()
