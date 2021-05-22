# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time: 2021/5/21 6:14
# @USER: 86199
# @File: Multi_threading
# @Software: PyCharm
# @Author: 张平路
------------------------------------------------- 
# @Attantion：
#    1、
#    2、
#    3、
-------------------------------------------------
"""
import numpy as np
import pandas as pd
import math
import random
import time
from threading import Thread
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

_result_list = []


def split_data(repeat,thread_count,compute, result_list):
    """

    :param data_list: the data need to be split
    :param split_count: the max number of threading
    :param compute: the function to compute result
    :param result_list: the result have the final result
    :return:
    """
    thread_list = [] # the list of threading

    for item in range(thread_count):
        thread = Thread(target=work, args=(repeat,item,compute,result_list))
        thread_list.append(thread)
        thread.start()

    # the main threading close after the child threading closing
    for thread_item in thread_list:
        thread_item.join()





def work(repeat, thread_count, compute,result_list):
    """
   每个线程执行的任务，让程序随机sleep几秒
    :param df:
    :param _list:
    :return:
    """
    sleep_time = random.randint(1, 5)
    logger.info(f'count is {thread_count},sleep {sleep_time}')
    # sleep random time
    time.sleep(sleep_time)
    compute(repeat,result_list)

# if __name__ == '__main__':
#
#     print(len(_result_list), _result_list)
