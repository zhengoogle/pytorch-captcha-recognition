#!/usr/bin/python3
from loguru import logger


# 输出开始结束时间差
def log_cost(start_time, end_time):
    timestamp = (end_time - start_time).seconds
    days = int(timestamp / (3600 * 24))
    hours = int(timestamp % (3600 * 24) / 3600)
    minutes = int(timestamp % 3600 / 60)
    seconds = timestamp % 3600 % 60
    logger.debug("total cost time: " + str(timestamp))
    logger.debug("{0}天{1}时{2}分{3}秒".format(days, hours, minutes, seconds))







