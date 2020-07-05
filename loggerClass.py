#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 7/5/2020
# @Author  : wu.hao
# @File    : loggerClass.py

import sys
import logging
from logging import handlers
# 默认日志格式
DEFAULT_LOG_FMT = "%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s: %(message)s"
# 默认时间格式
DEFUALT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
#把Logger变成为一个单例对象

class Logger():

    def __init__(self,gConfig,suffixName):
        # 1. 获取一个logger对象
        dictLogger = gConfig['logger']
        logging_name = dictLogger['LOGGING_NAME'.lower()] + '_' + suffixName
        self._logger = logging.getLogger(logging_name)
        # 2. 设置format对象
        self.formatter = logging.Formatter(fmt=DEFAULT_LOG_FMT, datefmt=DEFUALT_LOG_DATEFMT)
        # 3. 设置日志输出
        # 两者也可选其一
        # 3.1 设置文件日志模式
        self._logger.addHandler(self._get_file_handler(dictLogger['DEFAULT_LOG_FILENAME'.lower()],
                                                       int(dictLogger['LOGGING_BACKUP_COUNT'.lower()])))
        # 3.2 设置终端日志模式
        self._logger.addHandler(self._get_console_handler())
        # 4. 设置日志等级
        self._logger.setLevel(dictLogger['DEFAULT_LOG_LEVEL'.lower()])

    def _get_file_handler(self, filename,backupCount):
        '''返回一个文件日志handler'''
        # 1. 获取一个文件日志handler
        #filehandler = logging.FileHandler(filename=filename, encoding="utf-8")
        filehandler = handlers.TimedRotatingFileHandler(filename=filename, when="midnight", interval=1,
                                                        backupCount=backupCount,
                                                        atTime=None,encoding="utf-8")
        # 2. 设置日志格式
        filehandler.setFormatter(self.formatter)
        # 3. 返回
        return filehandler

    def _get_console_handler(self):
        '''返回一个输出到终端日志handler'''
        # 1. 获取一个输出到终端日志handler
        console_handler = logging.StreamHandler(sys.stdout)
        # 2. 设置日志格式
        console_handler.setFormatter(self.formatter)
        # 3. 返回handler
        return console_handler

    @property
    def logger(self):
        return self._logger


