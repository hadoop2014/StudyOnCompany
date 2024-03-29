#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserBaseClass.py
#from six import unichr

from loggerClass import *
import functools
import os
import io
import re
import sqlite3 as sqlite
import pysnooper
import utile
import shutil
from constant import *
from typing import Type, Callable, Optional
from pandas import DataFrame
import pandas as pd
import csv
from sklearn.preprocessing import MaxAbsScaler
import abc
import threading
import multiprocessing
import asyncio
import psutil
#from concurrent.futures import ProcessPoolExecutor
#数据读写处理的基类

# 用于定义单例,使用方法:
# class Example(metaclass = MetaSingleton)
class MetaSingleton(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, '_instance'):
            with MetaSingleton._instance_lock:
                if not hasattr(cls, '_instance'):
                    cls._instance = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instance


# 用于实现多进程的类, 采用@Multiprocess装饰函数即可, 最后调用Multiprocess.release(), 对需加进程锁的函数采用@Multiprocess.Lock装饰.
class Multiprocess():
    """
    用于装饰器
    例: 装饰InterPreterNature类的_process_single_parse方法, 使之采用多进程运行.
    1) 初始化时, python解释器调用__init__记录被装饰的函数
    2) python解释器扫描到被装饰的函数时,自动调用__get__函数, 将函数进行包装
    args:
        function - 被装饰的函数
    """
    processPool = []
    taskResults = []
    multiprocessingIsOn = False
    processQueue = multiprocessing.Queue()
    processLock = multiprocessing.Lock()
    semaphore = multiprocessing.Semaphore(multiprocessing.cpu_count())

    def __init__(self, function):
        self.function = function

    def __get__(self, instance, own):
        """
        被装饰的函数在调用时,首先会调用__get__函数对函数进行包装,使其按多进程方式运行.
        最后必须手工调用Multiprocess.release()等待进程结束
        args:
            instance - 为被装饰函数所属的对象
            own - 被装饰函数所属的类
        reutrn:
            wrap - 装饰器的内层函数
        """
        @functools.wraps(self.function)
        def function_wrap(*args):
            """
            对self.function函数进行装饰:
            1) 运行函数self.function前调用semaphore.acquire进行阻塞
            2) 运行函数,将运行结果放入到processQueue中
            3) 释放信号量semaphore
            args:
                cls - Multiprocess类
            reutrn:
                func_wrap - 被装饰过后的函数
            """
            result = self.function(instance, *args)
            self.processQueue.put(result)
            #self.taskResults.append(result)
            self.semaphore.release()
            return result

        def mutiprocess_wrap(*args):
            """
            判断multiprocessingIsOn标识是否设置,如果设置,则对函数进行多进程编程:
            1) 函数先通过function_wrap包装后, 放入进程中.
            2) processPool进程池中增加进程.
            3) 启动进程,然后在release函数中等待进程结束. 所以采用多进程编程后,最后必须调用Multiprocess.release()等待进程结束
            args:
                cls - Multiprocess类
            reutrn:
                func_wrap - 被装饰过后的函数
            """
            if not self.multiprocessingIsOn:
                return function_wrap(*args)
            else:
                self.semaphore.acquire()
                process = multiprocessing.Process(target=function_wrap, args=(*args,))
                process.start()
                self.processPool.append(process)
                if not self.processQueue.empty():
                    # 当processQueue队列满时,会导致子进程处在阻塞状态, 从而主进程死锁. 因此在join前必须执行一次self.processQueue.get().
                    self.taskResults.append(self.processQueue.get())
                    #instance.logger.info('Fetch data from processQueue, %d records!' % len(self.taskResults))
        return mutiprocess_wrap

    @classmethod
    def release(cls):
        """
        释放Multiprocess: 1) 等待进程池中所有的进程运行完, 然后清空进程池. 2) 取出进程队列里面所有的函数返回值.
        args:
            cls - MultiProcess类
        reutrn:
            taskResults - 进程返回值的列表
        """

        for process in cls.processPool:
            process.join()
        cls.processPool.clear()
        for _ in range(cls.processQueue.qsize()):
            cls.taskResults.append(str(cls.processQueue.get()))
        return cls.taskResults

    @classmethod
    def lock(cls, func):
        """
        对函数加进程锁,主要正对进程内如下场景:1)写文件. 2)写数据库.
        args:
            cls - Multiprocess类
        reutrn:
            func_wrap - 被装饰过后的函数,原理: 在运行函数func前, 先用processLock进行加锁.
        """
        @functools.wraps(func)
        def func_wrap(*args):
            with cls.processLock:
                return func(*args)
        return func_wrap


class SqilteBase():
    """
    用于操作Sqlite3数据库
    1) 对数据库的操作包括: 获取连接, 创建表, 丢弃表, 数据写入, 表是否存在, 记录是否存在, 执行脚本, 执行脚本文件
    2) _create_table, _write_to_sqlite3将被其所派生的子类覆盖
    3) 在每次继承后,必须调用工厂方法create_database来覆盖成员变量self.database,解决各子类的继承和派生问题.
    args:
        databasefile - 数据库所存放的文件路径+文件名
        logger - 当前实例的logger
    """
    def __init__(self, databasefile,logger):
        self._databasefile = databasefile
        self.logger = logger

    def _get_connect(self):
        #用于获取数据库连接
        return sqlite.connect(self._databasefile)

    def _create_tables(self,tableNames,dictTables, commonFields):
        # 用于想sqlite3数据库中创建新表
        conn = self._get_connect()
        cursor = conn.cursor()
        allTables = self._fetch_all_tables(cursor)
        allTables = list(map(lambda x: x[0], allTables))
        for tableName in tableNames:
            targetTableName =  tableName
            if targetTableName not in allTables:
                sql = " CREATE TABLE IF NOT EXISTS [%s] ( \n\t\t\t\t\t" % targetTableName
                for commonFiled, type in commonFields.items():
                    sql = sql + "[%s] %s\n\t\t\t\t\t," % (commonFiled, type)
                # 由表头转换生产的字段
                fieldFromHeader = dictTables[tableName]["fieldFromHeader"]
                if len(fieldFromHeader) != 0:
                    for field in fieldFromHeader:
                        sql = sql + "[%s] VARCHAR(20)\n\t\t\t\t\t," % field
                sql = sql[:-1]  # 去掉最后一个逗号
                # 创建新表
                standardizedFields = dictTables[tableName]['fieldName']
                #duplicatedFields = self._get_duplicated_field(standardizedFields)
                duplicatedFields = standardizedFields
                for fieldName in duplicatedFields:
                    if fieldName is not NaN:
                        if 'fieldType' in dictTables[tableName].keys() \
                            and fieldName in dictTables[tableName]['fieldType'].keys():
                            type = dictTables[tableName]['fieldType'][fieldName]
                        else:
                            type = 'NUMERIC'
                        sql = sql + "\n\t\t\t\t\t,[%s]  %s" % (fieldName, type)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库表%s成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e, ' 创建数据库表%s失败' % targetTableName)

                # 创建索引
                sql = "CREATE INDEX IF NOT EXISTS [%s索引] on [%s] (\n\t\t\t\t\t" % (targetTableName, targetTableName)
                sql = sql + ", ".join(str(field) for field, value in commonFields.items()
                                      if value.find('NOT NULL') >= 0)
                sql = sql + '\n\t\t\t\t\t)'
                try:
                    conn.execute(sql)
                    conn.commit()
                    print('创建数据库%s索引成功' % (targetTableName))
                except Exception as e:
                    # 回滚
                    conn.rollback()
                    print(e, ' 创建数据库%s索引失败' % targetTableName)
        cursor.close()
        conn.close()

    def _write_to_sqlite3(self,dataFrame:DataFrame, commonFields,tableName):
        conn = self._get_connect()
        dataFrame.to_sql(tableName, conn, if_exists='replace', index=False)
        conn.close()

    def _fetch_all_tables(self, cursor):
        #获取数据库中所有的表,用于判断待新建的表是否已经存在
        try:
            cursor.execute("select name from sqlite_master where type='table' order by name")
        except Exception as e:
            print(e)
        return cursor.fetchall()


    def _drop_table(self,conn,tableName):
        sql = 'drop table if exists \'{}\''.format(tableName)
        result = conn.execute(sql).fetchall()
        return result


    def _is_table_exist(self,conn, tableName):
        # 判断数据库中该表是否存在
        isTableExist = False
        sql = 'SELECT count(*) FROM sqlite_master WHERE type="table" AND name = \'{}\''.format(tableName)
        result = conn.execute(sql).fetchall()
        if len(result) > 0:
            isTableExist = result[0][0] > 0
        return isTableExist


    def _is_record_exist(self, conn, tableName, dataFrame:DataFrame,commonFields,specialKeys = None):
        #用于数据在插入数据库之前,通过组合的关键字段判断记录是否存在.
        #对于Sqlit3,字符串表示为'string' ,而不是"string".
        isRecordExist = False
        condition = self._get_condition(dataFrame,commonFields,specialKeys)
        if condition == NULLSTR:
            #condition为空时,说明dataFrame没有有效数据,直接返回False
            return isRecordExist
        sql = 'select count(*) from {} where '.format(tableName) + condition
        result = conn.execute(sql).fetchall()
        if len(result) > 0:
            isRecordExist = (result[0][0] > 0)
        return isRecordExist


    def _get_condition(self,dataFrame,commonFields, specialKeys = None):
        primaryKey = [key for key, value in commonFields.items() if value.find('NOT NULL') >= 0]
        if specialKeys is not None and isinstance(specialKeys,list):
            primaryKey = primaryKey + specialKeys
        # 对于Sqlit3,字符串表示为'string' ,而不是"string".
        joined = list()
        for key in primaryKey:
            if dataFrame[key].shape[0] == 0:
                joined = list()
                break
            current = '(' + ' or '.join(['{} = \'{}\''.format(key,value) for value in set(dataFrame[key].tolist())]) + ')'
            joined = joined + list([current])
        condition = NULLSTR
        if len(joined) > 0:
            condition = ' and '.join(joined)
        return condition


    @pysnooper.snoop()
    def _sql_executer(self,sql):
        isSuccess = False
        conn = self._get_connect()
        try:
            conn.execute(sql)
            conn.commit()
            self.logger.debug('success to execute sql(脚本执行成功):\n%s' % sql)
            isSuccess = True
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s\n%s' % (str(e),sql))
        conn.close()
        return isSuccess


    def _sql_executer_script(self,sql):
        isSuccess = False
        conn = self._get_connect()
        try:
            conn.executescript(sql)
            conn.commit()
            self.logger.info('success to execute sql(脚本执行成功)!')
            isSuccess = True
        except Exception as e:
            # 回滚
            conn.rollback()
            self.logger.error('failed to execute sql(脚本执行失败):%s' % (str(e)))
        conn.close()
        return isSuccess


class SpaceBase(metaclass=abc.ABCMeta):
    """
    虚拟基类,用于派生workingspace和loggingspace,分别用于处理工作目录和日志目录的各项工作
    args:
        directory - working_directory 或则 logging_directory
        logger - 当前实例的logger
    """
    def __init__(self, directory, logger):
        self.directory = directory
        self.logger = logger
        if os.path.exists(self.directory) == False:
            os.makedirs(self.directory)

    def clear_directory(self,directory):
        assert directory == self.directory ,\
            'It is only clear logging directory, but %s is not'%directory
        files = os.listdir(directory)
        for file in files:
            full_file = os.path.join(directory,file)
            if os.path.isdir(full_file):
                self.clear_directory(full_file)
            else:
                try:
                    os.remove(full_file)
                except:
                   self.logger('Failed to clear directory %s!'%full_file)

    def save(self, content, checkpoint_header=NULLSTR, drop_duplicate=NULLSTR, order_by=NULLSTR):
        ...

    def _remove_duplicate(self,checkpoint, content, checkpoint_header, drop_duplicate, order_by):
        ...

    def close(self):
        ...

    def _get_checkpoint(self) -> Optional[io.TextIOWrapper]:
        ...

    def get_content(self):
        ...

    def save_model(self, net, optimizer):
        ...

    def load_model(self, net, get_optimizer : Callable, ctx):
        ...

    def is_modelfile_exist(self):
        ...

    def model_filename(self) -> str:
        ...

    def processing_checkpoint(cls,func):
        ...

class WorkingspaceBase(SpaceBase):
    def __init__(self, working_directory, logger):
        super(WorkingspaceBase, self).__init__(working_directory, logger)

class LoggingspaceBase(SpaceBase):
    def __init__(self, logging_directory, logger):
        super(LoggingspaceBase, self).__init__(logging_directory, logger)

class CheckpointBase(WorkingspaceBase):
    """
    explain: 检查点基类
        从WorkingspaceBase派生, 即checkpoint存放在working_directory中,该类用于处理检查点的创建, 读取, 写入等
    args:
        working_directory - 检查点文件存放的目录
        logger - 当前实例的logger
        checkpointfile - 检查点文件名
        checkpointIsOn - 是否启用检查点功能
    """
    def __init__(self, working_directory, logger, checkpointfile, checkpointIsOn, max_keep_checkpoint,**kwargs):
        super(CheckpointBase,self).__init__(working_directory, logger)
        #checkpointfile = os.path.join(self.directory, checkpointfile)
        self.checkpointIsOn = checkpointIsOn
        self.prefix_checkpointfile = checkpointfile.split('.')[0]
        self.suffix_checkpointfile = checkpointfile.split('.')[-1]
        self.max_keep_checkpoint = max_keep_checkpoint
        self.checkpointfile = self._check_max_checkpoints(self.directory
                                                         , self.prefix_checkpointfile
                                                         , self.suffix_checkpointfile
                                                         , self.max_keep_checkpoint
                                                         , copy_file=True)
        self._init_checkpoint(self.checkpointfile)

    def _check_max_checkpoints(self, directory, prefix_checkpointfile, suffix_checkpointfile, max_keep_files, copy_file = False):
        """
        explain: 检查保存的检查点文件数量.
            如果directory目录下的检查点文件超过max_keep_checkpoints,则老化早期的检查点文件
        args:
            directory - 检查点文件所在的目录
            prefix_checkpointfile - 检查点文件的前缀名
            suffix_checkpointfile - 检查点文件的后缀名
            max_keep_checkpoints - 能保留的最大检查点文件个数
            copy_file - 把最新的文件拷贝一份, 文件名中的时间更新为当前时间. 对于checkpoint 设置copy_file = Ture, 对于model_savefile,则为False
        reutrn:
            checkpointfile - 检查点所保存的文件名
            1) 取directory目录下,日期最新的一个文件名,
            2) 如果directory目录下, 模型文件为空, 则构造一个日期为当天的模型文件名
        """
        files = os.listdir(directory)
        files = [file for file in files if self._is_file_needed(file, prefix_checkpointfile, suffix_checkpointfile)]
        files = sorted(files, reverse=True)
        current_filename = utile.construct_filename(directory, prefix_checkpointfile, suffix_checkpointfile)
        if len(files) > 0:
            checkpointfile = os.path.join(directory, files[0])
            if copy_file and current_filename != checkpointfile:
                # 拷贝一份文件,名称命名为current_filename
                shutil.copyfile(checkpointfile, current_filename)
            else:
                current_filename = checkpointfile

            if len(files) > max_keep_files:
                files_discard = files[max_keep_files:]
                for file in files_discard:
                    os.remove(os.path.join(directory, file))
            for file in files[:max_keep_files]:
                # 删除内容为空的文件
                fullfile = os.path.join(directory, file)
                if os.path.exists(fullfile) and os.path.getsize(fullfile) == 0:
                    os.remove(fullfile)
        return current_filename

    def _is_file_needed(self, fileName, prefix_filename, suffix_filename):
        isFileNeeded = False
        if prefix_filename != NULLSTR and fileName != NULLSTR:
            fileName = os.path.split(fileName)[-1]
            suffix = fileName.split('.')[-1]
            if utile.is_matched(prefix_filename, fileName) and suffix == suffix_filename:
                isFileNeeded = True
        return isFileNeeded

    def _init_checkpoint(self, checkpointfile):
        if self.checkpointIsOn:
            if not os.path.exists(checkpointfile):
                # 第一次创建checkpoint情况, 如果文件不存在,则创建它
                fw = open(checkpointfile,'w',newline='',encoding='utf-8')
                fw.close()

    def _get_checkpoint(self) -> Optional[io.TextIOWrapper]:
        if self.checkpointIsOn:
            checkpoint = open(self.checkpointfile, 'r+', newline='', encoding='utf-8')
        else:
            checkpoint = None
        return checkpoint

    @Multiprocess.lock
    def get_content(self):
        if self.checkpointIsOn == False:
            return list()
        checkpoint = self._get_checkpoint()
        reader = checkpoint.read().splitlines()
        checkpoint.close()
        return reader

    def is_file_in_checkpoint(self,content):
        if self.checkpointIsOn == False:
            return False
        reader = self.get_content()
        if content in reader:
            return True

    @Multiprocess.lock
    def save(self, content, checkpoint_header = NULLSTR, drop_duplicate = NULLSTR, order_by = NULLSTR):
        if self.checkpointIsOn == False:
            return
        checkpoint = self._get_checkpoint()
        reader = checkpoint.read().splitlines()
        reader = reader + [content]
        reader.sort()
        lines = [line + '\n' for line in reader]
        checkpoint.seek(0)
        checkpoint.truncate()
        checkpoint.writelines(lines)
        checkpoint.close()
        self.logger.info('Success to write checkpoint to file %s' % checkpoint.name)

    @Multiprocess.lock
    def remove_checkpoint_files(self,sourcefiles):
        assert isinstance(sourcefiles,list),'Parameter sourcefiles must be list!'
        if self.checkpointIsOn == False:
            return
        checkpoint = self._get_checkpoint()
        checkpoint.seek(0)
        reader = checkpoint.read().splitlines()
        resultfiles = list(set(reader).difference(set(sourcefiles)))
        resultfiles.sort()
        lines = [line + '\n' for line in resultfiles]
        checkpoint.seek(0)
        checkpoint.truncate()
        checkpoint.writelines(lines)
        checkpoint.close()
        removedfiles = list(set(reader).difference(set(resultfiles)))
        if len(removedfiles) > 0:
            removedlines = '\n\t\t\t\t'.join(removedfiles)
            self.logger.info("Success to remove from checkpointfile : %s"%(removedlines))


class StandardizeBase():
    """
    explain: 用于执行各种标准化
        1) 包括对company,reporttype,time,code, 文件名的标准化.
        2) 根据公司名company转化成标准化的公司代码code.
    args:
        data_directory - 数据所存放的目录,包括股票代码表
        logger - 日志记录器
        filenameStandardize - 文件名标准化的正则表达式
        companyStandardize - 公司名标准化的正则表达式
        reportTypeStandardize - 财报类型标准化的正则表达式
        codeStandardize - 公司代码标准化的正则表达式
        timeStandardize - 财报上报时间标准化的正则表达式
        tablePrefix - 表名的前缀, 取值为: 年度, 半年度, 季度
        reportTypes - 上报类型, 取值为: "年度报告","半年度报告","第一季度报告","第三季度报告"
        reportTypeAlias - 上报类型的别名
    """
    def __init__(self,data_directory, logger,
                 filenameStandardize, companyStandardize, reportTypeStandardize, codeStandardize,
                 timeStandardize, tablePrefix, reportTypes, reportTypeAlias, companyAlias, filenameAlias):
        self.data_directory = data_directory
        self.logger = logger
        self.filenameStandardize = filenameStandardize
        self.companyStandardize = companyStandardize
        self.reportTypeStandardize = reportTypeStandardize
        self.codeStandardize = codeStandardize
        self.timeStandardize = timeStandardize
        self.tablePrefix = tablePrefix
        self.reportTypes = reportTypes
        self.reportTypeAlias = reportTypeAlias
        self.companyAlias = companyAlias
        self.filenameAlias = filenameAlias


    def _get_company_time_type_code_by_filename(self, filename):
        time = self._get_time_by_filename(filename)
        type = self._get_report_type_by_filename(filename)
        company,code = self._get_company_code_by_content(filename)
        return company,time,type,code

    def _construct_standardize_filename(self,code, company, time, type):
        filename = '（' + code + '）' + company + '：' + time + type + '.PDF'
        return filename

    def _get_standardize_fileprefix(self, filename):
        standardize_filename = self._standardize(self.filenameStandardize, filename)
        company, time, reportType, code = self._get_company_time_type_code_by_filename(standardize_filename)  # 解决白云山:2020年第一季度报告,和白云山:2020年第一季度报告全文,取后者
        company = self._get_company_alias(company)
        if company is not NaN and time is not NaN and reportType is not NaN:
            standardize_filename = company + '：' + time + reportType
        return standardize_filename

    def _get_time_by_filename(self,filename):
        timeStandardize = self.timeStandardize #self.gJsonBase['timeStandardize'] # '\\d+年'
        time = self._standardize(timeStandardize, filename)
        return time

    def _is_year(self, year):
        '''
        explain: 判断是否为年
        '''
        isYear = False
        matchedYear = self._get_time_by_filename(year)
        if matchedYear is not NaN:
            isYear = True
        return isYear

    def _is_year_list(self, year_list):
        '''
        explain: 判断是否为年
        '''
        isYearList = [self._is_year(year) for year in year_list]
        return all(isYearList)

    def _get_company_code_by_content(self,content):
        codeStandardize = self.codeStandardize #self.gJsonBase['codeStandardize'] # （\\d+）
        code = self._standardize(codeStandardize, content)
        if code is not NaN:
            code = code.replace('（',NULLSTR).replace('）',NULLSTR)
        companyStandardize = self.companyStandardize #self.gJsonBase['companyStandardize']   # "[*A-Z]*[\\u4E00-\\u9FA5]+[A-Z0-9]*"
        company = self._standardize(companyStandardize, content)
        return company,code


    def _get_tablename_by_report_type(self, reportType, tableName):
        # 根据报告类型转换成相应的表名,如第一季度报告,合并资产负债表 转成 季报合并资产负债表
        assert reportType != NULLSTR, "reportType must not be NULL!"
        tablePrefix = self._get_tableprefix_by_report_type(reportType)
        return tablePrefix + tableName


    def _get_report_type_by_filename(self, filename):
        #assert,因为repair_table会传进来一个文件 通用数据：适用所有年度报告.xlsx 不符合标准文件名
        reportType = NULLSTR
        reportTypeStandardize = self.reportTypeStandardize #self.gJsonBase['reportTypeStandardize']  # "\\d+年([\\u4E00-\\u9FA5]+)"
        matched = re.findall(reportTypeStandardize, filename)
        if matched is not None and len(matched) > 0:
            type = matched.pop()
            reportType = self._get_report_type_alias(type)
        return reportType


    def _get_path_by_report_type(self, type):
        path = NULLSTR
        if type in self.reportTypes:
            path = os.path.join(self.data_directory,type)
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            self.logger.error("type(%s) is invalid ,which not in [%s] " % (type, self.reportTypes))
        return path


    def _get_path_by_filename(self, filename):
        type = self._get_report_type_by_filename(filename)
        path = self._get_path_by_report_type(type)
        return path

    def _standardize(self,fieldStandardize,field):
        standardizedField = field
        if isinstance(field, str) and isinstance(fieldStandardize, str) and fieldStandardize !="":
            matched = re.search(fieldStandardize, field)
            if matched is not None:
                standardizedField = matched[0]
            else:
                standardizedField = NaN
        return standardizedField

    def _get_tableprefix_by_report_type(self, reportType):
        assert reportType != NULLSTR,"reportType must not be NULL!"
        tablePrefix = NULLSTR
        dictTablePrefix = self.tablePrefix #self.gJsonBase['tablePrefix']
        if reportType in dictTablePrefix.keys():
            tablePrefix = dictTablePrefix[reportType]
        else:
            self.logger.error('reportType(%s) is invalid,it must one of %s'%(reportType,dictTablePrefix.keys()))
        return tablePrefix

    def _get_report_type_alias(self, reportType):
        aliasedReportType = NULLSTR
        reportTypeTotal = set(list(self.reportTypeAlias.keys()) + self.reportTypes)
        if reportType in reportTypeTotal:
            aliasedReportType = utile.alias(reportType, self.reportTypeAlias)
        return aliasedReportType

    def _get_company_alias(self,company):
        aliasedCompany = utile.alias(company, self.companyAlias)
        return aliasedCompany

    def _get_filename_alias(self,filename):
        aliasedFilename = utile.alias(filename, self.filenameAlias)
        return aliasedFilename

    def _is_file_name_valid(self,fileName):
        assert fileName != None and fileName != NULLSTR, "filename (%s) must not be None or NULL" % fileName
        isFileNameValid = False
        type = self._get_report_type_by_filename(fileName)
        if type != NULLSTR:
            isFileNameValid = True
        return isFileNameValid

    def _get_stockcode_dict(self, companyList) -> list:
        ...

    def _save_stockcode_list(self, stockList):
        ...

    def _get_stockcode_list(self, companyList) -> list:
        ...

    def _get_company_list(self, stockcodeList) -> list:
        ...

    def _get_company_by_code(self, stockcode) -> str:
        ...

class StandardizeStockcode(StandardizeBase):
    def __init__(self,stockcodefile, stockcodeHeader, dictStockCode, *args, **kwargs):
        super(StandardizeStockcode, self).__init__(*args, **kwargs)
        self.stockcodefile = os.path.join(self.data_directory, stockcodefile)
        self.stockcodeHeader = stockcodeHeader
        self.dictStockCode = dictStockCode
        self.cacheStockcodeDict = None

    def _get_stockcode_frame(self):
        stockcodeSpecial = [[company, code] for company, code in self.dictStockCode.items()]
        stockcodeHeader = self.stockcodeHeader
        if os.path.exists(self.stockcodefile):
            dataFrame = pd.read_csv(self.stockcodefile, names=stockcodeHeader, dtype=str)
            # stockcodeSpecial只有两列, 而stockcodeHeader有三列,最后一列为公司, 需要特殊处理
            dataFrameSpecial = pd.DataFrame(stockcodeSpecial, columns=stockcodeHeader[0:-1])
            dataFrameSpecial[stockcodeHeader[-1]] = '公司'
            dataFrame = dataFrame.append(dataFrameSpecial)
        else:
            dataFrame = pd.DataFrame(stockcodeSpecial, columns=stockcodeHeader)
        dataFrame = dataFrame.drop_duplicates()
        return dataFrame

    def _get_stockcode_dict(self, companys_or_codes, fieldIndex = 0):
        '''
        explain: 根据公司简称/公司代码列表返回[公司简称,公司代码,类型]的列表
        args:
            companys_or_codes - 公司简称/公司代码组成的列表
            fieldIndex - stockcodeHeader的索引, stockcodeHeader为: ["公司简称","公司代码","类型"]
                1) fieldIndex = 0时 companys_or_codes必须携带companys即公司简称列表
                2) fieldIndex = 1时 companys_or_codes必须携带sockcodes即公司代码列表
        return:
            stockList - [公司简称,公司代码,类型]组成的列表
        '''
        assert isinstance(companys_or_codes, list), "Parameter companyList (%s) must be a list" % type(companys_or_codes)
        stockList = []
        if len(companys_or_codes) == 0:
            return stockList
        stockcodeFrame = self._get_stockcode_frame()
        indexNeeded = stockcodeFrame[self.stockcodeHeader[fieldIndex]].isin(companys_or_codes) # 获取公司名称列表
        stockcodeFrame = stockcodeFrame[indexNeeded]
        stockList = stockcodeFrame.values.tolist()
        companyDiffer = set(companys_or_codes).difference(set([company[fieldIndex] for company in stockList]))
        if len(companyDiffer) > 0:
            self.logger.info("failed to get these stock list :%s, please check it at stockcode of interpreterBase.json" % (companyDiffer))
        return stockList

    def _get_stockcode_list(self, companyList):
        stockcodeDict = self._get_stockcode_dict(companyList)
        stockcodeFrame = pd.DataFrame(stockcodeDict, columns=self.stockcodeHeader)
        stockcodeFrame = stockcodeFrame[self.stockcodeHeader[1]]  # 获取公司代码列表
        stockcodeList = stockcodeFrame.values.tolist()
        return stockcodeList

    def _get_company_by_code(self, stockcode):
        if self.cacheStockcodeDict is None:
            stockcodeFrame = self._get_stockcode_frame()
            self.cacheStockcodeDict = dict(stockcodeFrame[[self.stockcodeHeader[1], self.stockcodeHeader[0]]].values.tolist())
        company = self.cacheStockcodeDict.get(stockcode,NULLSTR)
        company = self._get_company_alias(company)
        return company

    def __get__(self, instance, owner):
        return instance

    def _get_company_list(self, stockcodeList):
        '''
        explain: 根据公司代码列表返回公司简称列表
        args:
            stockcodeList - 公司代码组成的列表
        return:
            companyList - 公司简称组成的列表
        '''
        stockcodeDict = self._get_stockcode_dict(stockcodeList, fieldIndex = 1)
        stockcodeFrame = pd.DataFrame(stockcodeDict, columns=self.stockcodeHeader)
        stockcodeFrame = stockcodeFrame[self.stockcodeHeader[0]]  # 获取公司简称列表
        companyList = stockcodeFrame.values.tolist()
        return companyList

    def _save_stockcode_list(self, stockList):
        assert isinstance(stockList, list),"parameter stockList(%s) must be a list!"% stockList
        if os.path.exists(self.stockcodefile):
            os.remove(self.stockcodefile)
        stockList = sorted(stockList,key=lambda x: x[2] + x[1])
        stockcodefile = open(self.stockcodefile, 'w', newline= '', encoding= 'utf-8')
        stockcodefileWriter = csv.writer(stockcodefile)
        stockcodefileWriter.writerows(stockList)
        stockcodefile.close()
        self.logger.info('sucess to write stock code into file %s'% self.stockcodefile)


class BaseClass(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.gJsonInterpreter = gConfig['gJsonInterpreter'.lower()]
        self.gJsonBase = gConfig['gJsonBase'.lower()]
        #self.debugIsOn = gConfig['debugIsOn'.lower()]
        self.logger = Logger(gConfig,self._get_module_name()).logger # 不同的类继承BaseClass时,logger采用不同的名字
        self.database = self.create_database(SqilteBase)
        self.program_directory = gConfig['program_directory']
        self.workingspace = self.create_space(WorkingspaceBase)
        self.loggingspace = self.create_space(LoggingspaceBase)
        self.checkpoint = self.create_space(CheckpointBase)
        self.standard = self.create_standard(StandardizeBase)
        self.standardStockcode = self.create_standard(StandardizeStockcode)
        self.data_directory = gConfig['data_directory']
        self.unitestIsOn = gConfig['unittestIsOn'.lower()]
        self._data = list()
        self._index = 0
        self._length = len(self._data)
        # 编译器,文件解析器共同使用的关键字
        self.commonFields = self.gJsonBase['公共表字段定义']
        self.tableNames = self.gJsonBase['TABLE'].split('|')
        self.dictTables = {keyword: value for keyword,value in self.gJsonBase.items() if keyword in self.tableNames}
        # 此处变量用于多进程
        Multiprocess.multiprocessingIsOn = gConfig['multiprocessingIsOn'.lower()]

    def __iter__(self):
        return self


    def __next__(self):
        try:
            data = self._data[self._index]
        except IndexError:
            raise StopIteration
        self._index += 1
        return data


    def __getitem__(self, item):
        return self._data[item]


    def create_space(self, SpaceBase, **kwargs) -> SpaceBase:
        if WorkingspaceBase.__subclasscheck__(SpaceBase):
            space_directory = os.path.join(self.gConfig['working_directory'], self._get_module_path())
            if CheckpointBase.__subclasscheck__(SpaceBase):
                spacebase = SpaceBase(space_directory,
                                      self.logger,
                                      self.gConfig['checkpointfile'],
                                      self.gConfig['checkpointIsOn'.lower()],
                                      self.gConfig['max_keep_checkpoint'],
                                      **kwargs)
            else:
                spacebase = SpaceBase(space_directory, self.logger)
        elif LoggingspaceBase.__subclasscheck__(SpaceBase):
            space_directory = os.path.join(self.gConfig['logging_directory'], self._get_module_path())
            spacebase = SpaceBase(space_directory, self.logger)
        else:
            raise NotImplementedError('create_space class %s is not implement' % SpaceBase)
        return spacebase


    def create_database(self, DataBase) -> SqilteBase:
        # SqilteBase及其派生类的工厂方
        databasefile = os.path.join(self.gConfig['working_directory'], self.gConfig['database'])
        return DataBase(databasefile, self.logger)

    def create_standard(self, StandardizeBase, **kwargs) -> StandardizeBase:
        if StandardizeStockcode.__subclasscheck__(StandardizeBase):
            standard = StandardizeBase(self.gConfig['stockcodefile'],self.gJsonBase['stockcodeHeader'],
                                       self.gJsonBase['stockcode'],
                                       self.gConfig['data_directory'], self.logger,
                                       self.gJsonBase['filenameStandardize'], self.gJsonBase['companyStandardize'],
                                       self.gJsonBase['reportTypeStandardize'], self.gJsonBase['codeStandardize'],
                                       self.gJsonBase['timeStandardize'], self.gJsonBase['tablePrefix'],
                                       self.gJsonBase['reportType'],self.gJsonBase['reportTypeAlias'],
                                       self.gJsonBase['companyAlias'], self.gJsonBase['filenameAlias'],
                                       **kwargs)
        else:
            standard = StandardizeBase(self.gConfig['data_directory'], self.logger,
                                       self.gJsonBase['filenameStandardize'], self.gJsonBase['companyStandardize'],
                                       self.gJsonBase['reportTypeStandardize'], self.gJsonBase['codeStandardize'],
                                       self.gJsonBase['timeStandardize'], self.gJsonBase['tablePrefix'],
                                       self.gJsonBase['reportType'],self.gJsonBase['reportTypeAlias'],
                                       self.gJsonBase['companyAlias'], self.gJsonBase['filenameAlias'])
        return standard


    def _get_interpreter_keyword(self):
        # 编译器,文件解析器共同使用的关键字
        ...

    def _get_dict_tables(self,tableNames,dictTablesBase):
        """
            该函数目前只用在interpreterNature,interpreterCrawl,interpreterAnalysize中, 这些解释器用到了interpreterBase.json中配置的公共表字段定义
            而interpreterAccounting不需要调用该函数, 他用了一套独立的 公共表字段定义
            args:
                tableNames - 当前解释器下所能读取到的表名列表, 一般配置在 interpreterXXXX.json的 TABLE 关键字下
                dictTablesBase - 定义在interpreterBase.json下的 通用表配置
            reutrn:
                dictTables - 当前interpreterXXXX.json配置的表和interpreterBase.json配置的表进行融合, 融合的规则:
                '''
                1) dictTablesBase.keys()中的表名和tableNames中的表名重合,则从dictTablesBase中取出该表配置放到dictTable中;
                2) 当前解释器配置文件interpreterXXXX.json中配置的表配置, 更新到第一步的dictTable中
                '''
        """
        dictTables = {keyword: value for keyword, value in dictTablesBase.items() if
                           keyword in tableNames}
        dictTables.update({keyword: value for keyword, value in self.gJsonInterpreter.items() if
                           keyword in tableNames})
        # 如果表的配置中,还有 parent这段,则要和父表的字段进行合并,合并的原则: 1) 子表的value是一个值,覆盖附表；2)value是列表,则追加到父表；3)value是dict,则进行递归
        for tableName, dictTable in dictTables.items():
            parent = dictTable['parent']
            if parent != NULLSTR:
                mergedDictTable = self._merged_dict_table(dictTable,dictTables[parent])
                dictTables[tableName].update(mergedDictTable)
        return dictTables


    def _merged_dict_table(self,dictTable,dictTableParent):
        """
        args:
            dictTable - 当前表的配置参数
            dictTableParent - 父表的配置参数
        reutrn:
            dictTableMerged - 当前表和父表融合后的配置, 融合的规则:
            '''
            1) 当前表的value是一个值, 则覆盖父表;
            2) 当前表的value是一个list,则追加到父表；
            3) 当前表的value是一个dict,则进行递归调用;
            '''
        """
        dictTableMerged = dictTableParent.copy()
        for key,value in dictTable.items():
            # 遍历子表的值, 和父表进行合并
            if isinstance(value, list):
                # 如果子表的值是列表,则追加到父表
                if len(value) != 0:
                    dictTableMerged.setdefault(key,[]).extend(value)
            elif isinstance(value, dict):
                # 如果子表的值是dict, 则进行递归
                dictTableMergedChild = self._merged_dict_table(value, dictTableMerged[key])
                dictTableMerged.update({key: dictTableMergedChild})
            else:
                # 如果子表的值非上述几中,则覆盖父表
                if key != 'parent':
                    # 避免迭代循环
                    dictTableMerged.update({key: value})
        return dictTableMerged


    def _set_dataset(self,index=None):
        if isinstance(index,list):
            self._data = [page for i, page in enumerate(self._data) if i in index]
            self._index = 0
            self._length = len(self._data)


    def _get_token_type(self, local_name,value,typeLict,defaultType):
        #根据传入的TypeList,让lexer从defaultType中进一步细分出所需的type(从TypeList中选出)
        #Local_name中保存了每个Type所对应的正则表达式
        #VALUE为lexer所识别的值
        assert isinstance(typeLict,list),"parameter typeList must be a list!"
        type = defaultType
        for key in typeLict:
            match = re.search(local_name[key],value)
            if match is not None:
                type = key.split('_')[-1]
                break
        return type


    def _get_text(self,page):
        return page


    def _merge_table(self, dictTable=None,interpretPrefix=None):
        if dictTable is None:
            dictTable = list()
        return dictTable


    def _write_table(self,tableName,table):
        pass


    def _close(self):
        pass


    def _debug_info(self):
        pass


    def _get_module_path(self):
        module = self.__class__.__module__
        path = os.path.join(*module.split('.'))
        return path


    def _get_module_name(self):
        module = self.__class__.__module__
        module_name = module.split('.')[-1].lower()
        return module_name


    def _get_merged_columns(self,tableName):
        mergedColumns = [key for key in self.commonFields.keys() if key != "ID"]
        mergedColumns = mergedColumns + self.dictTables[tableName]['fieldName']
        return mergedColumns


    @property
    def index(self):
        return self._index - 1
