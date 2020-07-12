#!/usr/bin/env Python
# coding   : utf-8

import sys
import time
from time import sleep
from threading import Thread
from datafetch import getConfig
import json
import os
from loggerClass import Logger

check_book = None

def docParse(parser, interpreter, taskName, gConfig, lexer=None, debug=False, tracking=False):
    if gConfig['unittestIsOn'.lower()] == True:
        pass
    else:
        pass
    start_time = time.time()

    print("\n\npase (%s) file is starting!\n\n" % taskName)
    #parser.parse()
    taskResult = interpreter.doWork(parser,lexer=lexer,debug=debug,tracking=tracking)
    print('\n\nparse %s file end, time used %.4f' % (taskName, (time.time() - start_time)))
    return taskResult

def parseStart(gConfig, taskName, unittestIsOn):
    parser,interpreter = parserManager(taskName, gConfig)
    taskResult = docParse(parser, interpreter, taskName, gConfig)
    return taskResult

def parserManager(taskName, gConfig):
    module = __import__(check_book[taskName]['excelparser'],
                        fromlist=(check_book[taskName]['excelparser'].split('.')[-1]))
    excelParser = getattr(module,'create_object')(gConfig=gConfig)

    module = __import__(check_book[taskName]['sqlparser'],
                        fromlist=(check_book[taskName]['sqlparser'].split('.')[-1]))
    sqlParser = getattr(module,'create_object')(gConfig)

    module = __import__(check_book[taskName]['docparser'],
                        fromlist=(check_book[taskName]['docparser'].split('.')[-1]))
    docParser = getattr(module,'create_object')(gConfig)

    module = __import__(check_book[taskName]['interpreter'],
                        fromlist=(check_book[taskName]['interpreter'].split('.')[-1]))
    interpreter = getattr(module,'create_object')(gConfig,docParser,excelParser,sqlParser)

    return docParser,interpreter#,sourceFile,targetFile

def get_gConfig(taskName, gConfig, unittestIsOn):
    global check_book
    if check_book is not None:
        config_file = os.path.join(gConfig['config_directory'], check_book[taskName]['config_file'])
        config_file_json = os.path.join(gConfig['config_directory'], check_book[taskName]['config_file_json'])
    else:
        raise ValueError('check_book is None ,it may be some error occured when open the checkbook.json!')
    gConfig = getConfig.get_config(config_file)
    gConfigJson = getConfig.get_config_json(config_file_json)
    gConfig.update({"gConfigJson":gConfigJson})
    #在unitest模式,这三个数据是从unittest.main中设置，而非从文件中读取．
    gConfig['taskName'] = taskName
    gConfig['unittestIsOn'.lower()] = unittestIsOn
    if unittestIsOn:
        pass
    return gConfig

def set_check_book(gConfig):
    check_file = os.path.join(gConfig['config_directory'], gConfig['check_file'])
    global check_book
    if os.path.exists(check_file):
        with open(check_file, encoding='utf-8') as check_f:
            check_book = json.load(check_f)
    else:
        raise ValueError("%s is not exist,you must create first!" % check_file)

def validate_parameter(taskName, gConfig):
    assert taskName in gConfig['tasknamelist'], 'taskName(%s) is invalid,it must one of %s' % \
                                                 (taskName, gConfig['tasknamelist'])
    global check_book
    set_check_book(gConfig)
    return check_book[taskName]["config_file"] != '' \
           and check_book[taskName]['docparser'] != '' \
           and check_book[taskName]['sqlparser'] != '' \
           and check_book[taskName]['excelparser'] != '' \
           and check_book[taskName]['interpreter'] != ''

def run_task(taskName,gConfig,unittestIsOn):
    gConfig = get_gConfig(taskName, gConfig, unittestIsOn)
    taskResults = list()
    logger = Logger(gConfig,'execute').logger
    if taskName == 'batch':
        source_directory = os.path.join(gConfig['data_directory'],gConfig['source_directory'])
        sourcefiles = os.listdir(source_directory)
        for sourcefile in sourcefiles:
            logger.info('start process %s'%sourcefile)
            sourcefile = os.path.join(gConfig['source_directory'],sourcefile)
            gConfig.update({'sourcefile':sourcefile})
            taskResult = parseStart(gConfig,taskName,unittestIsOn)
            taskResults.append(taskResult)
    else:
        taskResult = parseStart(gConfig, taskName, unittestIsOn)
        taskResults.append(taskResult)
    logger.info(taskResults)

def main():
    gConfig = getConfig.get_config()
    if len(sys.argv) > 1 :
        if len(sys.argv) == 6:
            #该模式为从unittest.main调用
            unittestIsOn = bool(sys.argv[5])
            assert unittestIsOn == True , \
                'Now in dounittest mode, the num of argvs must be 5 whitch is taskName, framework,dataset and unittestIsOn'
        else:
            #该模式为从python -m 方式调用
            unittestIsOn = gConfig['unittestIsOn'.lower()]
        taskName = sys.argv[1]
    else:
        #该模式为从pycharm调用
        unittestIsOn = gConfig['unittestIsOn'.lower()]
        assert unittestIsOn == False, \
            'Now in training mode,unitestIsOn must be False whitch in configbase.txt'
        taskName = gConfig['taskName'.lower()]

    if validate_parameter(taskName,gConfig) == True:
        run_task(taskName,gConfig,unittestIsOn)
    else:
        raise ValueError("(%s %s %s %s) is not supported now!"%(gConfig))

if __name__=='__main__':
    main()


