#!/usr/bin/env Python
# coding   : utf-8

import sys
from datafetch import getConfig
from interpreterAssemble import InterpreterAssemble

'''
check_book = None

def docParse(parser, interpreter, taskName, gConfig, lexer=None, debug=False, tracking=False):
    if gConfig['unittestIsOn'.lower()] == True:
        pass
    else:
        pass
    start_time = time.time()

    print("\n\n%s %s parse is starting!\n\n" % (os.path.split(parser.sourceFile)[-1],taskName))
    taskResult = interpreter.doWork(parser,lexer=lexer,debug=debug,tracking=tracking)
    print('\n\nparse %s file end, time used %.4f' % (taskName, (time.time() - start_time)))
    return taskResult

def parseStart(gConfig, taskName, unittestIsOn):
    parser,interpreter = parserManager(taskName, gConfig)
    taskResult = docParse(parser, interpreter, taskName, gConfig)
    return taskResult

def parserManager(taskName, gConfig):
    check_book = getConfig.get_check_book()
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
    check_book = getConfig.get_check_book()
    if check_book is not None:
        config_file = os.path.join(gConfig['config_directory'], check_book[taskName]['config_file'])
        config_file_json = os.path.join(gConfig['config_directory'], check_book[taskName]['config_file_json'])
    else:
        raise ValueError('check_book is None ,it may be some error occured when open the checkbook.json!')
    gConfig = getConfig.get_config(config_file)
    gJsonAccounting,gJsonBase = getConfig.get_config_json(config_file_json)
    gConfig.update({"gJsonInterpreter".lower():gJsonAccounting})
    gConfig.update({"gJsonBase".lower():gJsonBase})
    #在unitest模式,这三个数据是从unittest.main中设置，而非从文件中读取．
    gConfig['taskName'] = taskName
    gConfig['unittestIsOn'.lower()] = unittestIsOn
    if unittestIsOn:
        pass
    return gConfig

def validate_parameter(taskName, gConfig):
    assert taskName in gConfig['tasknamelist'], 'taskName(%s) is invalid,it must one of %s' % \
                                                 (taskName, gConfig['tasknamelist'])
    check_book = getConfig.get_check_book()
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
            gConfig.update({'sourcefile':sourcefile})
            if not is_file_name_valid(sourcefile):
                logger.warninging("%s is not a valid file"%sourcefile)
                continue
            taskResult = parseStart(gConfig,taskName,unittestIsOn)
            taskResults.append(taskResult)
    else:
        taskResult = parseStart(gConfig, taskName, unittestIsOn)
        taskResults.append(taskResult)
    logger.info(taskResults)

def is_file_name_valid(fileName):
    assert fileName != None and fileName != NULLSTR,"filename (%s) must not be None or NULL"%fileName
    isFileNameValid = False
    pattern = '年度报告|季度报告'
    if isinstance(pattern, str) and isinstance(fileName, str):
        if pattern != NULLSTR:
            matched = re.search(pattern, fileName)
            if matched is not None:
                isFileNameValid = True
    return isFileNameValid
'''

def run_task_in_nature():
    interpreterNature = InterpreterAssemble().get_interpreter_nature()
    interpreterNature.doWork(debug=False)


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
    else:
        #该模式为从pycharm调用
        unittestIsOn = gConfig['unittestIsOn'.lower()]
        assert unittestIsOn == False, \
            'Now in training mode,unitestIsOn must be False whitch in configbase.ini'
    run_task_in_nature()


if __name__=='__main__':
    main()


