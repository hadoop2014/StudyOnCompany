import sys
import time
from time import sleep
from threading import Thread
from datafetch import getConfig
import json
import os

check_book = None

def docParse(parser,interpreter,docformat,gConfig,lexer=None,debug=False,tracking=False):
    if gConfig['unittestIsOn'.lower()] == True:
        pass
    else:
        pass
    start_time = time.time()

    print("\n\npase (%s) file is starting!\n\n"%docformat)
    #parser.parse()
    interpreter.doWork(parser,lexer=lexer,debug=debug,tracking=tracking)
    print('\n\nparse %s file end, time used %.4f'%(docformat,(time.time()-start_time)))

def parseStart(gConfig,docformat,unittestIsOn):
    gConfig = get_gConfig(docformat,gConfig,unittestIsOn)
    parser,interpreter = parserManager(docformat, gConfig)
    docParse(parser,interpreter,docformat, gConfig)

def parserManager(docformat,gConfig):
    module = __import__(check_book[docformat]['writeparser'],
                        fromlist=(check_book[docformat]['writeparser'].split('.')[-1]))
    writeParser = getattr(module,'create_object')(gConfig=gConfig)

    module = __import__(check_book[docformat]['sqlparser'],
                        fromlist=(check_book[docformat]['docparser'].split('.')[-1]))
    sqlParser = getattr(module,'create_object')(gConfig)

    module = __import__(check_book[docformat]['docparser'],
                        fromlist=(check_book[docformat]['docparser'].split('.')[-1]))
    docParser = getattr(module,'create_object')(gConfig,writeParser)
    #docParser = getattr(module, 'create_object')(gConfig, sqlParser)

    module = __import__(check_book[docformat]['interpreter'],
                        fromlist=(check_book[docformat]['interpreter'].split('.')[-1]))
    interpreter = getattr(module,'create_object')(gConfig,docParser)

    #sourceFile = gConfig['sourcefile']
    #targetFile = gConfig['targetfile']
    return docParser,interpreter#,sourceFile,targetFile

def get_gConfig(docformat,gConfig,unittestIsOn):
    global check_book
    if check_book is not None:
        config_file = os.path.join(gConfig['config_directory'],check_book[docformat]['config_file'])
        config_file_json = os.path.join(gConfig['config_directory'],check_book[docformat]['config_file_json'])
    else:
        raise ValueError('check_book is None ,it may be some error occured when open the checkbook.json!')
    gConfig = getConfig.get_config(config_file)
    gConfigJson = getConfig.get_config_json(config_file_json)
    gConfig.update({"gConfigJson":gConfigJson})
    #在unitest模式,这三个数据是从unittest.main中设置，而非从文件中读取．
    gConfig['docformat'] = docformat
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

def validate_parameter(docformat,gConfig):
    assert docformat in gConfig['docformatlist'], 'docformat(%s) is invalid,it must one of %s' % \
                                                (docformat, gConfig['docformatlist'])
    global check_book
    set_check_book(gConfig)
    return check_book[docformat]["config_file"] != '' and check_book[docformat]['docparser'] != ''

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
        docformat = sys.argv[1]
    else:
        #该模式为从pycharm调用
        unittestIsOn = gConfig['unittestIsOn'.lower()]
        assert unittestIsOn == False, \
            'Now in training mode,unitestIsOn must be False whitch in configbase.txt'
        docformat = gConfig['docformat']

    if validate_parameter(docformat,gConfig) == True:
        parseStart(gConfig, docformat, unittestIsOn)
    else:
        raise ValueError("(%s %s %s %s) is not supported now!"%(gConfig))

if __name__=='__main__':
    main()


