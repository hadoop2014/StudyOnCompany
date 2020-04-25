import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
from datafetch import getConfig
import json
import os

check_book = None

def set_check_book(gConfig):
    check_file = os.path.join(gConfig['config_directory'], gConfig['check_file'])
    global check_book
    if os.path.exists(check_file):
        with open(check_file, encoding='utf-8') as check_f:
            check_book = json.load(check_f)
    else:
        raise ValueError("%s is not exist,you must create first!" % check_file)


def validate_parameter(taskName,framework,dataset,mode,gConfig):
    assert taskName in gConfig['tasknamelist'], 'taskName(%s) is invalid,it must one of %s' % \
                                                (taskName, gConfig['tasknamelist'])
    assert framework in gConfig['frameworklist'], 'framework(%s) is invalid,it must one of %s' % \
                                                  (framework, gConfig['frameworklist'])
    assert dataset in gConfig['datasetlist'], 'dataset(%s) is invalid,it must one of %s' % \
                                              (dataset, gConfig['datasetlist'])
    assert mode in gConfig['modelist'], 'mode(%s) is invalid,it must one of %s' % \
                                              (dataset, gConfig['modelist'])
    global check_book
    set_check_book(gConfig)
    return check_book[taskName][framework][mode][dataset] and check_book[taskName][framework][mode]['model'] != ''

def main():
    gConfig = getConfig.get_config()
    if len(sys.argv) > 1 :
        if len(sys.argv) == 6:
            #该模式为从unittest.main调用
            unittestIsOn = bool(sys.argv[5])
            assert unittestIsOn == True , \
                'Now in unittest mode, the num of argvs must be 5 whitch is taskName, framework,dataset and unittestIsOn'
        else:
            #该模式为从python -m 方式调用
            unittestIsOn = gConfig['unittestIsOn'.lower()]
    else:
        #该模式为从pycharm调用
        unittestIsOn = gConfig['unittestIsOn'.lower()]
        assert unittestIsOn == False, \
            'Now in training mode,unitestIsOn must be False whitch in configbase.txt'

    if validate_parameter(gConfig) == True:
        pass
    else:
        raise ValueError("(%s %s %s %s) is not supported now!"%(gConfig))

if __name__=='__main__':
    main()


