#!/usr/bin/env Python
# coding   : utf-8

import sys
from datafetch import getConfig
from interpreterAssemble import InterpreterAssemble


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


