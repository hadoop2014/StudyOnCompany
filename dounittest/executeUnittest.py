#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : executeUnittest.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

import unittest
from interpreterAssemble import *


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        self.run_interpreter_yacc()


    def run_interpreter_yacc(self):
        interpreterAccounting = InterpreterAssemble().interpreter_assemble('accounting')
        interpreterAccounting.initialize(dictParameter={'sourcefile':interpreterAccounting.gConfig['sourcefile']})
        #interpreterAccounting.docParser._set_dataset(list([0,1,2,3,4,8,44,83,84,85]))
        #interpreterAccounting.docParser._set_dataset(list([0, 5,9,19,20,37,38,65,84,89]))
        #interpreterAccounting.docParser._set_dataset(list([0,1,3,6,7,8,9,13,24,53,73,97,127]))
        #interpreterAccounting.docParser._set_dataset(list([0,89,90,91,92,93,94,95,96,97,98,99,100,101,102,158,175]))
        interpreterAccounting.docParser._set_dataset(list([0,20,21,22,23,24,25,26,27,28]))
        interpreterAccounting.docParser.remove_checkpoint_files(list([interpreterAccounting.gConfig['sourcefile']]))
        interpreterAccounting.doWork(debug=True,tracking=False)


if __name__ == '__main__':
    unittest.main()
