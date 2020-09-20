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
        interpreterAccounting.initialize()
        interpreterAccounting.docParser._set_dataset(list([0,105,106,107]))
        interpreterAccounting.doWork(debug=True,tracking=False)


if __name__ == '__main__':
    unittest.main()
