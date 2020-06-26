#!/usr/bin/env Python
# coding=utf-8
# @Time    : 12/9/2019 5:03 PM
# @Author  : wu.hao
# @File    : docParserPdf.py
# @Note    : 用于年报,半年报,季报pdf文件的读写

import unittest
from interpreter import interpretAccounting
from execute import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        self.gConfig = getConfig.get_config('config_directory/configbase.txt')
        gConfigJson = getConfig.get_config_json('config_directory/interpretAccounting.json')
        self.gConfig.update({"gConfigJson": gConfigJson})
        self.run_interpreter_yacc()

    def run_interpreter_yacc(self):
        docformat = self.gConfig['docformat']
        unittestIsOn = self.gConfig['unittestIsOn'.lower()]
        if validate_parameter(docformat, self.gConfig) == True:
            parser, interpreter = parserManager(docformat, self.gConfig)
            # 2019年千和味业年报合并资产负债表所在的页数为71,72,73,合并利润表为76,77,78
            parser._set_dataset(list([0,1,83,84,91,92]))
            docParse(parser, interpreter, docformat, self.gConfig,lexer=None,debug=False,tracking=False)
        else:
            raise ValueError("(%s %s %s %s) is not supported now!" % (self.gConfig))
        #self.interpreter = interpretAccounting.create_object(gConfig=self.gConfig,docParser=testParser)
        #self.run_interpreter_yacc(testParser)

if __name__ == '__main__':
    unittest.main()
