import unittest
from datafetch import getConfig
from interpreter import interpretAccounting
from execute import *

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)
        self.run_interpreter()

    def run_interpreter(self):
        gConfig = getConfig.get_config('config_directory/configbase.txt')
        gConfigJson = getConfig.get_config_json('config_directory/interpretAccounting.json')
        gConfig.update({"gConfigJson": gConfigJson})
        interpreter = interpretAccounting.create_object(gConfig=gConfig)
        #test lexer
        input = ''
        input = input + '-1,370,249,543.00  1234  1,234'
        input = input + '  .00 元'
        input = input + ' -0.51% 27.19%'
        input = input + '(0.05) 0.0045'
        input = input + ' 63340.SH'
        #input = input + 'www.see.com.cn'
        #input = input + 'irm@qianhefood.com' '  ).'
        input = input + ' 六.31 之 ' +' 五.41（. 3） '
        input = input + '（QA、QC） （单位：元 币种：人民币） （2019-2020 年） (吨) （中国证监\n会公告[2013]43 号）'
        input = input + '（证监发[2012]37\n号） (2017 年-2019\n年) （＋，－） -稳富  5114002017043-L'
        #input = input + '元 千元 人民币 科目'
        input = input + ' 2008 年 12月5  日' + ' 2019年'
        input = input + ' ..' + '  ..................'
        input = input + ' 合并资产负债表\n2019 年 12 月 31 日 '
        input = input + ' 元  人民币'
        input = input + ' 眉山.千禾博物馆'
        input = input + ' 合并利润表\n2019年 1—12 月'
        input = input + ' 万元  人民币'
        #input = input + ' 中兴通讯股份有限公司'
        interpreter.lexer.input(input)
        for tok in interpreter.lexer:
            print(tok)
        #test yaac
        interpreter.parser.parse(input,lexer=interpreter.lexer,debug=True,tracking=True)

if __name__ == '__main__':
    unittest.main()
