#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import unittest
from interpreter import interpretAccounting
from execute import *
from baseClass import *

class baseParser(baseClass):
    def __init__(self,gConfig):
        super(baseParser,self).__init__(gConfig['gConfigJson'])

    def _load_data(self,input = None):
        if input is None:
            input = list()
        self._data = input
        self._index = 0
        self._length = len(self._data)

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.gConfig = getConfig.get_config('config_directory/configbase.txt')
        gConfigJson = getConfig.get_config_json('config_directory/interpretAccounting.json')
        self.gConfig.update({"gConfigJson": gConfigJson})
        testParser = baseParser(self.gConfig)
        self.interpreter = interpretAccounting.create_object(gConfig=self.gConfig,docParser=testParser)
        self.run_interpreter_lexer()
        self.run_interpreter_yacc(testParser)

    def run_interpreter_lexer(self):
        #test lexer
        input = ''
        input = input + ' -1,370,249,543.00  1234  1,234 0.0045 40.51% (-0.05%)'
        input = input + ' 63340.SH'
        input = input + ' www.see.com.cn'
        input = input + ' irm@qianhefood.com'
        input = input + ' 千禾味业股份有限公司'
        input = input + ' 2008 年 12月5  日  2019年  2019-2020 年  2019 年 12 月 31 日  2017 年-2019\n年 2019年 1—12 月'
        input = input + ' 单位：元 币种：人民币 吨 万元 美元'
        input = input + ' 科目'
        input = input + ' 合并资产负债表\n2019 年 12 月 31 日'
        input = input + ' 元  人民币'
        input = input + ' 合并利润表\n2019年 1—12 月'
        input = input + ' 万元  人民币'
        input = input + ' 眉山.千禾博物馆'
        input = input + ' ..' + '  ..................'
        input = input + '（＋，－） -稳富  5114002017043-L'
        input = input + ' \t'
        #input = input + ' ). 1) 2） 六.31 之 ' +' 五.41（. 3） （前额和 安第几个 dijg）'
        #input = input + ' 司（以下简称“公司”或“本公司”，在包括子公司时统称“本\n集团”） -\n的其他应收'
        self.interpreter.lexer.input(input)
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(-,'-',1,1)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'1,370,249,543.00',1,2)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'1234',1,20)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'1,234',1,26)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'0.0045',1,32)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'40.51',1,39)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(%,'%',1,44)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken((,'(',1,46)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(-,'-',1,47)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'0.05',1,48)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(%,'%',1,52)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(),')',1,53)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'63340',1,55)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NAME,'SH',1,61)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(WEBSITE,'www.see.com.cn',1,64)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(EMAIL,'irm@qianhefood.com',1,79)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(COMPANY,'千禾味业股份有限公司',1,98)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2008 年 12月5  日',1,109)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019年',1,125)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019-2020 年',1,132)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019 年 12 月 31 日',1,145)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2017 年-2019\\n年',1,163)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019年 1—12 月',1,177)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(UNIT,'单位：元',1,190)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(CURRENCY,'币种：人民币',1,195)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(UNIT,'吨',1,202)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(UNIT,'万元',1,204)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(CURRENCY,'美元',1,207)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(HEADER,'科目',1,210)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TABLE,'合并资产负债表',1,213)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019 年 12 月 31 日',2,221)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(UNIT,'元',2,238)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(CURRENCY,'人民币',2,241)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TABLE,'合并利润表',2,245)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(TIME,'2019年 1—12 月',3,251)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(UNIT,'万元',3,264)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(CURRENCY,'人民币',3,268)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'眉山',3,272)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'千禾博物馆',3,275)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(（,'（',3,303)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(PUNCTUATION,'＋',3,304)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(PUNCTUATION,'－',3,306)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(）,'）',3,307)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(-,'-',3,309)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'稳富',3,310)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'5114002017043',3,314)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(-,'-',3,327)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NAME,'L',3,328)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"None")

    def run_interpreter_yacc(self,testParser):
        input = ''
        input = input + ' 合并资产负债表\n2019 年 12 月 31 日'
        input = input + ' 编制单位: 千禾味业食品股份有限公司'
        input = input + ' 元  人民币 '
        input = input + ' \n在职员工的数量合计（人） 2731'
        input = input + ' 合并资产负债表\n2019 年 12 月 31 日'
        input = input + ' 元  人民币 '
        input = input + ' 合并资产负债表\n编制单位：深圳华侨城股份有限公司'
        input = input + ' 2017 年 12 月 31 日 单位：元'
        input = input + ' 合并利润表\n单位：元 '
        input = input + ' 现金流量表补充资料 '
        input = input + ' (1).现金流量表补充资料 \n√适用 □不适用 \n 单位：元 币种：人民币'
        input = input + ' 千禾味业食品股份有限公司 2019 年年度报告'
        #input = input + ' ). 1) 2） 六.31 之  五.41（. 3）'
        input = input + ' \n公司代码：603027'
        input = input + ' --现金 --非现金资产的公允价值'
        input = input + ' \n公司代码：603027 公司简称：千禾味业'
        input = input + ' \n股票简称 亿纬锂能 股票代码 300014'
        input = input + ' (现金) (1) (12.33%) (%) 2)'
        input = input + ' 应纳税增值额(应纳税额按应纳 16%、13%、10%、9%、6% \n税销售额乘以适用税率扣除当\n期允许抵扣的进项税后的余额\n计算) '
        input = input + ' (应纳税额按应纳 (16%、13%、10%、9%、6% 有) \n税销售额乘以适用税率扣除当)'
        self.interpreter.lexer.input(input)
        for tok in self.interpreter.lexer:
            print(tok)
        #test yaac
        testParser._load_data([input])
        self.interpreter.doWork(testParser,lexer=self.interpreter.lexer,debug=True,tracking=True)
        #self.interpreter.parser.parse(testParser,lexer=self.interpreter.lexer,debug=True,tracking=True)

if __name__ == '__main__':
    unittest.main()
