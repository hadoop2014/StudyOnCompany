#!/usr/bin/env Python
# coding   : utf-8

import unittest
from interpreter import interpretAccounting
from execute import *
from baseClass import *

class BaseParser(BaseClass):
    def __init__(self,gConfig):
        super(BaseParser, self).__init__(gConfig)

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
        testParser = BaseParser(self.gConfig)
        self.interpreter = interpretAccounting.create_object(gConfig=self.gConfig, docParser=testParser)
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
        input = input + ' 1．不能重分类进损益的其他综合收益'
        input = input + ' 减：所得税费用'
        input = input + ' （一）归属母公司所有者的其他综合收益的税后净额'
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
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(NUMERIC,'1',3,332)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'不能重分类进损益的其他综合收益',3,334)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'减',3,350)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'所得税费用',3,352)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(（,'（',3,358)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'一',3,359)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(）,'）',3,360)")
        self.assertEqual(self.interpreter.lexer.token().__str__(),"LexToken(DISCARD,'归属母公司所有者的其他综合收益的税后净额',3,361)")
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
        input = input + ' 公司近三年（含报告期）的普通股股利分配方案或预案、资本公积金转增股本方案或预案\n单位：元 币种：人民币'
        input = input + ' 2019年年度报告 ' \
                        '\n七、 近三年主要会计数据和财务指标 ' \
                        '\n(一) 主要会计数据 ' \
                        '\n单位：元  币种：人民币 ' \
                        '\n ' \
                        '\n本期比上年同期增' \
                        '\n主要会计数据  2019年  2018年  2017年 ' \
                        '\n减(%) ' \
                        '\n营业收入  1,894,218,317.34  1,546,043,486.35  22.52  1,179,727,843.01 ' \
                        '\n归属于上市公司股东的净 463,073,018.65  332,092,898.42  39.44  216,574,727.65 ' \
                        '\n利润 ' \
                        '\n归属于上市公司股东的扣 453,153,425.68  326,953,777.64  38.60  212,920,593.45 ' \
                        '\n除非经常性损益的净利润 ' \
                        '\n经营活动产生的现金流量 647,843,745.11  469,415,593.60  38.01  346,553,967.15 ' \
                        '\n净额 ' \
                        '\n  本期末比上年同期' \
                        '\n2019年末  2018年末  2017年末 ' \
                        '\n末增减（%） ' \
                        '\n归属于上市公司股东的净 1,785,441,698.82  1,315,208,572.85  35.75  992,569,273.83 ' \
                        '\n资产 ' \
                        '\n总资产  2,659,796,439.52  2,132,691,687.02  24.72  1,794,532,319.05 ' \
                        '\n ' \
                        '\n'
        self.interpreter.lexer.input(input)
        for tok in self.interpreter.lexer:
            print(tok)
        #test yaac
        testParser._load_data([input])
        #self.interpreter.doWork(testParser,lexer=self.interpreter.lexer,debug=True,tracking=True)
        self.interpreter.parser.parse(input,lexer=self.interpreter.lexer,debug=True,tracking=True)

if __name__ == '__main__':
    unittest.main()
