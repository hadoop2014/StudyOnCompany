#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据
from ply import lex,yacc
import time
from functools import reduce
from interpreterAccounting.interpreterBaseClass import *


class InterpreterAccounting(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterAccounting, self).__init__(gConfig)
        self.docParser = memberModuleDict['docParser'.lower()]
        self.excelParser = memberModuleDict['excelParser'.lower()]
        self.sqlParser = memberModuleDict['sqlParser'.lower()]
        self.currentPageNumber = 0
        self.interpretDefine()


    def interpretDefine(self):
        tokens = self.tokens
        literals = self.literals
        # Tokens
        #采用动态变量名
        local_name = locals()
        for token in self.tokens:
            local_name['t_'+token] = self.dictTokens[token]
        self.logger.info('\n'+str({key:value for key,value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        def t_TIME(t):
            "\\d+\\s*年\\s*\\d+\\s*月\\s*\\d+\\s*日|\\d+\\s*年\\s*\\d+—\\d+\\s*月|\\d+\\s*(年|月|日)*-\\d+\\s*(年|月|日)|\\d+\\s*年|[○一二三四五六七八九〇]{4}\\s*年"
            return t

        def t_NUMERIC(t):
            r'\d+[,\d]*(\.\d{4})|\d+[,\d]*(\.\d{1,2})?'
            #从NUMRIC细分出NUMERO
            typeList = ['t_NUMERO']
            t.type = self._get_token_type(local_name, t.value, typeList, defaultType='NUMERIC')
            return t


        def t_NAME(t):
            r'[a-zA-Z_][a-zA-Z0-9_]*'
            # 从NAME细分出TAIL
            typeList = ['t_TAIL']
            t.type = self._get_token_type(local_name, t.value, typeList, defaultType='NAME')
            return t

        #t_ignore = " \t\n"
        t_ignore = self.ignores
        #解决亿纬锂能2018年财报中,'无形资产情况'和'单位: 元'之间,插入了《.... 5 .....》中间带有数字,导致误判为搜索到页尾
        t_ignore_COMMENT = "《.*》"


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")


        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)


        # Build the lexer
        self.lexer = lex.lex(outputdir=self.working_directory,reflags=int(re.MULTILINE))


        # Parsing rules
        precedence = (
            ('right', 'UMINUS'),
        )


        # dictionary of names_global
        self.names = {}


        def p_statement_grouphalf(p):
            '''statement : statement ')'
                         | statement '）'
                         | statement error '''
            p[0] = p[1]


        def p_statement_statement(p):
            '''statement : statement expression
                         | expression '''
            p[0] = p[1]


        def p_expression_reduce(p):
            '''expression : fetchtable expression
                          | fetchdata expression
                          | fetchtitle expression
                          | title expression
                          | illegalword
                          | parenthese
                          | skipword
                          | tail '''
            p[0] = p[1]


        def p_fetchtable_search(p):
            '''fetchtable : TABLE optional time optional unit finis '''
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[3],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            #unit = p[5].split(':')[-1].split('：')[-1]
            #self.names['货币单位'] = self._unit_transfer(unit)
            unit = self.names['unit']
            currency = self.names['currency']
            company = self.names['company']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency,company)

        '''
        def p_fetchtable_searchlong(p):
            "'fetchtable : TABLE optional time optional CURRENCY UNIT finis ''
            #解决海螺水泥2018年年报无法识别合并资产负债表,合并利润表等情况
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[3],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            unit = p[6].split(':')[-1].split('：')[-1]
            self.names['货币单位'] = self._unit_transfer(unit)
            currency = p[5]
            #currency = self.names_global['currency']
            company = self.names['company']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency,company)
        '''

        def p_fetchtable_searchnotime(p):
            '''fetchtable : TABLE optional unit finis '''
            #              | TABLE optional UNIT CURRENCY finis'''
            #第二个语法针对的是主要会计数据
            #TABLE optional UNIT CURRENCY NUMERO HEADER解决万东医疗2019年年报普通股现金分红情况表搜索不到问题,被TABLE optional UNIT finis 取代
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[3],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            #unit = p[3].split(':')[-1].split('：')[-1]
            #self.names['货币单位'] = self._unit_transfer(unit)
            unit = self.names['unit']
            currency = self.names['currency']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)

        '''
        def p_fetchtable_searchbracket(p):
            ''fetchtable : TABLE optional '（' unit '）' finis ''
            #第二个语法针对的是主要会计数据
            #解决海螺水泥2018年财报分季度主要财务表的识别问题
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[4],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            #unit = p[4].split(':')[-1].split('：')[-1]
            #self.names['货币单位'] = self._unit_transfer(unit)
            unit = self.names['unit']
            currency = self.names['currency']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)
        '''
        '''
        def p_fetchtable_searchcurrencyunit(p):
            'fetchtable : TABLE optional CURRENCY UNIT finis ''
            #第二个语法针对的是主要会计数据
            #解决海螺水泥2018年财报现金流量表补充资料 的识别问题
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[4],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            unit = p[4].split(':')[-1].split('：')[-1]
            self.names['货币单位'] = self._unit_transfer(unit)
            currency = self.names['currency']
            #interpretPrefix = '\n'.join([slice for slice in p if slice is not None and slice != 'optional']) + '\n'
            interpretPrefix = '\n'.join([slice.value for slice in p.slice if slice.value is not None and slice.type != 'optional']) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)
        '''

        def p_fetchtable_timedouble(p):
            '''fetchtable : TABLE optional time optional TIME '''
            #              | TABLE optional time DISCARD TIME'''
            #处理主要会计数据的的场景,存在第一次匹配到,又重新因为表头而第二次匹配到的场景
            #TABLE optional TIME DISCARD TIME解决康龙化成：2019年年度报告中主要会计数据的表头不规范, 两个TIME直接插入一个DISCARD
            tableName = self._get_tablename_alias(str.strip(p[1]))
            if len(self.names[tableName]['page_numbers']) != 0:
                if self.currentPageNumber == self.names[tableName]['page_numbers'][-1]:
                    self.logger.info("fetchtable warning(search again)%s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
                    return
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
            unit = NULLSTR
            currency = self.names['currency']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)

        def p_fetchtable_header(p):
            '''fetchtable : TABLE optional HEADER HEADER'''
            #专门用于解决杰瑞股份2016,2017,2018,2019年年度报告中现金流量表补充资料,无形资产情况搜索不到的情况.
            tableName = self._get_tablename_alias(str.strip(p[1]))
            if len(self.names[tableName]['page_numbers']) != 0:
                if self.currentPageNumber == self.names[tableName]['page_numbers'][-1]:
                    self.logger.info("fetchtable warning(search again)%s -> %s %s page %d" % (
                    p[1], tableName, p[3], self.currentPageNumber))
                    return
            if self._is_reatch_max_pages(self.names[tableName], tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
            unit = NULLSTR
            currency = self.names['currency']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName, tableBegin, interpretPrefix, unit, currency)


        def p_fetchtable_reatchtail(p):
            '''fetchtable : TABLE optional unit tail
                          | TABLE optional time optional unit tail
                          | TABLE optional tail
                          | TABLE optional time tail '''
            #              | TABLE optional unit tail'''
            #              | TABLE optional COMPANY tail'''
            #处理在页尾搜索到fetch的情况,NUMERIC为页尾标号,设置tableBegin = False,则_merge_table中会直接返回,直接搜索下一页
            #TABLE optional COMPANY NUMERIC解决大立科技2018年年报合并资产负债表出现在页尾的情况.
            #TABLE optional UNIT CURRENCY NUMERIC解决郑煤机2019年财报无形资产情况出现在页尾
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable warning(reach tail) %s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            unit = NULLSTR
            currency = self.names['currency']
            interpretPrefix = '\n'.join([str(slice) for slice in p[:-1] if slice is not None]) + '\n'
            tableBegin = False
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)

        '''
        def p_fetchtable_reatchtail_wrong(p):
            ''fetchtablewrong : TABLE optional NUMERIC '-'
                          | TABLE optional NUMERIC DISCARD''
            #TABLE optional NUMERIC '-' 在原语法末尾增加'-',原因是解决杰瑞股份2018年年报中第60页出现合并现金流量表无影响。....2018-067号公告,导致原语法TABLE optional NUMERIC误判
            #TABLE optional NUMERIC DISCARD解决青松股份2016年年报第10页出现无形资产情况表的误判
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.warning("fetchtable warning(reach tail but is wrong) %s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
        '''

        def p_fetchtable_skipword(p):
            '''fetchtablewrong : TABLE optional TABLE
                          | TABLE optional PUNCTUATION
                          | TABLE '(' NUMERO ')'
                          | TABLE optional '（' NUMERO '）'
                          | TABLE '(' discard ')'
                          | TABLE optional '（' discard '）'
                          | TABLE optional NUMERO '-'
                          | TABLE optional NUMERO DISCARD'''
            #TABLE optional NUMERIC '-' 在原语法末尾增加'-',原因是解决杰瑞股份2018年年报中第60页出现合并现金流量表无影响。....2018-067号公告,导致原语法TABLE optional NUMERIC误判
            #TABLE optional NUMERIC DISCARD解决青松股份2016年年报第10页出现无形资产情况表的误判
            #去掉 TABLE '(' DISCARD ')'
            #去掉TABLE optional TIME discard NUMERIC,该语句是干扰项
            #去掉了语法TABLE term,该语法和TABLE optional NUMERIC冲突
            #去掉合并资产负债表项目
            #去掉TABLE PUNCTUATION减少语法冲突
            #去掉TABLE parenthese,识别海螺水泥2018年年报分季度主要财务时,发生冲突
            interpretPrefix = '\n'.join([str(slice) for slice in p if slice is not None]) + '\n'
            self.logger.error("fetchtable in wrong mode,prefix: %s page %d"%(interpretPrefix.replace('\n','\t'),self.currentPageNumber))
            p[0] = p[1] + p[2]


        def p_optional_optional(p):
            '''optional : optional discard
                        | optional COMPANY
                        | optional LOCATION
                        | optional HEADER
                        | numero
                        | discard
                        | '(' NAME ')'
                        | empty '''
            # | optional fetchtitle
            # | optional title
            # optional CURRENCY 解决海螺水泥2018年年报无法识别合并资产负债表,合并利润表,现金流量表补充资料等情况
            #第2条规则optional fetchtitle DISCARD解决大立科技：2018年年度报告,合并资产负债表出现在表尾,而第二页开头为"浙江大立科技股份有限公司 2018 年年度报告全文"的场景
            #第3条规则'(' NAME ')' optional解决海螺水泥2018年年度报告,现金流量补充资料,紧接一行(a) 将净利润调节为经营活动现金流量 金额单位：人民币元.
            #fetchtitle DISCARD DISCARD DISCARD DISCARD解决上峰水泥2019年年报主要会计数据在末尾的问题
            #DISCARD PUNCTUATION DISCARD DISCARD和t_ignore_COMMENT = "《.*》"一起解决亿纬锂能2018年财报中搜索无形资产情况时误判为到达页尾
            #DISCARD DISCARD解决贝达药业2016年财报主要会计数据无法搜索到的问题
            #DISCARD LOCATION COMPANY解决海天味业2019年财报中出现合并资产负债表 2019 年 12 月 31 日  编制单位: 佛山市海天调味食品股份有限公司 单位:元 币种:人民币
            #optional TIME REPORT解决隆基股份2017年财报,合并资产利润表达到页尾,而下一页开头出现"2017年年度报告"
            #DISCARD DISCARD DISCARD解决理邦仪器：2019年年度报告的主要会计数据识别不到的问题
            #20200923,去掉DISCARD optional,optional fetchtitle DISCARD,fetchtitle DISCARD DISCARD DISCARD DISCARD,fetchtitle DISCARD,'(' NAME ')' optional,DISCARD DISCARD DISCARD
            for slice in p.slice:
                if slice.type == 'COMPANY':
                    self.names['company'] = self._eliminate_duplicates(slice.value)
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix

        '''
        def p_optional(p):
            ''optional : numero
                        | discard
                        | '(' NAME ')'
                        | empty''
            # 第3条规则'(' NAME ')' 解决海螺水泥2018年年度报告,现金流量补充资料,紧接一行(a) 将净利润调节为经营活动现金流量 金额单位：人民币元.
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix
            print('optional',p[0])
        '''

        def p_fetchdata_referencedouble(p):
            '''fetchdata : REFERENCE DISCARD REFERENCE NUMERIC
                         | REFERENCE NUMERIC REFERENCE DISCARD '''
            if self.names[self._get_reference_alias(p[1])] == NULLSTR :
                self.names.update({self._get_reference_alias(p[1]):p[2]})
            if self.names[self._get_reference_alias(p[3])] == NULLSTR:
                self.names.update({self._get_reference_alias(p[3]):p[4]})
            self.logger.info('fetchdata reference %s -> %s %s,%s -> %s %s'%(p[1],self._get_reference_alias(p[1]),p[2]
                             ,p[3],self._get_reference_alias(p[3]),p[4]))


        def p_fetchdata_referencetriple(p):
            '''fetchdata : REFERENCE NUMERIC term REFERENCE DISCARD DISCARD'''
            #解决华新水泥2018年报中,股票简称不能识别的问题
            if self.names[self._get_reference_alias(p[1])] == NULLSTR :
                self.names.update({self._get_reference_alias(p[1]):p[2]})
            if self.names[self._get_reference_alias(p[4])] == NULLSTR:
                self.names.update({self._get_reference_alias(p[4]):p[5]})
            self.logger.info('fetchdata reference %s -> %s %s,%s -> %s %s'%(p[1],self._get_reference_alias(p[1]),p[2]
                             ,p[3],self._get_reference_alias(p[3]),p[4]))


        def p_fetchdata_critical(p):
            '''fetchdata : CRITICAL NUMERIC fetchdata
                         | CRITICAL NUMERIC
                         | CRITICAL '-'
                         | CRITICAL empty
                         | CRITICAL LOCATION
                         | CRITICAL COMPANY'''
            critical = self._get_critical_alias(p[1])
            if self.names[critical] == NULLSTR :
                self.names.update({critical:p[2]})
                if critical == '公司地址':
                    if p[2] != NULLSTR:
                        self.names.update({critical:self._eliminate_duplicates(p[2])})
                    else:
                        self.names.update({critical:self.names['注册地址']})
                elif critical == '公司名称' and p[2] != NULLSTR :
                    self.names.update({critical:self._eliminate_duplicates(p[2])})
                elif critical == '注册地址' and p[2] != NULLSTR:
                    self.names.update({critical: self._eliminate_duplicates(p[2])})

            self.logger.info('fetchdata critical %s->%s %s page %d' % (p[1],critical,p[2],self.currentPageNumber))


        def p_fetchdata_skipword(p):
            '''fetchdatawrong : CRITICAL CRITICAL
                         | REFERENCE NUMERIC NAME
                         | REFERENCE NUMERIC TIME
                         | REFERENCE REFERENCE
                         | REFERENCE DISCARD'''
            p[0] = p[1]


        def p_illegalword(p):
            '''illegalword : TIME
                           | LOCATION
                           | REPORT
                           | NUMERO
                           | NUMERO NUMERO
                           | NUMERO '）'
                           | fetchtablewrong
                           | fetchdatawrong
                           | fetchtitlewrong '''
            #TABLE discard parenthese  该语句和TABLE optional ( UNIT ) finis语句冲突
            #所有语法开头的关键字,其非法的语法都可以放到该语句下,可答复减少reduce/shift冲突
            #TIME 是title语句的其实关键字,其他的如TABLE是fetchtable的关键字 ....
            #TABLE parenthese 解决现金流量表补充资料出现如下场景: 现金流量补充资料   (1)现金流量补充资料   单位： 元
            p[0] = p[1]


        def p_fetchtitle_company(p):
            '''fetchtitle : COMPANY TIME REPORT'''
            if self.names['公司名称'] == NULLSTR :
                self.names.update({'公司名称':self._eliminate_duplicates(p[1])})
            if self.names['报告时间'] == NULLSTR :
                years = self._time_transfer(p[2])
                self.names.update({'报告时间':years})
            if self.names['报告类型'] == NULLSTR:
                self.names.update({'报告类型':p[3]})
            self.logger.info('fetchtitle %s %s%s page %d'
                             % (self.names['公司名称'],self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            p[0] = p[1] + p[2] + p[3]

        '''
        def p_fetchtitle_company_reverse(p):
            ''fetchtitle : TIME REPORT COMPANY''
            if self.names['公司名称'] == NULLSTR :
                self.names.update({'公司名称':p[3]})
            if self.names['报告时间'] == NULLSTR :
                years = self._time_transfer(p[1])
                self.names.update({'报告时间':years})
            if self.names['报告类型'] == NULLSTR:
                self.names.update({'报告类型':p[2]})
            self.logger.info('fetchtitle %s %s%s page %d'
                             % (self.names['公司名称'],self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            p[0] = p[1] + p[2] + p[3]
        '''

        def p_fetchtitle_long(p):
            '''fetchtitle : COMPANY NAME parenthese TIME REPORT '''
            #解决海螺水泥2018年报第1页title的识别问题
            if self.names['公司名称'] == NULLSTR :
                self.names.update({'公司名称': self._eliminate_duplicates(p[1])})
            if self.names['报告时间'] == NULLSTR :
                years = self._time_transfer(p[4])
                self.names.update({'报告时间':years})
            if self.names['报告类型'] == NULLSTR:
                self.names.update({'报告类型':self._get_unit_alias(p[5])})
            self.logger.info('fetchtitle long %s %s%s page %d'
                             % (self.names['公司名称'],self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            p[0] = p[1] + p[4] + p[5]


        def p_fetchtitle_skipword(p):
            '''fetchtitlewrong : COMPANY error
                         | COMPANY TIME DISCARD
                         | COMPANY TIME NUMERIC
                         | COMPANY DISCARD
                         | COMPANY PUNCTUATION
                         | COMPANY NAME DISCARD
                         | COMPANY '''
            #去掉COMPANY UNIT,原因是正泰电器2018年财报中出现fetchtable : TABLE optional TIME DISCARD COMPANY UNIT error,出现了语法冲突
            #去掉COMPANY NUMERIC,原因是大立科技2018年年报中合并资产负债表出现在页尾会出现判断失误.
            #TIME REPORT 解决千和味业2019年财报中出现  "2019年年度报告",修改为在useless中增加REPORT
            #去掉fetchdata : COMPANY error,和fetchtitle : COMPANY error冲突
            p[0] = p[1]


        def p_title(p):
            '''title : TIME REPORT'''
            if self.names['报告时间'] == NULLSTR \
                and self.names['报告类型'] == NULLSTR:
                years = self._time_transfer(p[1])
                self.names.update({'报告时间':years})
                self.names.update({'报告类型':p[2]})
            self.logger.info('title %s%s page %d'% (self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            p[0] = p[1] + p[2]


        def p_parenthese_group(p):
            '''parenthese : '(' content ')'
                        | '(' content '）'
                        | '（' content '）'
                        | '（' content ')'
                        | '（' '）' '''
            #专门用于处理括号里的内容
            p[0] = p[2]


        def p_parenthese(p):
            '''content : content '（' content '）'
                       | content '(' content ')'
                       | content '（' '）'
                       | content discard
                       | content REFERENCE NUMERIC
                       | content REFERENCE NAME
                       | content term
                       | content TIME
                       | content PUNCTUATION
                       | content NAME
                       | content UNIT
                       | content '-'
                       | content CURRENCY
                       | content HEADER
                       | content LOCATION
                       | content WEBSITE
                       | content NUMERO
                       | content '%'
                       | content COMPANY
                       | content REFERENCE
                       | content CRITICAL
                       | TIME
                       | NAME
                       | PUNCTUATION
                       | WEBSITE
                       | UNIT
                       | discard
                       | term
                       | NUMERO
                       | LOCATION
                       | CURRENCY
                       | '%'
                       | '％'
                       | '-'
                       | COMPANY '''
            p[0] = p[1]


        #def p_skipword_group(p):
        #    '''skipword : '(' skipword ')'
        #                | '(' skipword '）'
        #                | '（' skipword '）'
        #                | '（' skipword ')' '''
        #    p[0] = p[2]


        def p_skipword(p):
            '''skipword : useless skipword
                        | term skipword
                        | useless
                        | '-' useless
                        | term
                        | '-' term %prec UMINUS'''
            p[0] = p[1]



        #def p_useless_reduce(p):
        #    '''useless : '(' useless ')'
        #               | '(' useless '）'
        #               | '（' useless '）'
        #               | '（' useless ')'
        #               | '-' useless '''
        #    p[0] = p[1]


        def p_useless(p):
            '''useless : PUNCTUATION
                       | discard
                       | WEBSITE
                       | EMAIL
                       | NAME
                       | HEADER
                       | UNIT
                       | CURRENCY
                       | '-'
                       | '%'
                       | '％' '''
            p[0] = p[1]
            #                       | LOCATION


        #def p_term_group(p):
        #    '''term : '(' term ')'
        #            |  '（' term '）'
        #            | '-' term %prec UMINUS '''
        #    p[0] = -p[2]  #财务报表中()表示负值


        def p_term_percentage(p):
            '''term : NUMERIC '%'
                    | NUMERIC '％' '''
            p[0] = round(float(p[1].replace(',','')) * 0.01,4)


        def p_term_numeric(p):
            '''term : NUMERIC '''
            if p[1].find('.') < 0 :
                p[0] = int(p[1].replace(',',''))
            else:
                p[0] = float(p[1].replace(',',''))


        def p_discard(p):
            '''discard : DISCARD discard
                       | DISCARD '''
            p[0] = p[1]

        '''
        def p_company(p):
            ''company : COMPANY
                       | LOCATION COMPANY
                       | discard COMPANY
                       ''
            for slice in p.slice:
                if slice.type == 'COMPANY':
                    self.names['company'] = self._eliminate_duplicates(p[1])
            p[0] = p[1]
        '''


        def p_unit(p):
            '''unit : UNIT
                    | UNIT CURRENCY
                    | CURRENCY UNIT
                    | '（' UNIT '）'
                    | discard unit'''
            for slice in p.slice:
                if slice.type == 'UNIT':
                    unit = slice.value.split(':')[-1].split('：')[-1]
                    self.names['unit'] = unit
                    self.names['货币单位'] = self._unit_transfer(unit)
                elif slice.type == 'CURRENCY':
                    self.names['currency'] = slice.value
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix


        def p_time(p):
            '''time : TIME
                    | TIME REPORT'''
            p[0] = p[1]


        def p_finis(p):
            '''finis : NUMERO
                     | NUMERO HEADER
                     | empty '''
            #if p.slice[1].type == 'CURRENCY':
            #    p[1] = p[1].split(':')[-1].split('：')[-1]
            #    self.names['currency'] = p[1]
            p[0] = '\n'.join([str(slice) for slice in p if slice is not None])
            print('finis',p[0])


        def p_tail(p):
            '''tail : NUMERO TAIL
                    | NUMERO NUMERO TAIL'''
            tail = ' '.join([str(slice) for slice in p if slice is not None])
            self.logger.info('fetchtail %s page %d'%(tail,self.currentPageNumber))
            p[0] = p[1]


        def p_numero(p):
            '''numero : NUMERO
                      | discard NUMERO '''
            #用于解决三诺生物2018年年报中,多个表TABLE后插入数字
            p[0] = p[1]


        def p_empty(p):
            '''empty : '''
            p[0] = NULLSTR


        def p_error(p):
            if p:
                print("Syntax error at '%s:%s' page %d" % (p.value,p.type,self.currentPageNumber))
            else:
                print("Syntax error at EOF page %d"%self.currentPageNumber)


        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def doWork(self,lexer=None,debug=False,tracking=False):
        start_time = time.time()
        fileName = os.path.split(self.docParser.sourceFile)[-1]
        if self.docParser.is_file_in_checkpoint(fileName):
            self.logger.info('the file %s is already in checkpointfile,no need to process!'%fileName)
            return
        self.logger.info("\n\n%s parse is starting!\n\n" % (fileName))
        self._get_time_type_by_name(self.gConfig['sourcefile'])
        for data in self.docParser:
            self.currentPageNumber = self.docParser.index
            text = self.docParser._get_text(data)
            self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)
        self._process_critical_table()
        sourceFile = os.path.split(self.docParser.sourceFile)[-1]
        self.logger.info('%s\tcritical:'%sourceFile + ','.join([self.names['公司名称'],self.names['报告时间'],self.names['报告类型']
                         ,str(self.names['公司代码']),self.names['公司简称'],self.names['公司地址']
                         ,str(self.names['货币单位']),self.names["注册地址"]]))
        self.logger.info('%s\tprocess_info:'%sourceFile + str(self.sqlParser.process_info))
        failedTable = set(self.tableNames).difference(set(self.sqlParser.process_info.keys()))
        if len(failedTable) == 0:
            self.logger.info("%s\tall table is success fetched!\n"%(sourceFile))
            self.docParser.save_checkpoint(fileName)
        else:
            self.logger.info('%s\ttable(%s) is failed to fetch'
                             %(sourceFile,failedTable))
        self.docParser._close()
        self.logger.info('\n\n parse %s file end, time used %.4f' % (fileName,(time.time() - start_time)))
        return self.sqlParser.process_info


    def _process_fetch_table(self, tableName, tableBegin, interpretPrefix, unit=NULLSTR, currency=NULLSTR, company=NULLSTR):
        assert tableName is not None and tableName != NULLSTR, 'tableName must not be None'
        self.names[tableName].update({'tableName': tableName, 'unit': unit, 'currency': currency
                                     ,'company':company
                                     ,'tableBegin': tableBegin
                                     ,'page_numbers': self.names[tableName]['page_numbers'] + list([self.currentPageNumber])})
        if self.names[tableName]['tableEnd'] == False:
            if self.names["公司地址"] == NULLSTR:
                self.names["公司地址"] = self.names["注册地址"]
            self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                         ,'公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                         ,'报告类型': self.names['报告类型']
                                         ,'公司地址': self.names['公司地址']
                                         ,'行业分类': self.names['行业分类']
                                         ,'货币单位': self._unit_transfer(unit)})
            self.docParser._merge_table(self.names[tableName], interpretPrefix)
            if self.names[tableName]['tableEnd'] == True:
                self.excelParser.initialize(dict({'sourcefile': self.gConfig['sourcefile']}))
                self.excelParser.writeToStore(self.names[tableName])
                self.sqlParser.writeToStore(self.names[tableName])
        self.logger.info('\nprefix: %s:'%interpretPrefix.replace('\n','\t') + str(self.names[tableName]))


    def _process_critical_table(self,tableName = '关键数据表'):
        assert tableName is not None and tableName != NULLSTR,"tableName must not be None"
        table = self._construct_table(tableName)
        if self.names["公司地址"] == NULLSTR:
            self.names["公司地址"] = self.names["注册地址"]
        self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                     ,'公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                     ,'报告类型': self.names['报告类型']
                                     ,'公司地址': self.names['公司地址']
                                     ,'行业分类': self.names['行业分类']
                                     ,'货币单位': self.names['货币单位']})
        self.names[tableName].update({"table":table})
        self.names[tableName].update({"tableName":tableName})
        self.excelParser.initialize(dict({'sourcefile': self.gConfig['sourcefile']}))
        self.excelParser.writeToStore(self.names[tableName])
        self.sqlParser.writeToStore(self.names[tableName])


    def _eliminate_duplicates(self,source):
        target = NULLSTR
        if source == NULLSTR:
            return target
        target = reduce(self._deduplicate, list(source))
        target = ''.join(target)
        return  target


    def _get_time_type_by_name(self,filename):
        time = self._standardize('\\d+年',filename)
        type = self._standardize('|'.join(self.gJsonBase['报告类型']),filename)
        company = self._standardize(self.gJsonInterpreter['DISCARD'],filename)
        code = self._standardize('（\\d+）',filename)
        if self.names['报告时间'] == NULLSTR and time is not NaN:
            self.names["报告时间"] = time
        if self.names['报告类型'] == NULLSTR and type is not NaN:
            self.names['报告类型'] = type
        if self.names['公司简称'] ==NULLSTR and company is not NaN:
            self.names['公司简称'] = company
        if self.names['公司代码'] ==NULLSTR and code is not NaN:
            code = code.replace('（',NULLSTR).replace('）',NULLSTR)
            self.names['公司代码'] = code
        self.logger.info('fetch data from filename:%s %s %s %s'
                         %(self.names["公司代码"],self.names["公司简称"],self.names["报告时间"],self.names["报告类型"]))


    def _construct_table(self,tableNmae):
        headers = self.dictTables[tableNmae]['header']
        fields = self.dictTables[tableNmae]['fieldName']
        assert isinstance(headers,list) and isinstance(fields,list)\
            ,"headers (%s) and fields(%s) must be list"%(str(headers),str(fields))
        rows = [list([key,value]) for key,value in self.names.items() if key in fields]
        table = [headers] + rows
        for row in rows:
            if row[-1] == NULLSTR:
                self.logger.warning('critical %s failed to fetch'%row[0])
        return table


    def _is_reatch_max_pages(self, fetchTable,tableName):
        maxPages = self.dictTables[tableName]['maxPages']
        if len(fetchTable['page_numbers']) > maxPages:
            isReatchMaxPages = True
            self.logger.error("table %s is reatch max page numbers:%d > %d"
                             %(tableName,len(fetchTable['page_numbers']),maxPages))
        elif fetchTable['tableEnd'] == True:
            isReatchMaxPages = True
        else:
            isReatchMaxPages = False
        return isReatchMaxPages


    def _time_transfer(self,time):
        transfer = dict({
            '○':'0',
            '一':'1',
            '二':'2',
            '三':'3',
            '四':'4',
            '五':'5',
            '六':'6',
            '七':'7',
            '八':'8',
            '九':'9',
            '〇':'0',
            '年':'年'
        })
        timelist = [transfer[number] for number in list(time) if number in transfer.keys()]
        if len(timelist) > 1 :
            time = ''.join(timelist)
        time = time.replace(' ',NULLSTR)
        return time


    def _unit_transfer(self,unit):
        transfer = dict({
            '元': 1,
            '千元': 1000,
            '万元': 10000,
            '百万元': 1000000,
            '千万元': 10000000
        })
        unitStandardize = self._standardize("(元|千元|万元|百万元|千万元)",unit)
        if unitStandardize in transfer.keys():
            unitStandardize = transfer[unitStandardize]
        else:
            unitStandardize = 1
            self.logger.warning('%s is not the unit of currency'%unit)
        return unitStandardize


    def initialize(self,dictParameter=None):
        for tableName in self.tableNames:
            self.names.update({tableName:{'tableName':NULLSTR,'time':NULLSTR,'unit':NULLSTR,'currency':NULLSTR
                                          ,'company':NULLSTR,'公司名称':NULLSTR,'公司代码':NULLSTR,'公司简称':NULLSTR
                                          ,'报告时间':NULLSTR,'报告类型':NULLSTR,"公司地址":NULLSTR,'行业分类':NULLSTR
                                          ,'货币单位': 1 #货币单位默认为1
                                          ,"注册地址": NULLSTR
                                          ,'table':NULLSTR,'tableBegin':False,'tableEnd':False
                                          ,"page_numbers":list()}})
        self.names.update({'unit':NULLSTR,'currency':NULLSTR,'company':NULLSTR,'time':NULLSTR})
        for commonField,_ in self.commonFileds.items():
            self.names.update({commonField:NULLSTR})
        for cirtical in self.criticals:
            self.names.update({self._get_critical_alias(cirtical):NULLSTR})
        if dictParameter is not None:
            self.docParser._load_data(dictParameter['sourcefile'])
            self.gConfig.update(dictParameter)
        else:
            self.docParser._load_data()


def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterAccounting(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter

