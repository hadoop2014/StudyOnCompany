#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据
from ply import lex,yacc
import time
import pandas as pd
from xlrd.biffh import XLRDError
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
            "\\d+\\s*年\\s*\\d+\\s*月\\s*\\d+\\s*日|\\d+\\s*年\\s*\\d+\\s*[—|-]\\s*\\d+\\s*月|\\d+\\s*(年|月|日)*\\s*[—|-]\\s*\\d+\\s*(年|月|日)|\\d+\\s*年\\s*\\d+\\s*月|\\d+\\s*年|[○一二三四五六七八九〇]{4}\\s*年"
            #该函数会覆盖interpreterAccounting.json中定义的TIME的正则表达式,所以必须使得函数中的正则表达式和.json文件中配置的相同
            return t


        def t_NUMERIC(t):
            r'\d+[,\d]*(\.\d{4})|\d+[,\d]*(\.\d{1,2})?'
            #从NUMRIC细分出NUMERO
            typeList = ['t_NUMERO']
            t.type = self._get_token_type(local_name, t.value, typeList, defaultType='NUMERIC')
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
                          | illegalword
                          | parenthese
                          | skipword
                          '''
            #去掉| tail  | parenthese TAIL
            p[0] = p[1]


        def p_fetchtable(p):
            '''fetchtable : TABLE optional time optional unit finis
                          | TABLE optional unit finis
                          | TABLE optional time optional TIME
                          | TABLE optional HEADER HEADER'''
            # TABLE optional time optional CURRENCY UNIT finis解决海螺水泥2018年年报无法识别合并资产负债表,合并利润表等情况
            # TABLE optional UNIT CURRENCY finis第二个语法针对的是主要会计数据
            # TABLE optional UNIT CURRENCY NUMERO HEADER解决万东医疗2019年年报普通股现金分红情况表搜索不到问题,被TABLE optional UNIT finis 取代
            # fetchtable : TABLE optional '（' unit '）' finis 解决海螺水泥2018年财报分季度主要财务表的识别问题
            # TABLE optional CURRENCY UNIT finis 解决海螺水泥2018年财报现金流量表补充资料 的识别问题
            # TABLE optional TIME DISCARD TIME解决康龙化成：2019年年度报告中主要会计数据的表头不规范, 两个TIME直接插入一个DISCARD
            # TABLE optional HEADER HEADER 专门用于解决杰瑞股份2016,2017,2018,2019年年度报告中现金流量表补充资料,无形资产情况搜索不到的情况.
            tableName = self._get_tablename_alias(str.strip(p[1]))
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            #interpretPrefix必须用\n做连接,lexer需要用到\n
            self.logger.info("fetchtable %s -> %s :%s page %d!" % (p[1],tableName,interpretPrefix.replace('\n',' '),self.currentPageNumber))
            if len(self.names[tableName]['page_numbers']) != 0:
                # 处理主要会计数据的的场景,存在第一次匹配到,又重新因为表头而第二次匹配到的场景,实际通过语法规则,已经规避了该问题
                if self.currentPageNumber == self.names[tableName]['page_numbers'][-1]:
                    self.logger.info("fetchtable warning(search again)%s -> %s %s page %d" % (p[1], tableName, p[3], self.currentPageNumber))
                    return
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            #unit = self.names['unit']
            #currency = self.names['currency']
            #company = self.names['company']
            self.names['公司名称'] = self.names['company']
            self.names['货币单位'] = self._unit_transfer(self.names['unit'])
            self.names['货币名称'] = self.names['currency']
            #if self.names['公司地址'] == NULLSTR:
            #    self.names['公司地址'] = self.names['address']
            #tableBegin = True
            #self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency,company)
            self._process_fetch_table(tableName, tableBegin=True, interpretPrefix=interpretPrefix)
            self.logger.info('\nprefix: %s:' % interpretPrefix.replace('\n', '\t') + str(self.names[tableName]))


        def p_fetchtable_reatchtail(p):
            '''fetchtable : TABLE optional unit tail
                          | TABLE optional time optional unit tail
                          | TABLE optional tail
                          | TABLE optional time tail '''
            #处理在页尾搜索到fetch的情况,NUMERIC为页尾标号,设置tableBegin = False,则_merge_table中会直接返回,直接搜索下一页
            #TABLE optional COMPANY NUMERIC解决大立科技2018年年报合并资产负债表出现在页尾的情况.
            #TABLE optional UNIT CURRENCY NUMERIC解决郑煤机2019年财报无形资产情况出现在页尾
            #TABLE optional COMPANY tail已经被TABLE optional unit tail取代
            tableName = self._get_tablename_alias(str.strip(p[1]))
            interpretPrefix = '\n'.join([str(slice) for slice in p[:-1] if slice is not None]) + '\n'
            self.logger.info("fetchtable warning(reach tail) %s -> %s : %s page %d" % (p[1], tableName, interpretPrefix.replace('\n',''), self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            #unit = NULLSTR
            #currency = self.names['currency']
            #tableBegin = False
            #self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)
            self._process_fetch_table(tableName, tableBegin=False, interpretPrefix=interpretPrefix)#, unit, currency)
            self.logger.info('\nprefix: %s:' % interpretPrefix.replace('\n', '\t') + str(self.names[tableName]))


        def p_fetchtable_wrong(p):
            '''fetchtablewrong :  TABLE optional PUNCTUATION
                          | TABLE optional '(' NUMERO ')'
                          | TABLE optional '（' NUMERO '）'
                          | TABLE optional '（' discard '）'
                          | TABLE optional '(' discard ')'
                          | TABLE optional '(' LABEL ')'
                          | TABLE optional LABEL
                          | TABLE optional '-'
                          | TABLE optional time optional  NUMERIC'''
            #TABLE optional TABLE去掉,上海机场2018年年报出现 现金流量表补充资料 1、 现金流量表补充资料
            # TABLE optional '(' NAME ')' 和optional  '(' NAME ')'冲突
            # TABLE '(' discard ')' 可用
            #TABLE optional NUMERIC '-' 在原语法末尾增加'-',原因是解决杰瑞股份2018年年报中第60页出现合并现金流量表无影响。....2018-067号公告,导致原语法TABLE optional NUMERIC误判
            #TABLE optional NUMERIC DISCARD解决青松股份2016年年报第10页出现无形资产情况表的误判
            #去掉 TABLE '(' DISCARD ')'
            #去掉TABLE optional TIME discard NUMERIC,该语句是干扰项
            #去掉了语法TABLE term,该语法和TABLE optional NUMERIC冲突
            #去掉合并资产负债表项目
            #去掉TABLE PUNCTUATION减少语法冲突
            #去掉TABLE parenthese,识别海螺水泥2018年年报分季度主要财务时,发生冲突
            #TABLE optional time optional  NUMERIC可以过滤掉三诺生物2019年年报第60页出现了合并资产负责表,但不是所需要的,真正的表在第100页
            #TABLE optional NUMERO DISCARD 需要去掉,会导致三诺生物2018年年报 合并资产负债表搜索失败
            #TABLE optional '(' LABEL ')'解决海天味业2016年年报 出现" 近三年主要会计数据和财务指标(一) 主要会计数据",第一个TABLE '近三年主要会计数据和财务指标'是误判
            interpretPrefix = '\n'.join([str(slice) for slice in p if slice is not None]) + '\n'
            self.logger.warning("fetchtable in wrong mode,prefix: %s page %d"%(interpretPrefix.replace('\n','\t'),self.currentPageNumber))
            p[0] = p[1] + p[2]


        def p_optional(p):
            '''optional : optional discard
                        | optional COMPANY
                        | optional LOCATION
                        | optional HEADER
                        | optional NUMERO
                        | optional '(' NAME ')'
                        | optional '（' LABEL '）'
                        | discard
                        | NUMERIC
                        | empty '''
            # optional fetchtitle 被optioanl COMPANY time取代
            # optional CURRENCY 解决海螺水泥2018年年报无法识别合并资产负债表,合并利润表,现金流量表补充资料等情况
            #第2条规则optional fetchtitle DISCARD解决大立科技：2018年年度报告,合并资产负债表出现在表尾,而第二页开头为"浙江大立科技股份有限公司 2018 年年度报告全文"的场景
            #第3条规则optional '(' NAME ')' 解决海螺水泥2018年年度报告,现金流量补充资料,紧接一行(a) 将净利润调节为经营活动现金流量 金额单位：人民币元.
            #fetchtitle DISCARD DISCARD DISCARD DISCARD解决上峰水泥2019年年报主要会计数据在末尾的问题
            #DISCARD PUNCTUATION DISCARD DISCARD和t_ignore_COMMENT = "《.*》"一起解决亿纬锂能2018年财报中搜索无形资产情况时误判为到达页尾
            #DISCARD DISCARD解决贝达药业2016年财报主要会计数据无法搜索到的问题
            #DISCARD LOCATION COMPANY解决海天味业2019年财报中出现合并资产负债表 2019 年 12 月 31 日  编制单位: 佛山市海天调味食品股份有限公司 单位:元 币种:人民币
            #optional TIME REPORT解决隆基股份2017年财报,合并资产利润表达到页尾,而下一页开头出现"2017年年度报告"
            #DISCARD DISCARD DISCARD解决理邦仪器：2019年年度报告的主要会计数据识别不到的问题
            #20200923,去掉DISCARD optional,optional fetchtitle DISCARD,fetchtitle DISCARD DISCARD DISCARD DISCARD,fetchtitle DISCARD,'(' NAME ')' optional,DISCARD DISCARD DISCARD
            #optional NUMERO用于解决三诺生物2018年年报中,多个表TABLE后插入数字
            for slice in p.slice:
                if slice.type == 'COMPANY':
                    self.names['company'] = self._eliminate_duplicates(slice.value)
                if slice.type == 'LOCATION':
                    self.names['address'] = self._eliminate_duplicates(slice.value)
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix


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
            '''fetchdata : REFERENCE NUMERIC NUMERIC REFERENCE DISCARD DISCARD'''
            #解决华新水泥2018年报中,股票简称不能识别的问题
            if self.names[self._get_reference_alias(p[1])] == NULLSTR :
                self.names.update({self._get_reference_alias(p[1]):p[2]})
            if self.names[self._get_reference_alias(p[4])] == NULLSTR:
                self.names.update({self._get_reference_alias(p[4]):p[5]})
            self.logger.info('fetchdata reference %s -> %s %s,%s -> %s %s'%(p[1],self._get_reference_alias(p[1]),p[2]
                             ,p[3],self._get_reference_alias(p[3]),p[4]))


        def p_fetchdata_critical(p):
            '''fetchdata : CRITICAL term fetchdata
                         | CRITICAL term
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


        def p_fetchdata_wrong(p):
            '''fetchdatawrong : REFERENCE NUMERIC NAME
                         | REFERENCE NUMERIC TIME
                         | REFERENCE NUMERIC DISCARD
                         | REFERENCE NUMERIC LOCATION
                         | REFERENCE REFERENCE
                         | REFERENCE DISCARD
                         | REFERENCE LABEL'''
            p[0] = p[1]


        def p_fetchtitle(p):
            '''fetchtitle : TIME REPORT
                          | company TIME REPORT
                          | company selectable TIME REPORT'''
            #TIME REPORT COMPANY没有必要，用TIME REPORT生效即可，COMPANY可以通过CRITICAL COMPANY获取
            #COMPANY selectable TIME REPORT 解决海螺水泥2018年报第1页title的识别问题
            for slice in p.slice:
                if slice.type == 'company':
                    if self.names['公司名称'] == NULLSTR :
                        self.names.update({'公司名称':self.names['company']})
                    if self.names['公司地址'] == NULLSTR:
                        self.names.update({'公司地址': self.names['address']})
                if self.names['报告时间'] == NULLSTR and slice.type == 'TIME':
                    years = self._time_transfer(slice.value)
                    self.names.update({'报告时间':years})
                if self.names['报告类型'] == NULLSTR and slice.type == 'REPORT':
                    self.names.update({'报告类型':self._get_report_alias(slice.value)})
                #if self.names['公司地址'] == NULLSTR and slice.type == 'LOCATION':
                #    self.names.update({'公司地址': slice.value})
            #self.logger.info('fetchtitle %s %s %s%s page %d'
            #        % (self.names['公司地址'],self.names['公司名称'],self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            self.logger.info('fetchtitle %s page %d '%(prefix,self.currentPageNumber))
            p[0] = prefix


        def p_fetchtitle_wrong(p):
            '''fetchtitlewrong : fetchtitlewrong TAIL
                         | company error
                         | company TIME DISCARD
                         | company TIME NUMERIC
                         | company TIME NUMERO
                         | company TIME LABEL
                         | company selectable DISCARD
                         | company selectable TAIL
                         | company
                         | TIME '''
            # company DISCARD 去掉
            # company PUNCTUATION 去掉
            # company NAME DISCARD 去掉
            #去掉COMPANY UNIT,原因是正泰电器2018年财报中出现fetchtable : TABLE optional TIME DISCARD COMPANY UNIT error,出现了语法冲突
            #去掉COMPANY NUMERIC,原因是大立科技2018年年报中合并资产负债表出现在页尾会出现判断失误.
            #TIME REPORT 解决千和味业2019年财报中出现  "2019年年度报告",修改为在skipword中增加REPORT
            #去掉fetchdata : COMPANY error,和fetchtitle : COMPANY error冲突
            p[0] = p[1]


        def p_selectable(p):
            '''selectable : selectable parenthese
                          | selectable NAME
                          | NAME '''
            #解决海螺水泥2018年报第1页title的识别问题
            #解决康龙化成2019年年报第1页title的识别问题
            p[0] = p[1]


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
                       | content '(' ')'
                       | content discard
                       | content REFERENCE NUMERIC
                       | content REFERENCE NAME
                       | content NUMERIC
                       | content TIME
                       | content REPORT
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
                       | content TAIL
                       | content LABEL
                       | TIME
                       | NAME
                       | PUNCTUATION
                       | WEBSITE
                       | UNIT
                       | discard
                       | NUMERIC
                       | NUMERO
                       | LOCATION
                       | CURRENCY
                       | HEADER
                       | LABEL
                       | '%'
                       | '％'
                       | '-'
                       | COMPANY
                       | REFERENCE '''
            p[0] = p[1]


        def p_illegalword(p):
            '''illegalword : NUMERO
                           | NUMERO NUMERO
                           | NUMERO '）'
                           | TAIL
                           | fetchtablewrong
                           | fetchdatawrong
                           | fetchtitlewrong'''
            #TABLE discard parenthese  该语句和TABLE optional ( UNIT ) finis语句冲突
            #所有语法开头的关键字,其非法的语法都可以放到该语句下,可答复减少reduce/shift冲突
            #TIME 是title语句的其实关键字,其他的如TABLE是fetchtable的关键字 ....
            #TABLE parenthese 解决现金流量表补充资料出现如下场景: 现金流量补充资料   (1)现金流量补充资料   单位： 元
            p[0] = p[1]


        def p_skipword(p):
            '''skipword : skipword NUMERIC
                       | skipword '%'
                       | skipword TAIL
                       | discard
                       | LOCATION
                       | REPORT
                       | WEBSITE
                       | EMAIL
                       | NAME
                       | HEADER
                       | UNIT
                       | CURRENCY
                       | NUMERIC
                       | PUNCTUATION
                       | LABEL
                       | '-'
                       | '%'
                       | '％' '''
            p[0] = p[1]


        def p_discard(p):
            '''discard : DISCARD discard
                       | DISCARD '''
            p[0] = p[1]


        def p_company(p):
            '''company : COMPANY
                       | LOCATION COMPANY'''
            for slice in p.slice:
                if slice.type == 'COMPANY':
                    self.names['company'] = self._eliminate_duplicates(slice.value)
                if slice.type == 'LOCATION':
                    self.names['address'] = self._eliminate_duplicates(slice.value)
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix


        def p_unit(p):
            '''unit : UNIT
                    | UNIT CURRENCY
                    | CURRENCY UNIT
                    | '（' UNIT '）'
                    | '(' DISCARD CURRENCY UNIT ')' '''
            # 去掉discard unit,作为最小粒度语法单元,引入discard会带来语法冲突
            #'(' DISCARD CURRENCY UNIT ')'解决海天味业2016年年报中出现 (金额单位：人民币元)
            for slice in p.slice:
                if slice.type == 'UNIT':
                    unit = slice.value.split(':')[-1].split('：')[-1]
                    self.names['unit'] = unit
                    self.names['货币单位'] = self._unit_transfer(unit)
                elif slice.type == 'CURRENCY':
                    self.names['currency'] = slice.value.split(':')[-1].split('：')[-1]
            prefix = ' '.join([str(slice) for slice in p if slice is not None])
            p[0] = prefix


        def p_time(p):
            '''time : TIME
                    | TIME REPORT'''
            #仅用于fetchtable
            p[0] = p[1]

        def p_term(p):
            '''term : NUMERIC
                    | NUMERO'''
            p[0] = p[1]


        def p_finis(p):
            '''finis : NUMERO
                     | NUMERO HEADER
                     | HEADER
                     | empty '''
            p[0] = '\n'.join([str(slice) for slice in p if slice is not None])
            #self.logger.info('finis： %s'%p[0])


        def p_tail(p):
            '''tail : NUMERO TAIL
                    | NUMERO NUMERO TAIL'''
            tail = ' '.join([str(slice) for slice in p if slice is not None])
            #self.logger.info('fetchtail %s page %d'%(tail,self.currentPageNumber))
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
        self.excelParser.initialize(dict({'sourcefile': self.gConfig['sourcefile']}))
        #初始化process_info,否则计算出来的结果不正确
        self.sqlParser.process_info = {}
        for data in self.docParser:
            self.currentPageNumber = self.docParser.index
            text = self.docParser._get_text(data)
            self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)
        self._process_critical_table()
        sourceFile = os.path.split(self.docParser.sourceFile)[-1]
        self.logger.info('%s\tcritical:'%sourceFile + ','.join([self.names['公司名称'],self.names['报告时间'],self.names['报告类型']
                         ,str(self.names['公司代码']),self.names['公司简称'],self.names['公司地址']
                         ,str(self.names['货币单位']),self.names["注册地址"]]))
        failedTable = set(self.tableNames).difference(set(self.sqlParser.process_info.keys()))
        if len(failedTable) == 0:
            self.logger.info('success to process %s\tprocess_info:' % sourceFile + str(self.sqlParser.process_info))
            self.logger.info("all table is success fetched %s!\n"%(sourceFile))
            self.docParser.save_checkpoint(fileName)
            resultInfo = dict({'sourcefile': fileName, 'processtime':(time.time() - start_time),'failedTable': failedTable})
        else:
            self.logger.info('failed to fetch %s\t tables:%s!\n' % (sourceFile, failedTable))
            self.logger.info('now start to repair the fetch failed tables!\n')
            repairedTable = self._process_repair_table(failedTable)
            self.logger.info('success to process %s\t process_info:' % sourceFile + str(self.sqlParser.process_info))
            if len(repairedTable) > 0:
                self.logger.info('success to repair %s\t tables:%s'%(sourceFile ,repairedTable))
            failedTable = failedTable.difference(repairedTable)
            if len(failedTable) > 0:
                self.logger.info('failed to process %s\t tables:%s!' %(sourceFile,failedTable))
            else:
                self.logger.info("all table is success processed %s!\n" % (sourceFile))
            resultInfo = dict({'sourcefile': fileName, 'processtime':(time.time() - start_time)
                              ,'failedTable': list([(tableName,self.names[tableName]['page_numbers']) for tableName in failedTable])})
        self.docParser._close()
        self.logger.info('\n\n parse %s file end, time used %.4f' % (fileName,(time.time() - start_time)))
        return resultInfo


    def _process_fetch_table(self, tableName, tableBegin, interpretPrefix) :#, unit=NULLSTR, currency=NULLSTR, company=NULLSTR):
        assert tableName is not None and tableName != NULLSTR, 'tableName must not be None'
        #self.names[tableName].update({'tableName': tableName, 'unit': unit, 'currency': currency
        #                             ,'company':company
        #                             ,'tableBegin': tableBegin
        #                             ,'page_numbers': self.names[tableName]['page_numbers'] + list([self.currentPageNumber])})
        self.names[tableName].update({'tableName': tableName,'tableBegin': tableBegin
                                     ,'page_numbers': self.names[tableName]['page_numbers'] + list([self.currentPageNumber])})
        #self.names['货币单位'] = self._unit_transfer(unit)
        if self.names[tableName]['tableEnd'] == False:
            self.docParser._merge_table(self.names[tableName], interpretPrefix)
            if self.names[tableName]['tableEnd'] == True:
                self._parse_table(tableName)
                '''
                if self.names["公司地址"] == NULLSTR:
                    self.names["公司地址"] = self.names["注册地址"]
                self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                                 , '公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                                 , '报告类型': self.names['报告类型']
                                                 , '公司地址': self.names['公司地址']
                                                 , '行业分类': self.names['行业分类']
                                                 , '货币单位': self.names['货币单位']})
                self.excelParser.writeToStore(self.names[tableName])
                self.sqlParser.writeToStore(self.names[tableName])
                '''
        #self.logger.info('\nprefix: %s:'%interpretPrefix.replace('\n','\t') + str(self.names[tableName]))


    def _process_critical_table(self,tableName = '关键数据表'):
        assert tableName is not None and tableName != NULLSTR,"tableName must not be None"
        table = self._construct_table(tableName)
        self.names[tableName].update({'tableName': tableName,'table':table, 'tableBegin': True,"tableEnd":True})
        self._parse_table(tableName)
        '''
        if self.names["公司地址"] == NULLSTR:
            self.names["公司地址"] = self.names["注册地址"]
        self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                     ,'公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                     ,'报告类型': self.names['报告类型']
                                     ,'公司地址': self.names['公司地址']
                                     ,'行业分类': self.names['行业分类']
                                     ,'货币单位': self.names['货币单位']})
        #self.names[tableName].update({"table":table})
        #self.names[tableName].update({"tableName":tableName})
        #self.excelParser.initialize(dict({'sourcefile': self.gConfig['sourcefile']}))
        self.excelParser.writeToStore(self.names[tableName])
        self.sqlParser.writeToStore(self.names[tableName])
        '''


    def _process_repair_table(self,failedTable):
        assert isinstance(failedTable,set) and len(failedTable) > 0, "failedTable must be a set and not be NULL"
        #isRepaired = False
        company,reportTime,reportType = self.names['公司简称'],self.names['报告时间'],self.names['报告类型']
        #self.names[tableName].update({"table": table})
        repairedTable = set()
        for tableName in sorted(failedTable):
            self.logger.info('\n')
            self.logger.info('now start to repair %s'%tableName)
            table = self._repair_table(company, reportTime, reportType, tableName)
            #if len(table) == 0:
            #    self.logger.info(
            #        "failed to repair %s %s,it may be not configure in repair_lists of interpreterBase.json"
            #        % (self.gConfig['sourcefile'], tableName))
            #    return isRepaired
            if len(table) > 0:
                self.names[tableName].update({'tableName': tableName, 'table': table, 'tableBegin': True, "tableEnd": True})
                self._parse_table(tableName)
            #if isRepaired:
            #    self.logger.info('success to repaire %s %s from repaire_lists!' % (sourceFile, tableName))
                repairedTable.add(tableName)
        #self._parse_table(tableName)
        '''
        if self.names["公司地址"] == NULLSTR:
            self.names["公司地址"] = self.names["注册地址"]
        self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                         , '公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                         , '报告类型': self.names['报告类型']
                                         , '公司地址': self.names['公司地址']
                                         , '行业分类': self.names['行业分类']
                                         , '货币单位': self.names['货币单位']})
        self.excelParser.writeToStore(self.names[tableName])
        self.sqlParser.writeToStore(self.names[tableName])
        '''
        return repairedTable


    def _parse_table(self,tableName):
        if self.names["公司地址"] == NULLSTR:
            self.names["公司地址"] = self.names["注册地址"]
        self.names[tableName].update({'公司代码': self.names['公司代码'], '公司简称': self.names['公司简称']
                                         , '公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                         , '报告类型': self.names['报告类型']
                                         , '公司地址': self.names['公司地址']
                                         , '行业分类': self.names['行业分类']
                                         , '货币单位': self.names['货币单位']
                                         , '货币名称': self.names['货币名称'] })
        self.excelParser.writeToStore(self.names[tableName])
        self.sqlParser.writeToStore(self.names[tableName])


    def _repair_table(self,company,reportTime,reportType,tableName):
        assert company != NULLSTR and reportTime != NULLSTR and reportType != NULLSTR and tableName != NULLSTR \
              ,"Parameter: company(%s) reportTime(%s) reportType(%s) tableName(%s) must not be NULL!"%(company,reportTime,reportType,tableName)
        repair_lists = self.gJsonBase['repair_lists']
        table = list()
        sourceFile = NULLSTR
        try:
            tableList = repair_lists[company][reportType][reportTime]['tableList']
            tableFile = repair_lists[company][reportType][reportTime]['tableFile']
            filePath = repair_lists['filePath']
            if isinstance(tableList,list) and tableName in tableList and tableFile != NULLSTR:
                sourceFile = os.path.join(filePath,tableFile)
                if os.path.exists(sourceFile):
                    dataFrame = pd.read_excel(sourceFile,sheet_name=tableName,header=None,dtype=str)
                    dataFrame.fillna(NULLSTR,inplace=True)
                    table = np.array(dataFrame).tolist()
                else:
                    self.logger.info('%s failed to load data from %s'%(self.gConfig['sourcefile'],sourceFile))
            else:
                self.logger.warning("%s failed to find %s in taleList %s"%(self.gConfig['sourcefile'],tableName,tableList))
        except XLRDError as e:
            print(e)
            self.logger.info('failed to find %s in %s'%(tableName,sourceFile))
        except Exception as e:
            print(e)
            self.logger.info('failed to fetch config %s %s %s %s in repair_lists of interpreterBase.json!'
                              %(company,reportTime,reportType,tableName))
        table = [list(map(lambda x:str(x).replace('\n',NULLSTR),row)) for row in table]
        return table


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
            if unit == NULLSTR:
                self.logger.warning('unit is NULLSTR!')
            else:
                self.logger.warning('%s is not the unit of currency!'%unit)
        return unitStandardize


    def initialize(self,dictParameter=None):
        for tableName in self.tableNames:
            self.names.update({tableName:{'tableName':NULLSTR
                                          ,'公司名称':NULLSTR,'公司代码':NULLSTR,'公司简称':NULLSTR
                                          ,'报告时间':NULLSTR,'报告类型':NULLSTR,"公司地址":NULLSTR,'行业分类':NULLSTR
                                          ,'货币单位': 1 #货币单位默认为1
                                          ,'货币名称': NULLSTR
                                          ,"注册地址": NULLSTR
                                          ,'table':NULLSTR,'tableBegin':False,'tableEnd':False
                                          ,"page_numbers":list()}})
        self.names.update({'unit':NULLSTR,'currency':NULLSTR,'company':NULLSTR,'time':NULLSTR,'address':NULLSTR})
        for commonField,_ in self.commonFileds.items():
            self.names.update({commonField:NULLSTR})
            self.names.update({'货币名称': NULLSTR})
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

