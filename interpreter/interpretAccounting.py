#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpretAccounting.py
# @Note    : 用于从财务报表中提取财务数据

from interpreter.interpretBaseClass import *


class InterpretAccounting(InterpretBase):
    def __init__(self,gConfig,docParser,excelParser,sqlParser):
        super(InterpretAccounting, self).__init__(gConfig)
        self.docParser = docParser
        self.excelParser = excelParser
        self.sqlParser = sqlParser
        #self.initialize()
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

        #t_ignore = " \t\n"
        t_ignore = self.ignores

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

        # dictionary of names
        self.names = {}

        def p_statement_grouphalf(p):
            '''statement : statement ')'
                         | statement '）' '''
            p[0] = p[1]

        def p_statement_statement(p):
            '''statement : statement expression
                         | expression '''
            p[0] = p[1]

        def p_expression_reduce(p):
            '''expression : fetchtable expression
                          | fetchdata expression
                          | skipword '''
            p[0] = p[1]

        def p_fetchtable_search(p):
            '''fetchtable : TABLE optional TIME optional UNIT finis '''
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[3],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            unit = p[5].split(':')[-1].split('：')[-1]
            currency = self.names['currency']
            company = self.names['company']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency,company)

        def p_fetchtable_searchnotime(p):
            '''fetchtable : TABLE optional UNIT finis '''
            #第二个语法针对的是主要会计数据
            tableName = self._get_tablename_alias(str.strip(p[1]))
            self.logger.info("fetchtable %s -> %s %s page %d" % (p[1],tableName,p[3],self.currentPageNumber))
            if self._is_reatch_max_pages(self.names[tableName],tableName) is True:
                self.docParser.interpretPrefix = NULLSTR
                return
            unit = p[3].split(':')[-1].split('：')[-1]
            currency = self.names['currency']
            interpretPrefix = '\n'.join([slice for slice in p if slice is not None]) + '\n'
            tableBegin = True
            self._process_fetch_table(tableName,tableBegin,interpretPrefix,unit,currency)

        def p_fetchtable_timetime(p):
            '''fetchtable : TABLE optional TIME TIME'''
            #处理主要会计数据的的场景,存在第一次匹配到,又重新因为表头而第二次匹配到的场景
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

        def p_fetchtable_reatchtail(p):
            '''fetchtable : TABLE optional UNIT NUMERIC
                          | TABLE optional TIME optional UNIT finis NUMERIC
                          | TABLE optional NUMERIC
                          | TABLE optional TIME NUMERIC'''
            #处理在页尾搜索到fetch的情况,NUMERIC为页尾标号,设置tableBegin = False,则_merge_table中会直接返回,直接搜索下一页
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

        def p_fetchtable_skipword(p):
            '''fetchtable : TABLE HEADER
                          | TABLE PUNCTUATION
                          | TABLE optional TABLE
                          | TABLE optional PUNCTUATION'''
            #去掉了语法TABLE term,该语法和TABLE optional NUMERIC冲突
            #去掉合并资产负债表项目
            interpretPrefix = '\n'.join([str(slice) for slice in p if slice is not None]) + '\n'
            self.logger.error("fetchtable in wrong mode,prefix: %s page %d"%(interpretPrefix.replace('\n','\t'),self.currentPageNumber))
            pass

        def p_fetchdata_title(p):
            '''fetchdata : COMPANY TIME UNIT '''
            self.names.update({'公司名称':p[1]})
            self.names.update({'报告时间':p[2]})
            self.names.update({'报告类型':p[3]})
            self.logger.info('fetchdata title %s %s%s page %d'
                             % (self.names['公司名称'],self.names['报告时间'],self.names['报告类型'],self.currentPageNumber))
            p[0] = p[1] + p[2] + p[3]

        def p_fetchdata_criticaldouble(p):
            '''fetchdata : CRITICAL DISCARD CRITICAL term
                         | CRITICAL term CRITICAL DISCARD '''
            self.names.update({self._get_critical_alias(p[1]):p[2]})
            self.names.update({self._get_critical_alias(p[3]):p[4]})
            print('fetchdata critical',p[1],'->',self._get_critical_alias(p[1]),p[2],p[3],'->',self._get_critical_alias(p[3]),p[4])

        def p_fetchdata_critical(p):
            '''fetchdata : CRITICAL term fetchdata
                         | CRITICAL term '''
            critical = self._get_critical_alias(p[1])
            self.names.update({critical:p[2]})
            self.logger.info('fetchdata critical %s->%s %s page %d' % (p[1],critical,p[2],self.currentPageNumber))

        def p_fetchdata_skipword(p):
            '''fetchdata : COMPANY TIME DISCARD
                         | COMPANY DISCARD
                         | COMPANY PUNCTUATION
                         | COMPANY NUMERIC
                         | COMPANY UNIT
                         | COMPANY error
                         | COMPANY empty
                         | CRITICAL CRITICAL'''
            p[0] = p[1]

        def p_skipword_group(p):
            '''skipword : '(' skipword ')'
                        | '(' skipword '）'
                        | '（' skipword '）'
                        | '（' skipword ')' '''
            p[0] = p[2]

        def p_skipword(p):
            '''skipword : useless skipword
                        | term skipword
                        | useless
                        | term
                        | '(' skipword error '''
            p[0] = p[1]

        def p_useless_reduce(p):
            '''useless : '(' useless ')'
                       | '(' useless '）'
                       | '（' useless '）'
                       | '（' useless ')'
                       | '-' useless '''
            p[0] = p[1]

        def p_useless(p):
            '''useless : PUNCTUATION
                       | DISCARD
                       | WEBSITE
                       | EMAIL
                       | NAME
                       | HEADER
                       | TIME
                       | UNIT
                       | CURRENCY
                       | '-'
                       | '%'
                       | '％' '''
            p[0] = p[1]

        def p_term_group(p):
            '''term : '(' term ')'
                    |  '（' term '）'
                    | '-' term %prec UMINUS '''
            p[0] = -p[2]  #财务报表中()表示负值

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

        def p_optional_optional(p):
            '''optional : DISCARD optional
                        | optional fetchdata DISCARD'''
            #第2条规则解决大立科技：2018年年度报告,合并资产负债表出现在表尾,而第二页开头为"浙江大立科技股份有限公司 2018 年年度报告全文"的场景
            p[0] = p[1] + p[2]

        def p_optional(p):
            '''optional : empty
                        | COMPANY '''
            if p.slice[1].type == 'COMPANY':
                self.names['company'] = p[1]
            p[0] = p[1]
            print('optional',p[0])

        def p_finis(p):
            '''finis : empty
                     | CURRENCY
                     | HEADER'''
            if p.slice[1].type == 'CURRENCY':
                p[1] = p[1].split(':')[-1].split('：')[-1]
                self.names['currency'] = p[1]
            p[0] = p[1]
            print('finis',p[0])

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

    def doWork(self,docParser,lexer=None,debug=False,tracking=False):
        for data in docParser:
            self.currentPageNumber = docParser.index
            text = docParser._get_text(data)
            self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)
        sourceFile = os.path.split(self.docParser.sourceFile)[-1]
        self.logger.info('%s\tcritical:'%sourceFile + ','.join([self.names['公司名称'],self.names['报告时间'],self.names['报告类型']
                          ,str(self.names['股票代码']),self.names['股票简称']]))
        self.logger.info('%s\tprocess_info:'%sourceFile + str(self.sqlParser.process_info))
        failedTable = set(self.tableNames).difference(set(self.sqlParser.process_info.keys()))
        if len(failedTable) == 0:
            self.logger.info("%s\tall table is success fetched!"%(sourceFile))
        else:
            self.logger.info('%s\ttable(%s) is failed to fetch'
                             %(sourceFile,failedTable))
        docParser._close()
        return self.sqlParser.process_info

    def _process_fetch_table(self, tableName, tableBegin, interpretPrefix, unit=NULLSTR, currency=NULLSTR, company=NULLSTR):
        assert tableName is not None and tableName != NULLSTR, 'tableName must not be None'
        self.names[tableName].update({'tableName': tableName, 'unit': unit, 'currency': currency
                                     ,'company':company
                                     ,'tableBegin': tableBegin
                                     ,'page_numbers': self.names[tableName]['page_numbers'] + list([self.currentPageNumber])})
        if self.names[tableName]['tableEnd'] == False:
            self.names[tableName].update({'股票代码': self.names['股票代码'], '股票简称': self.names['股票简称']
                                             , '公司名称': self.names['公司名称'], '报告时间': self.names['报告时间']
                                             , '报告类型': self.names['报告类型']})
            self.docParser._merge_table(self.names[tableName], interpretPrefix)
            if self.names[tableName]['tableEnd'] == True:
                self.excelParser.writeToStore(self.names[tableName])
                self.sqlParser.writeToStore(self.names[tableName])
        self.logger.info('\nprefix: %s:'%interpretPrefix.replace('\n','\t') + str(self.names[tableName]))

    def _is_reatch_max_pages(self, fetchTable,tableName):
        maxPages = self.dictTables[tableName]['maxPages']
        if len(fetchTable['page_numbers']) >= maxPages:
            isReatchMaxPages = True
            self.logger.error("table %s is reatch max page numbers:%d >= %d"
                             %(tableName,len(fetchTable['page_numbers']),maxPages))
        elif fetchTable['tableEnd'] == True:
            isReatchMaxPages = True
        else:
            isReatchMaxPages = False
        return isReatchMaxPages

    def initialize(self):
        for tableName in self.tableNames:
            self.names.update({tableName:{'tableName':NULLSTR,'time':NULLSTR,'unit':NULLSTR,'currency':NULLSTR
                                          ,'company':NULLSTR,'公司名称':NULLSTR,'股票代码':NULLSTR,'股票简称':NULLSTR
                                          ,'报告时间':NULLSTR,'报告类型':NULLSTR
                                          ,'table':NULLSTR,'tableBegin':False,'tableEnd':False
                                          ,"page_numbers":list()}})
        self.names.update({'unit':NULLSTR,'currency':NULLSTR,'company':NULLSTR,'time':NULLSTR})
        for commonField,_ in self.commonFileds.items():
            self.names.update({commonField:NULLSTR})
        for cirtical in self.criticals:
            self.names.update({self._get_critical_alias(cirtical):NULLSTR})

def create_object(gConfig,docParser=None,excelParser=None,sqlParser=None):
    interpreter=InterpretAccounting(gConfig, docParser, excelParser, sqlParser)
    interpreter.initialize()
    return interpreter

