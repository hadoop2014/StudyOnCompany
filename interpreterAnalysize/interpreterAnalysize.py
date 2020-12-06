#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAnalysize.py
# @Note    : 用于财务数据分析

from interpreterAnalysize.interpreterBaseClass import *
from ply import lex,yacc


class InterpreterAnalysize(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterAnalysize, self).__init__(gConfig)
        self.dataVisualization = memberModuleDict['dataVisualization']
        self.companyEvaluate = memberModuleDict['companyEvaluate']
        #self.modelPropose = memberModuleDict['modelPropose']
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
        t_ignore_COMMENT = r'#.*'


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


        def p_statement_expression(p):
            '''statement : statement expression
                         | expression'''
            pass


        def p_expression_create_table(p):
            '''expression : CREATE TABLE'''
            tableName = p[2]
            self._process_create_table(tableName)


        def p_expression_visualize_table(p):
            '''expression : SCALE VISUALIZE TABLE'''
            tableName = p[3]
            scale = p[1]
            self._process_visualize_table(tableName,scale)


        def p_expression_train_model(p):
            '''expression : TRAIN MODEL'''
            modelName = p[2]
            self._process_train_model(modelName)


        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
                self.logger.error("Syntax error at '%s:%s'" % (p.value, p.type))
            else:
                print("Syntax error at EOF page")
                self.logger.error("Syntax error at EOF page")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)


    def doWork(self,commond,lexer=None,debug=False,tracking=False):
        text = commond
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)


    def _process_create_table(self,tableName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_create_table!')
            return
        for reportType in self.gConfig['报告类型']:
            sql_file = self.dictTables[tableName]['create']
            tablePrefix = self._get_tableprefix_by_report_type(reportType)
            sql_file = os.path.join(self.program_directory,tablePrefix,sql_file)
            if not os.path.exists(sql_file):
                self.logger.error('create script is not exist,you must create it first :%s!'%sql_file)
                continue
            create_sql = self._get_file_context(sql_file)
            isSuccess = self._sql_executer_script(create_sql)
            assert isSuccess,"failed to execute sql"


    def _process_visualize_table(self,tableName,scale):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        visualize_file = self.dictTables[tableName]['visualize']
        if visualize_file == NULLSTR:
            self.logger.warning('the visualize of table %s is NULL,it can not be visualized!'%tableName)
            return
        self.dataVisualization.initialize(self.gConfig)
        self.dataVisualization.read_and_visualize(visualize_file,tableName,scale)


    def _process_visualize_table_batch(self,tableName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        visualize_file = self.dictTables[tableName]['visualize']
        if visualize_file == NULLSTR:
            self.logger.warning('the visualize of table %s is NULL,it can not be visualized!'%tableName)
            return
        self.dataVisualization.read_and_visualize(visualize_file,tableName)


    def _process_train_model(self,modelName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        self.logger.info("Reatch the interpreterAnalysize just for debug : train %s" % modelName)

    '''
    def _process_generate_table(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_generate_table!')
            return
   '''


    def initialize(self,dictParameter=None):
        if dictParameter is not None:
            self.gConfig.update(dictParameter)


def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterAnalysize(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter