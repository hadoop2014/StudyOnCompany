#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 6/13/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAnalysize.py
# @Note    : 用于财务数据分析

from interpreterAnalysize.interpreterBaseClass import *
import pandas as pd
from ply import lex,yacc

class InterpreterAnalysize(InterpreterBase):
    def __init__(self,gConfig,memberModuleDict):
        super(InterpreterAnalysize, self).__init__(gConfig)
        self.dataVisualization = memberModuleDict['datavisualization']
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

        # dictionary of names
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
            '''expression : VISUALIZE TABLE'''
            tableName = p[2]
            self._process_visualize_table(tableName)

        #def p_expression_generate_table(p):
        #    '''expression : GENERATE TABLE'''
        #    self._process_generate_table()

        def p_error(p):
            if p:
                print("Syntax error at '%s:%s'" % (p.value,p.type))
            else:
                print("Syntax error at EOF page")

        # Build the docparser
        self.parser = yacc.yacc(outputdir=self.working_directory)

    def doWork(self,commond,lexer=None,debug=False,tracking=False):
        text = commond
        self.parser.parse(text,lexer=self.lexer,debug=debug,tracking=tracking)

    def _process_create_table(self,tableName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_create_table!')
            return
        sql_file = self.dictTables[tableName]['create']
        sql_file = os.path.join(self.program_directory,sql_file)
        create_sql = self._get_file_context(sql_file)
        isSuccess = self._sql_executer_script(create_sql)
        assert isSuccess,"failed to execute sql"

    def _process_visualize_table(self,tableName):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_visualize_table!')
            return
        visualize_file = self.dictTables[tableName]['visualize']
        if visualize_file == NULLSTR:
            self.logger.warning('the visualize of table %s is NULL,it can not be visualized!'%tableName)
            return
        #visualize_file = os.path.join(self.working_directory,visualize_file)
        self.dataVisualization.read_and_visualize(visualize_file,tableName)

    def _process_generate_table(self):
        if self.unitestIsOn:
            self.logger.info('Now in unittest mode,do nothing in _process_generate_table!')
            return

    def initialize(self):
        pass

def create_object(gConfig,memberModuleDict):
    interpreter=InterpreterAnalysize(gConfig, memberModuleDict)
    interpreter.initialize()
    return interpreter