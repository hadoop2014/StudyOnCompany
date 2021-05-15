#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据

from baseClass import *

#数据读写处理的基类
class InterpreterBase(BaseClass):
    def __init__(self,gConfig):
        super(InterpreterBase, self).__init__(gConfig)
        self.mainprogram = os.path.join(self.program_directory,self.gConfig['mainprogram'])
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        self._get_interpreter_keyword()
        self.database._create_tables(list(self.dictTables.keys()),self.dictTables,self.commonFields)
        self._load_data()


    def _load_data(self,input=None):
        file_context = self._get_file_context(self.mainprogram)
        self._data = file_context
        self._index = 0
        self._length = len(self._data)


    def _get_text(self,page=None):
        pageText = self._data
        return pageText


    def _get_class_name(self, gConfig):
        #获取解释器的名称
        dataset_name = re.findall('Interpreter(.*)', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['interpreterlist'], \
            'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['interpreterlist'], dataset_name, self.__class__.__name__)
        return dataset_name


    def _get_interpreter_keyword(self):
        super(InterpreterBase,self)._get_interpreter_keyword()
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gJsonInterpreter['tokens']
        self.literals = self.gJsonInterpreter['literals']
        self.ignores = self.gJsonInterpreter['ignores']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.dictTokens.update({"CATEGORY":'|'.join(self.gJsonInterpreter[self.gJsonInterpreter['CATEGORY']].keys())})
        self.tableNames = [tableName for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        #self.dictTables = {keyword: value for keyword, value in self.gJsonInterpreter.items() if
        #                   keyword in self.tableNames}
        self.dictTables = self._get_dict_tables(self.tableNames,self.dictTables)
        self.websites = self.gJsonInterpreter['WEBSITE'].split('|')
        self.models = self.gJsonInterpreter['MODEL'].split('|')


    def _get_keyword(self,tableKeyword):
        #获取解析文件所需的关键字
        dictKeyword = {keyword:value for keyword,value in self.gJsonInterpreter.items() if keyword in tableKeyword}
        return dictKeyword


    def _get_invalid_field(self):
        return [NONESTR,NULLSTR]


    def interpretDefine(self):
        #定义一个解释器语言词法
        pass


    def initialize(self):
        #初始化一个解释器语言
        pass
