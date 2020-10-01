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
        self.interpreter_name = self._get_class_name(self.gConfig)
        self.logging_directory = os.path.join(self.gConfig['logging_directory'], 'interpreter', self.interpreter_name)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self._get_interpreter_keyword()


    def _get_class_name(self, gConfig):
        #获取解释器的名称
        dataset_name = re.findall('Interpreter(.*)', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['interpreterlist'], \
            'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['interpreterlist'], dataset_name, self.__class__.__name__)
        return dataset_name


    def _get_interpreter_keyword(self):
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gJsonInterpreter['tokens']
        self.literals = self.gJsonInterpreter['literals']
        self.ignores = self.gJsonInterpreter['ignores']
        self.referenceAlias = self.gJsonInterpreter['referenceAlias']
        self.references = self.gJsonInterpreter['REFERENCE'].split('|')
        self.references = list(set([self._get_reference_alias(reference) for reference in self.references]))
        self.criticalAlias = self.gJsonInterpreter['criticalAlias']
        self.criticals = self.gJsonInterpreter['CRITICAL'].split('|')
        self.criticals = list(set([self._get_critical_alias(cirtical) for cirtical in self.criticals]))
        self.unitAlias = self.gJsonInterpreter['unitAlias']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableAlias = self.gJsonInterpreter['tableAlias']
        #tableNames标准化,去掉正则表达式中的$^
        self.tableNames = [self._standardize("[\\u4E00-\\u9FA5]{3,}",tableName) for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.tableNames = list(set([self._get_tablename_alias(tableName) for tableName in self.tableNames]))
        self.dictTables = {keyword: value for keyword,value in self.gJsonInterpreter.items() if keyword in self.tableNames}
        self.commonFileds = self.gJsonInterpreter['公共表字段定义']
        self.tableKeyword = self.gJsonInterpreter['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)


    def _get_unit_alias(self,unit):
        aliasedUnit = self._alias(unit, self.unitAlias)
        return aliasedUnit


    def _get_reference_alias(self,refernece):
        aliasedRefernece = self._alias(refernece, self.referenceAlias)
        return aliasedRefernece


    def _get_critical_alias(self,critical):
        aliasedCritical = self._alias(critical, self.criticalAlias)
        return aliasedCritical


    def _get_tablename_alias(self,tablename):
        aliasedTablename = self._alias(tablename, self.tableAlias)
        return aliasedTablename


    def _alias(self, name, dictAlias):
        alias = name
        aliasKeys = dictAlias.keys()
        if len(aliasKeys) > 0:
            if name in aliasKeys:
                alias = dictAlias[name]
        return alias


    def _unduplicate(self,field1,field2):
        unduplicate = list(field1)
        if unduplicate[-1] != field2:
            unduplicate = unduplicate + [field2]
        return unduplicate


    def _merge(self,field1, field2,isFieldJoin=True):
        if self._is_valid(field2):
            if self._is_valid(field1):
                if isFieldJoin == True:
                    return field1 + field2
                else:
                    return field2
            else:
                return field2
        else:
            if self._is_valid(field1):
                return field1
            else:
                #配合无形资产情况表中"headerStandardize": "^[\\u4E00-\\u9FA5|a-z|A-Z][\\u4E00-\\u9FA5|a-z|A-Z|0-9|\\-|,]*"情形
                #如果返回None,_is_header_in_row函数会返回True,导致误判
                return NULLSTR


    def _standardize(self,fieldStandardize,field):
        standardizedField = field
        if isinstance(field, str) and isinstance(fieldStandardize, str) and fieldStandardize !="":
            matched = re.search(fieldStandardize, field)
            if matched is not None:
                standardizedField = matched[0]
            else:
                standardizedField = NaN
        else:
            if not self._is_valid(field):
                standardizedField = NaN
        return standardizedField


    def _is_valid(self, field):
        isFieldValid = False
        if isinstance(field,str):
            if field not in self._get_invalid_field():
                isFieldValid = True
        return isFieldValid


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
