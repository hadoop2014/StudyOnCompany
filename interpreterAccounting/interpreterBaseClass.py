#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据
from ply import lex
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
        self.reportAlias = self.gJsonInterpreter['reportAlias']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableAlias = self.gJsonInterpreter['tableAlias']
        #tableNames标准化,去掉正则表达式中的$^
        self.tableNames = [self._standardize("[\\u4E00-\\u9FA5]{3,}",tableName) for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.tableNames = list(set([self._get_tablename_alias(tableName) for tableName in self.tableNames]))
        self.dictTables = {keyword: value for keyword,value in self.gJsonInterpreter.items() if keyword in self.tableNames}
        self.commonFileds = self.gJsonInterpreter['公共表字段定义']
        self.tableKeyword = self.gJsonInterpreter['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)
        self.dictLexers = self._construct_lexers()


    def _construct_lexers(self):
        dictLexer = dict()
        for tableName in self.dictTables.keys():
            dictTokens = self._get_dict_tokens(tableName)
            lexer = self._get_lexer(dictTokens,tableName)
            dictLexer.update({tableName:{'lexer':lexer,"dictToken":dictTokens}})
            self.logger.info('success to create lexer for %s!' %tableName)
        return dictLexer


    def _get_lexer(self,dictTokens,tableName):
        # Tokens
        # 采用动态变量名
        tokens = list(dictTokens.keys())
        local_name = locals()
        for token in tokens:
            local_name['t_' + token] = dictTokens[token]
        #self.logger.info(
        #    '%s:\n'%tableName + str({key: value for key, value in local_name.items() if key.split('_')[-1] in tokens}).replace("',","'\n"))

        t_ignore = " \n"


        def t_newline(t):
            r'\n+'
            t.lexer.lineno += t.value.count("\n")

        def t_error(t):
            print("Illegal character '%s'" % t.value[0])
            t.lexer.skip(1)
            #t.type = 'ILLEGALFIELD'
            #return t


        # Build the lexer
        lexer = lex.lex(outputdir=self.working_directory)
        return lexer


    def _get_dict_tokens(self,tableName):
        #对所有的表字段进行标准化
        standardFields = self._get_standardized_field(self.dictTables[tableName]['fieldName'], tableName)
        if len(standardFields) != len(set(standardFields)):
            self.logger.info("the fields of %s has duplicated!" % tableName)
        #去掉标准化字段后的重复字段
        standardFields = list(set(standardFields))
        #建立token和标准化字段之间的索引表
        fieldsIndex = dict([('FIELD' + str(index), field) for index, field in enumerate(standardFields)])
        #对字段别名表进行标准化,并去重
        dictAlias = {}
        for key, value in self.dictTables[tableName]['fieldAlias'].items():
            keyStandard = self._get_standardized_field(key, tableName)
            valueStandard = self._get_standardized_field(value, tableName)
            if keyStandard != valueStandard:
                dictAlias.update({keyStandard: valueStandard})
            else:
                self.logger.warning("%s has same field after standardize in fieldAlias:%s %s"%(tableName,key,value))
        #判断fieldAlias中经过标准化后是否有重复字段,如果存在,则配置是不合适的
        if len(self.dictTables[tableName]['fieldAlias'].keys()) != len(dictAlias.keys()):
            self.logger.warning('It is duplicated field in fieldAlias of %s' % tableName)
        #判断fieldAlias中是否存在fieldName中不存在的字段,如果存在,则配置上存在错误.
        fieldDiff = set(dictAlias.values()).difference(standardFields)
        if len(fieldDiff) > 0:
            if NaN in fieldDiff:
                self.logger.error('error in fieldAlias of %s, NaN is exists'%tableName)
            else:
                self.logger.error("error in fieldAlias of %s,field not exists : %s"%(tableName,' '.join(list(fieldDiff))))
        #在dictAlias中去掉fieldName中不存在的字段
        dictAlias = dict([(key,value) for key,value in dictAlias.items() if value not in fieldDiff])
        #对fieldAlias和fieldName进行合并
        dictMerged = {}
        for key, value in dictAlias.items():
            dictMerged.setdefault(value, []).append(key)
        for key in standardFields:
            dictMerged.setdefault(key,[]).append(key)

        #构造dictTokens,token搜索的正则表达式即字段名的前面加上patternPrefix,后面加上patternSuffix
        patternPrefix = self.dictTables[tableName]['patternPrefix']
        patternSuffix = self.dictTables[tableName]['patternSuffix']
        dictTokens = dict()
        for token,field in fieldsIndex.items():
            #生成pattern时要逆序排列,确保长的字符串在前面
            fieldList = sorted(dictMerged[field],key = lambda x:len(x),reverse=True)
            pattern = [patternPrefix+field+patternSuffix for field in fieldList]
            dictTokens.update({token:'|'.join(pattern)})
        #最后把ILLEGALFIELD加上
        #dictTokens.update({'ILLEGALFIELD':self.gJsonInterpreter['ILLEGALFIELD']})
        return dictTokens


    def _get_standardized_field(self,fieldList,tableName):
        assert fieldList is not None,'sourceRow(%s) must not be None'%fieldList
        fieldStandardize = self.dictTables[tableName]['fieldStandardize']
        if isinstance(fieldList,list):
            standardizedFields = [self._standardize(fieldStandardize,field) for field in fieldList]
        else:
            standardizedFields = self._standardize(fieldStandardize,fieldList)
        return standardizedFields


    def _get_report_alias(self, report):
        aliasedReport = self._alias(report, self.reportAlias)
        return aliasedReport


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


    def _deduplicate(self, field1, field2):
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
