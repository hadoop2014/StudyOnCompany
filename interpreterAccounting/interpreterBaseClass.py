#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据
from functools import reduce
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
        self.tableNames = [self._standardize(self.gJsonInterpreter['TABLEStandardize'],tableName)
                           for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.tableNames = list(set([self._get_tablename_alias(tableName) for tableName in self.tableNames]))
        self.dictTables = {keyword: value for keyword,value in self.gJsonInterpreter.items() if keyword in self.tableNames}
        self.commonFileds = self.gJsonInterpreter['公共表字段定义']
        self.tableKeyword = self.gJsonInterpreter['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)
        self.dictTables = self._fields_replace_punctuate(self.dictTables)
        self.dictTables = self._fields_standardized(self.dictTables)


    def _fields_standardized(self,dictTables):
        for tableName in dictTables.keys():
            for tokenName in ['field','header']:
                standardFields = self._get_standardized_keyword(dictTables[tableName][tokenName + 'Name']
                                                                ,dictTables[tableName][tokenName + 'Standardize'])
                dictTables[tableName].update({tokenName + 'Name':standardFields})
                discardFields = self._get_standardized_keyword(dictTables[tableName][tokenName + 'Discard']
                                                               ,dictTables[tableName][tokenName + 'Standardize'])
                dictTables[tableName].update({tokenName + 'Discard':discardFields})
                virtualPassMatching = self.gJsonInterpreter['VIRTUALPASSMATCHING']
                dictAlias = {}
                for key, value in dictTables[tableName][tokenName + 'Alias'].items():
                    keyStandard = self._get_standardized_keyword(key,dictTables[tableName][tokenName + 'Standardize'])
                    valueStandard = self._get_standardized_keyword(value, dictTables[tableName][tokenName + 'Standardize'])
                    if valueStandard == virtualPassMatching:
                        # 把PASSMATCHING加入到dictTocken中
                        dictAlias.update({key: virtualPassMatching})
                    elif keyStandard != valueStandard:
                        dictAlias.update({keyStandard: valueStandard})
                    else:
                        self.logger.warning("%s has same field after standardize in fieldAlias:%s %s" % (tableName, key, value))
                if len(dictTables[tableName][tokenName + 'Alias'].keys()) != len(dictAlias.keys()):
                    self.logger.warning('It is same field after standard in fieldAlias of %s:%s'
                                        % (tableName, ' '.join(dictTables[tableName][tokenName + 'Alias'].keys())))
                dictTables[tableName].update({tokenName + 'Alias':dictAlias})
        return dictTables


    def _fields_replace_punctuate(self,dictTables):
        #对header,field中的英文标点符号进行替换
        virtualPassMatching = self.gJsonInterpreter['VIRTUALPASSMATCHING']

        for tableName in dictTables.keys():
            for tokenName in ['field','header']:
                dictTables[tableName].update({tokenName + 'Name':list(map(self._replace_fieldname,dictTables[tableName][tokenName + 'Name']))})
                #dictTables[tableName].update({'fieldName':list(map(self._replace_fieldname,dictTables[tableName]['fieldName']))})
                dictTables[tableName].update({tokenName + 'Discard': list(map(self._replace_fieldname, dictTables[tableName][tokenName + 'Discard']))})
                #dictTables[tableName].update({'fieldDiscard': list(map(self._replace_fieldname, dictTables[tableName]['fieldDiscard']))})
                #dictTables[tableName].update({'fieldFirst': self._replace_fieldname(self.dictTables[tableName]['fieldFirst'])})
                #dictTables[tableName].update({'fieldLast': self._replace_fieldname(self.dictTables[tableName]['fieldLast'])})
                #headerAlias中有正则表达式,不能直接替换
                dictAlias = dict()
                for key,value in dictTables[tableName][tokenName+'Alias'].items():
                    if value != virtualPassMatching:
                        dictAlias.update({self._replace_fieldname(key):self._replace_fieldname(value)})
                    else:
                        #PASSMATCHING用于正则表达式,不能做符号替换
                        dictAlias.update({key:value})
                dictTables[tableName].update({tokenName+'Alias':dictAlias})
                #dictTables[tableName].update(
                #    {tokenName + 'Alias':dict(zip(list(map(self._replace_fieldname, dictTables[tableName][tokenName +'Alias'].keys()))
                #                       ,list(map(self._replace_fieldname,dictTables[tableName][tokenName + 'Alias'].values()))))})
            dictTables[tableName].update(
                {'maxFieldLen':reduce(max,list(map(len,dictTables[tableName]['fieldName'])))})
        self.logger.warning("函数_fields_replace_punctuate把interpreterAccounting.json中配置的所有表的字段名中的英文标点替换为中文的,"
                            + "而'headerFirst','headerSecond'中采用了正则表达式,这要求正则表达式中不要出现'('')''-''.'等字符!")
        return dictTables


    def _get_standardized_field(self,fieldList,tableName):
        assert fieldList is not None,'sourceRow(%s) must not be None'%fieldList
        fieldStandardize = self.dictTables[tableName]['fieldStandardize']
        if isinstance(fieldList,list):
            standardizedFields = [self._standardize(fieldStandardize,field) for field in fieldList]
        else:
            standardizedFields = self._standardize(fieldStandardize,fieldList)
        return standardizedFields


    def _get_standardized_keyword(self,keywordList,standardPattern):
        assert keywordList is not None,'sourceRow(%s) must not be None'%keywordList
        #fieldStandardize = self.dictTables[tableName]['fieldStandardize']
        if isinstance(keywordList,list):
            standardizedKeywords = [self._standardize(standardPattern,keyword) for keyword in keywordList]
        else:
            standardizedKeywords = self._standardize(standardPattern,keywordList)
        return standardizedKeywords


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


    def _replace_fieldname(self, field):
        #所有英文标点替换成中文标点,避免和正则表达式中的保留字冲突
        field = field.replace('(','（').replace(')','）').replace(' ',NULLSTR)
        field = field.replace(':','：').replace('-','－').replace('—','－')
        #―‖为鱼跃医疗2016年年报中出现的不规范字符,等同于““
        field = field.replace('―','“').replace('‖','“')
        #主营业务分行业情况 中用到了.,不能直接替换,要采用re.sub替换
        field = field.replace('.','．')
        #field = re.sub('(^\\d).','\g<1>．',field)
        #field = re.sub('(^[一二三四五六七八九〇]).','\g<1>、',field)
        #解决海螺水泥2014年报中,所有的处置,购置 误写为 购臵,处臵
        field = re.sub('([处购])臵','\g<1>置',field)
        #解决华侨城A 2018年年报无形资产情况表中出现 '1、将净利润调节为经营活动现金流量',替换成 '1．将净利润调节为经营活动现金流量'
        field = re.sub('(^\\d)(、)','\g<1>．',field)
        #解决海天味业2015年年报中,出现''2 现金及现金等价物净变动情况''
        field = re.sub('(^\\d)(?=[\\u4E00-\\u9FA5])', '\g<1>．', field)
        #解决尚荣医疗2019年年报中,无形资产情况表中出现' 一．二．',全部替换为' 一、 二、'
        field = re.sub('(^[一二三四五六七八九〇])．','\g<1>、',field)
        # 解决尚荣医疗2019年年报中,无形资产情况表中出现“一” 情况
        field = re.sub('“一”(号填列)','“－”\g<1>',field)
        return field


    def _replace_value(self,value):
        value = value.replace('(', '（').replace(')', '）').replace(' ', NULLSTR)
        #数值中-,.号是有意义的不能随便替换
        return value


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
