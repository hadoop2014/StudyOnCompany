#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据
from functools import reduce
from baseClass import *
import itertools


#数据读写处理的基类
class InterpreterBase(BaseClass):
    def __init__(self,gConfig):
        super(InterpreterBase, self).__init__(gConfig)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
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
        self.criticals = self._get_criticals()
        #self.reportAlias = self.gJsonInterpreter['reportAlias']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableAlias = self.gJsonInterpreter['tableAlias']  # 必须在self.tableNames之前定义
        self.tableNames = self._get_tablenames()
        self.dictTables = {keyword: value for keyword,value in self.gJsonInterpreter.items() if keyword in self.tableNames}
        self.commonFields = self.gJsonInterpreter['公共表字段定义']
        self.tableKeyword = self.gJsonInterpreter['TABLE']
        self.dictKeyword = self._get_keyword(self.tableKeyword)
        self.dictTables = self._fields_replace_punctuate(self.dictTables)
        self.dictTables = self._fields_standardized(self.dictTables)
        self.dictReportType = self._get_report_type_tables(self.dictTables)


    def _get_tablenames(self):
        # 去掉tableName中带正则表达式保留字符的串,如 ()[].
        tableNames = [tableName for tableName in self.gJsonInterpreter['TABLE'].split('|') if not re.search('[()\[\]]',tableName)]
        # tableNames标准化,去掉正则表达式中的$^
        tableNames = [self._standardize(self.gJsonInterpreter['TABLEStandardize'], tableName)
                           for tableName in tableNames]
        tableNames = [self._get_tablename_alias(tableName) for tableName in tableNames if tableName is not NaN]
        tableNames = list(set(tableNames + list(self.tableAlias.values())))
        return tableNames


    def _get_criticals(self):
        criticals = self.gJsonInterpreter['CRITICAL'].split('|')
        # 去掉所有带正则表达critical
        criticals = [critical for critical in criticals if not re.search('[().*\[\]]+',critical)]
        criticals = [self._get_critical_alias(cirtical) for cirtical in criticals]
        criticals = list(set(criticals + list(self.criticalAlias.values())))
        return criticals


    def _get_table_field(self,tableName):
        # 把标准化后的fieldName 及 fieldAlias中的key字段,融合成 tablefield,用于从财报中提取所需要的字段.
        assert tableName in self.dictTables.keys,'%s is not in dictTable.keys() which is %s'%(tableName,self.dictTables.keys)
        fieldName = self.dictTables[tableName]['fieldName']
        dictAlias = self.dictTables[tableName]['fieldAlias']
        tableField = fieldName + [fieldName for fieldName,aliasName in dictAlias.items() if not aliasName.startswith('虚拟字段')]
        return tableField


    def _get_report_type_tables(self,dictTables):
        # 根据每张表中 'reportType'的配置,生成每中报告类型(reportType)中包含了多少张 表(tableName)
        dictReportType = dict()
        for tableName in dictTables.keys():
            dictTemp =  dict(list(zip(dictTables[tableName]['reportType'], [tableName] * len(dictTables[tableName]['reportType']))))
            for reportType, valueList in dictTemp.items():
                dictReportType.setdefault(reportType, []).append(valueList)
        return dictReportType


    def _fields_standardized(self,dictTables):
        for tableName in dictTables.keys():
            for tokenName in ['field','header']:
                standardFields = self._get_standardized_keyword(dictTables[tableName][tokenName + 'Name']
                                                                ,dictTables[tableName][tokenName + 'Standardize'])
                standardFields = [field for field in standardFields if field is not NaN]
                dictTables[tableName].update({tokenName + 'Name':standardFields})
                discardFields = self._get_standardized_keyword(dictTables[tableName][tokenName + 'Discard']
                                                               ,dictTables[tableName][tokenName + 'Standardize'])
                discardFields = [field for field in discardFields if field is not NaN]
                dictTables[tableName].update({tokenName + 'Discard':discardFields})
                virtualPassMatching = self.gJsonInterpreter['VIRTUALPASSMATCHING']
                virtualStoping = self.gJsonInterpreter['VIRTUALSTOPING']
                dictAlias = {}
                for key, value in dictTables[tableName][tokenName + 'Alias'].items():
                    keyStandard = self._get_standardized_keyword(key,dictTables[tableName][tokenName + 'Standardize'])
                    valueStandard = self._get_standardized_keyword(value, dictTables[tableName][tokenName + 'Standardize'])
                    if keyStandard is NaN or valueStandard is NaN:
                        continue
                    if valueStandard == virtualPassMatching:
                        # 把PASSMATCHING加入到dictTocken中
                        dictAlias.update({key: virtualPassMatching})
                    elif valueStandard == virtualStoping:
                        dictAlias.update({key: virtualStoping})
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
                # 计算 年度报告,半年度报告,第一季度报告,第三季度报告 需要解析多少张表
                dictTables[tableName].update(
                    {'max'+tokenName.title()+'Len':reduce(max,list(map(len,dictTables[tableName][tokenName + 'Name'])))})
            horizontalTable = self.dictTables[tableName]['horizontalTable']
            if horizontalTable:
                #正对水平表,maxFieldLen和maxHeaderLen需要互换
                dictTables[tableName]['maxHeaderLen'],dictTables[tableName]['maxFieldLen'] \
                    = dictTables[tableName]['maxFieldLen'] ,dictTables[tableName]['maxHeaderLen']
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
        field = field.replace('，','、') #解决海康威视2016年合并现金流量表中出现: 处置固定资产，无形资产和其他长期资产收回的现金净额
        #field = re.sub('(^\\d).','\g<1>．',field)
        #field = re.sub('(^[一二三四五六七八九〇]).','\g<1>、',field)
        #解决海螺水泥2014年报中,所有的处置,购置 误写为 购臵,处臵
        field = re.sub('([处购])臵','\g<1>置',field)
        #解决华侨城A 2018年年报无形资产情况表中出现 '1、将净利润调节为经营活动现金流量',替换成 '1．将净利润调节为经营活动现金流量'
        field = re.sub('(^\\d)、','\g<1>．',field)
        #解决麦克韦尔2018年报中出现"一”
        field = re.sub('"一”', '“－”', field)
        #解决海天味业2015年年报中,出现''2 现金及现金等价物净变动情况''
        field = re.sub('(^\\d)(?=[\\u4E00-\\u9FA5])', '\g<1>．', field)
        #解决双林生物2019年年报,现金流量补充资料中出现 （1）将净利润调节为经营活动现金流量净利润,其标号为（1）
        field = re.sub('^（(\\d)）', '\g<1>．', field)
        #解决尚荣医疗2019年年报中,无形资产情况表中出现' 一．二．',全部替换为' 一、 二、'
        field = re.sub('(^[一二三四五六七八九〇])．','\g<1>、',field)
        #解决尚荣医疗2019年年报中,无形资产情况表中出现“一” 情况
        field = re.sub('“一”(号填列)','“－”\g<1>',field)
        #field = re.sub('\\[注\\]$',NULLSTR,field) #解决海康威视2014年报,主要会计数据最后一个字段出现 归属于上市公司股东的净资产（元）[注]
        return field


    def _replace_value(self,value):
        value = value.replace('(', '（').replace(')', '）').replace(' ', NULLSTR)
        #数值中-,.号是有意义的不能随便替换
        return value


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
