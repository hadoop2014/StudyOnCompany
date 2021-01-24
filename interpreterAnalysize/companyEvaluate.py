#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 12/3/2020 5:03 PM
# @Author  : wu.hao
# @File    : companyEvaluate.py
# @Note    : 用于公司价格分析

from interpreterAnalysize.interpreterBaseClass import *


class CompanyEvaluate(InterpreterBase):
    def __init__(self,gConfig):
        super(CompanyEvaluate,self).__init__(gConfig)
        self.checkpointIsOn = gConfig['checkpointIsOn'.lower()]

    '''
    def _read_and_analysize(self,tableName,scale):
        dataFrame = self._sql_to_dataframe(tableName,tableName,scale)
    '''

    def _sql_to_dataframe(self,tableName,sourceTableName,scale):
        if scale == "批量":
            assert ('公司简称' in self.gConfig.keys() and self.gConfig['公司简称'] != NULLSTR) \
                and ('报告时间' in self.gConfig.keys() and self.gConfig['报告时间'] != NULLSTR) \
                and ('报告类型' in self.gConfig.keys() and self.gConfig['报告类型'] != NULLSTR)\
                ,"parameter 公司简称(%s) 报告时间(%s) 报告类型(%s) is not valid parameter"\
                 %(self.gConfig['公司简称'],self.gConfig['报告时间'],self.gConfig['报告类型'])
            #批量处理模式时会进入此分支
            dataFrame = pd.DataFrame(self._get_stock_list(self.gConfig['公司简称']),columns=self.gJsonBase['stockcodeHeader'])
            condition = self._get_condition(dataFrame)
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
            sql = sql + '\nwhere ' + condition
            #sql = sql + '\nwhere (' + ' or '.join(['公司简称 =' + '\'' +  company + '\''   for company in self.gConfig['公司简称']]) + ')'
            #sql = sql + '    and (' + ' or '.join(['报告时间 =' + '\'' + reporttime + '\'' for reporttime in self.gConfig['报告时间']]) + ')'
            #sql = sql + '    and (' + ' or '.join(['报告类型 =' + '\'' + reportype + '\'' for reportype in self.gConfig['报告类型']]) + ')'
        else:
            sql = ''
            sql = sql + '\nselect * '
            sql = sql + '\nfrom %s' % (sourceTableName)
        order = self.dictTables[tableName]["order"]
        if isinstance(order,list) and len(order) > 0:
            sql = sql + '\norder by ' + ','.join(order)
        dataframe = pd.read_sql(sql, self._get_connect())
        return dataframe


    def _get_class_name(self, gConfig):
        return "analysize"


    def initialize(self,dictParameter = None):
        super(CompanyEvaluate,self).initialize()
        if dictParameter is not None:
            self.gConfig.update(dictParameter)


def create_object(gConfig):
    companyEvaluate = CompanyEvaluate(gConfig)
    companyEvaluate.initialize()
    return companyEvaluate