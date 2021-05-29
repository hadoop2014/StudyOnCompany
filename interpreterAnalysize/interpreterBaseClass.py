#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 5/9/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAccounting.py
# @Note    : 用于从财务报表中提取财务数据

from modelBaseClass import *

#数据读写处理的基类
class InterpreterBase(ModelBase):
    def __init__(self,gConfig):
        super(InterpreterBase, self).__init__(gConfig)
        self._get_interpreter_keyword()


    def _get_interpreter_keyword(self):
        #编译器,文件解析器共同使用的关键字
        self.tokens = self.gJsonInterpreter['tokens']
        self.literals = self.gJsonInterpreter['literals']
        self.ignores = self.gJsonInterpreter['ignores']
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableNames = [tableName for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.dictTables = self._get_dict_tables(self.tableNames,self.dictTables)
        self.models = self.gJsonInterpreter['MODEL'].split('|')
        self.dictModels = self._get_models_parameters(self.models)


    def _get_models_parameters(self,models):
        """
        args:
            models - 当前解释器配置文件下的所有模型名称,定义在文件interpreterAnalysize.json,如:
            公司价格预测模型
        reutrn:
            dictModels - 把interpreterAnalysize.json中的模型配置参数和interpreterBase.json中的"模型公共参数"进行融合, 融合的规则:
            1) interpreterAnalysize.json中的模型配置参数 更新到 "模型公共参数" 中;
            2) 模型配置参数model所指定的子模型配置参数 更新到 interpreterAnalysize.json中的模型配置参数中;
        """
        assert isinstance(models,list) and len(models) > 0,"models(%s) must be a list and not be NULL!"% models
        dictModels = {}
        #dictModel = {}
        dictModelCommon = {}
        for key, value in self.gJsonBase['模型公共参数'].items():
            # 把模型公共参数进行展开, 放到一个dict中
            if isinstance(value, dict):
                dictModelCommon.update(value)
            else:
                dictModelCommon.update({key: value})
        for model in models:
            dictModel = dictModelCommon.copy()
            for key, value in self.gJsonInterpreter[model].items():
                if isinstance(value, dict):
                    dictModel.update(value)
                else:
                    dictModel.update({key : value})
            modelName = dictModel['model']
            if modelName in self.gJsonInterpreter.keys():
                for key, value in self.gJsonInterpreter[modelName].items():
                    if isinstance(value, dict):
                        dictModel.update(value)
                    else:
                        dictModel.update({key: value})
                #dictModel.update(self.gJsonInterpreter[modelName])
            else:
                self.logger.error('model(%s) is not configure in interpreterAnalysize.json!' % (modelName))
            dictModels.update({model : dictModel})
        return dictModels


    def interpretDefine(self):
        #定义一个解释器语言词法
        ...


    def initialize(self):
        #初始化一个解释器语言
        ...
