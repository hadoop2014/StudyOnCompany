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
        #self.interpreter_name = self._get_class_name(self.gConfig)
        #self.logging_directory = os.path.join(self.gConfig['logging_directory'], 'interpreter', self.interpreter_name)
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
        self.dictTokens = {token:value for token,value in self.gJsonInterpreter.items() if token in self.tokens}
        self.tableNames = [tableName for tableName in self.gJsonInterpreter['TABLE'].split('|')]
        self.dictTables = {keyword: value for keyword, value in self.gJsonInterpreter.items() if
                           keyword in self.tableNames}
        self.models = self.gJsonInterpreter['MODEL'].split('|')
        self.dictModels = self._get_models_parameters(self.models)


    def _get_models_parameters(self,models):
        # 把interpreterBase.json中的模型公共参数 和 interpreterAnalysize.json中的模型专用配置参数合并到一起
        assert isinstance(models,list) and len(models) > 0,"models(%s) must be a list and not be NULL!"% models
        dictModels = {}
        dictModel = {}
        for key, value in self.gJsonBase['模型公共参数'].items():
            # 把模型公共参数进行展开, 放到一个dict中
            if isinstance(value, dict):
                dictModel.update(value)
            else:
                dictModel.update({key: value})
        for model in models:
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
        pass


    def initialize(self):
        #初始化一个解释器语言
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
