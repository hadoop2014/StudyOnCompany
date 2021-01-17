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
        self.dictTables = self._get_dict_tables()
        self.models = self.gJsonInterpreter['MODEL'].split('|')
        self.dictModels = self._get_models_parameters(self.models)


    def _get_dict_tables(self):
        dictTables = {keyword: value for keyword, value in self.gJsonInterpreter.items() if
                           keyword in self.tableNames}
        # 如果表的配置中,还有 parent这段,则要和父表的字段进行合并,合并的原则: 1) 子表的value是一个值,覆盖附表；2)value是列表,则追加到父表；3)value是dict,则进行递归
        for tableName, dictTable in dictTables.items():
            parent = dictTable['parent']
            if parent != NULLSTR:
                mergedDictTable = self._merged_dict_table(dictTable,dictTables[parent])
                dictTables[tableName].update(mergedDictTable)
        return dictTables


    def _merged_dict_table(self,dictTable,dictTableParent):
        """
        args:
            dictTable - 当前表的配置参数
            dictTableParent - 父表的配置参数
        reutrn:
            dictTableMerged - 当前表和父表融合后的配置, 融合的规则:
                '''
                1) 当前表的value是一个值, 则覆盖父表;
                2) 当前表的value是一个list,则追加到父表；
                3) 当前表的value是一个dict,则进行递归调用;
                '''
        """
        dictTableMerged = dictTableParent.copy()
        for key,value in dictTable.items():
            # 遍历子表的值, 和父表进行合并
            if isinstance(value, list):
                # 如果子表的值是列表,则追加到父表
                if len(value) != 0:
                    dictTableMerged.setdefault(key,[]).append(*value)
            elif isinstance(value, dict):
                # 如果子表的值是dict, 则进行递归
                dictTableMergedChild = self._merged_dict_table(value, dictTableMerged[key])
                dictTableMerged.update({key: dictTableMergedChild})
            else:
                # 如果子表的值非上述几中,则覆盖父表
                if key != 'parent':
                    # 避免迭代循环
                    dictTableMerged.update({key: value})
        return dictTableMerged


    def _get_models_parameters(self,models):
        """
        args:
            models - 当前解释器配置文件下的所有模型名称,定义在文件interpreterAnalysize.json,如:
                '''
                公司价格预测模型
                '''
        reutrn:
            dictModels - 把interpreterAnalysize.json中的模型配置参数和interpreterBase.json中的"模型公共参数"进行融合, 融合的规则:
                '''
                1) interpreterAnalysize.json中的模型配置参数 更新到 "模型公共参数" 中;
                2) 模型配置参数model所指定的子模型配置参数 更新到 interpreterAnalysize.json中的模型配置参数中;
                '''
        """
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
        ...


    def initialize(self):
        #初始化一个解释器语言
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)

