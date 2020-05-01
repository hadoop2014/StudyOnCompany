import re
import os
import ply.lex as lex
import ply.yacc as yacc

#数据读写处理的基类
class interpretBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.interpreter_name = self.get_interpreter_name(self.gConfig)
        self.working_directory = os.path.join(self.gConfig['working_directory'],'interpreter',self.interpreter_name)
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.interpretDefine()
        self.grammarDefine()

    def get_interpreter_name(self,gConfig):
        #获取解释器的名称
        dataset_name = re.findall('interpret(.*)', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['interpreterlist'], \
            'interpreterlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name

    def interpretDefine(self):
        #定义一个解释器语言词法
        pass

    def grammarDefine(self):
        #定义语法
        pass

    def initialize(self):
        #初始化一个解释器语言
        pass
