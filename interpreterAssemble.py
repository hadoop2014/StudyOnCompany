#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 8/23/2020 5:03 PM
# @Author  : wu.hao
# @File    : interpreterAssemble.py
# @Note    : 根据checkbook.json中的配置装载模块
from datafetch import getConfig
from baseClass import *

#数据读写处理的基类
class InterpreterAssemble(BaseClass):
    def __init__(self):
        gConfig = getConfig.get_config()
        gConfig.update({"gJsonBase".lower():getConfig.get_config_json_base()})
        super(InterpreterAssemble, self).__init__(gConfig)
        self.check_book = getConfig.get_check_book()


    def get_gConfig(self,interpreterName,unittestIsOn = False):
        assert self.check_book is not None,'check_book is None ,it may be some error occured when open the checkbook.json!'
        gConfig = getConfig.get_config(self.check_book[interpreterName]["config_file"])
        program_directory = self.check_book[interpreterName]['program_directory']
        gJsonInterpreter, gJsonBase = getConfig.get_config_json(self.check_book[interpreterName]['config_json'])
        gConfig.update({"gJsonInterpreter".lower(): gJsonInterpreter})
        gConfig.update({"gJsonBase".lower(): gJsonBase})
        gConfig.update({'program_directory':program_directory})
        #gConfig.update({'debugIsOn'.lower():debugIsOn})
        if unittestIsOn:
            gConfig.update({'unittestIsOn'.lower():unittestIsOn})
        # 在unitest模式,这三个数据是从unittest.main中设置，而非从文件中读取．
        return gConfig


    def interpreter_assemble(self,interpreterName,unittestIsOn = False):
        gConfig = getConfig.get_config()
        assert interpreterName in gConfig['interpreterlist'], 'interpreterName(%s) is invalid,it must one of %s' % \
                                                    (interpreterName, gConfig['interpreterlist'])
        gConfig = self.get_gConfig(interpreterName,unittestIsOn)
        memberModuleDict = self.member_module_assemble(gConfig,self.check_book[interpreterName]["member_modules"])
        module = __import__(self.check_book[interpreterName]['module'],
                            fromlist=(self.check_book[interpreterName]['module'].split('.')[-1]))
        interpreter = getattr(module, 'create_object')(gConfig, memberModuleDict)
        return interpreter


    def member_module_assemble(self,gConfig,memberModules):
        moduleDict = {}
        if not memberModules:
            return moduleDict
        for moduleName,module in memberModules.items():
            moduleCase = __import__(module,fromlist=(module.split('.')[-1]))
            moduleCase = getattr(moduleCase, 'create_object')(gConfig)
            moduleDict.update({moduleName:moduleCase})
        return moduleDict


    def get_interpreter_nature(self,unittestIsOn = False):
        gConfig = getConfig.get_config()
        interpreterDict = {}
        for interpreterName in gConfig['interpreterlist']:
            self.validate_parameter(interpreterName)
            if interpreterName == 'nature':
                continue
            interpreter = self.interpreter_assemble(interpreterName,unittestIsOn)
            interpreterDict.update({interpreterName:interpreter})
        interpreterName = 'nature'
        gConfig = self.get_gConfig(interpreterName,unittestIsOn)
        #program_directory = self.check_book[interpreterName]['program_directory']
        #gConfig.update({'program_directory':program_directory})
        module = __import__(self.check_book[interpreterName]['module'],
                            fromlist=(self.check_book[interpreterName]['module'].split('.')[-1]))
        interpreterNature = getattr(module, 'create_object')(gConfig, interpreterDict)
        return interpreterNature


    def validate_parameter(self, interpreterName):
        assert self.check_book[interpreterName]["module"] != '' \
               and self.check_book[interpreterName]['config_json'] != '' \
               ,'the config (module or config_json) of %s in checkbook.json must not be empty'%interpreterName


    def get_data_class(self,gConfig,dataset):
        assert self.check_book is not None, 'check_book is None ,it may be some error occured when open the checkbook.json!'
        assert dataset in self.check_book['datafetch'].keys(),'dataset(%s) must be in list:%s'\
                                                              %(dataset,self.check_book['datafetch'].keys())
        module = __import__(self.check_book['datafetch'][dataset],
                            fromlist=(self.check_book['datafetch'][dataset].split('.')[-1]))
        getdataClass = getattr(module, 'create_model')(gConfig)
        return getdataClass
