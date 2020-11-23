#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 9/25/2019 5:03 PM
# @Author  : wu.hao
# @File    : crawlBaseClass.py

import time
from interpreterCrawl.interpreterBaseClass import *


#深度学习模型的基类
class CrawlBase(InterpreterBase):
    def __init__(self,gConfig):
        super(CrawlBase, self).__init__(gConfig)
        self.start_time = time.time()
        self.logging_directory = self.gConfig['logging_directory']
        self.data_directory = self.gConfig['data_directory']
        self.logging_directory = os.path.join(self.logging_directory, 'docparser', self._get_class_name(gConfig))
        self.model_savefile = os.path.join(self.working_directory, self._get_class_name(self.gConfig) + '.model')
        self.checkpoint_filename = self._get_class_name(self.gConfig) + '.ckpt'
        self.source_directory = os.path.join(self.data_directory, self.gConfig['source_directory'])
        self.sourceFile = os.path.join(self.data_directory, self.gConfig['source_directory'],
                                       self.gConfig['sourcefile'])
        self.taskResult = os.path.join(self.gConfig['working_directory'], self.gConfig['taskResult'.lower()])
        self.checkpointIsOn = self.gConfig['checkpointIsOn'.lower()]


    def _get_class_name(self, gConfig):
        parser_name = re.findall('Crawl(.*)', self.__class__.__name__).pop().lower()
        #assert parser_name in gConfig['docformatlist'], \
        #    'docformatlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
        #    (gConfig['docformatlist'], parser_name, self.__class__.__name__)
        return parser_name


    def saveCheckpoint(self):
        pass


    def getSaveFile(self):
        if self.model_savefile == NULLSTR:
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile) == False:
                return None
                # 文件不存在
        return self.model_savefile


    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd(), self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)


    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        pass
        return


    def debug(self, layer, name=NULLSTR):
        pass


    def clear_logging_directory(self, logging_directory):
        assert logging_directory == self.logging_directory, \
            'It is only clear logging directory, but %s is not' % logging_directory
        files = os.listdir(logging_directory)
        for file in files:
            full_file = os.path.join(logging_directory, file)
            if os.path.isdir(full_file):
                self.clear_logging_directory(full_file)
            else:
                try:
                    os.remove(full_file)
                except:
                    print('%s is not be removed' % full_file)


    def initialize(self):
        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)