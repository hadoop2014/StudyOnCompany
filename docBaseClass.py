# -*- coding: utf-8 -*-
# @Time    : 9/25/2019 5:03 PM
# @Author  : wuhao
# @File    : modelBaseClass.py
import os
import time
import re
import json
import logging


#深度学习模型的基类
class docBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.start_time = time.time()
        self.working_directory = os.path.join(self.gConfig['working_directory'],self.get_parser_name(gConfig))
        self.logging_directory = self.gConfig['logging_directory']
        self.data_directory = self.gConfig['data_directory']
        self.logging_directory = os.path.join(self.logging_directory, self.get_parser_name(gConfig))
        self.model_savefile = os.path.join(self.working_directory,
                                           self.get_parser_name(self.gConfig) + '.model')
        self.checkpoint_filename = self.get_parser_name(self.gConfig)+'.ckpt'
        self.debugIsOn = self.gConfig['debugIsOn'.lower()]
        self.check_book = self.get_check_book()

    def get_check_book(self):
        check_file = os.path.join(self.gConfig['config_directory'], self.gConfig['check_file'])
        check_book = None
        if os.path.exists(check_file):
            with open(check_file, encoding='utf-8') as check_f:
                check_book = json.load(check_f)
        else:
            raise ValueError("%s is not exist,you must create first!" % check_file)
        return check_book

    def get_parser_name(self,gConfig):
        parser_name = re.findall('docParser(.*)', self.__class__.__name__).pop().lower()
        assert parser_name in gConfig['docformatlist'], \
            'docformatlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['docformatlist'], parser_name, self.__class__.__name__)
        return parser_name

    def saveCheckpoint(self):
        pass

    def debug_info(self,*args):
        pass

    def clear_logging_directory(self,logging_directory):
        assert logging_directory == self.logging_directory ,\
            'It is only clear logging directory, but %s is not'%logging_directory
        files = os.listdir(logging_directory)
        for file in files:
            full_file = os.path.join(logging_directory,file)
            if os.path.isdir(full_file):
                self.clear_logging_directory(full_file)
            else:
                try:
                    os.remove(full_file)
                except:
                   print('%s is not be removed'%full_file)
