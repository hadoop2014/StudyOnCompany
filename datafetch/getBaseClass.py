#!/usr/bin/env Python
# coding   : utf-8
from baseClass import *

numeric_types = (int,float)

#数据读写处理的基类
class GetdataBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.dataset_name = self._get_class_name(self.gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],self.dataset_name)
        self.logging_directory = self.gConfig['logging_directory']
        self.data_directory = self.gConfig['data_directory']
        self.logging_directory = os.path.join(self.logging_directory, 'docparser', self._get_class_name(gConfig))
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        self.cpu_num = self.gConfig['cpu_num']
        self.train_iter=None
        self.test_iter=None
        self.rawshape = self.get_rawshape(self.gConfig)


    def get_classnum(self,gConfig):
        is_find = False
        dataset_name = self.get_dataset_name(gConfig)
        for key in gConfig:
            if key.find('.') >= 0:
                dataset_key = re.findall('(.*)\.', key).pop().lower()
                if dataset_key == dataset_name:
                    is_find = True
                    break
        if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             % (dataset_name, gConfig['datasetlist']))
        return gConfig[dataset_name+'.classnum']


    def get_rawshape(self,gConfig):
        is_find = False
        dataset_name = self.get_dataset_name(gConfig)
        for key in gConfig:
            if key.find('.') >= 0:
                dataset_key = re.findall('(.*)\.',key).pop().lower()
                if dataset_key == dataset_name:
                    is_find = True
        if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name+'.channels'],gConfig[dataset_name+'.dim_x'],gConfig[dataset_name+'.dim_y']]


    def get_dataset_name(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name


    def load_data(self,*args):
        pass


    def _get_class_name(self, gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name


    # 装饰器，用于在unittest模式下，只返回一个数据，快速迭代
    @staticmethod
    def getdataForUnittest(getdata):
        def wapper(self, batch_size):
            if self.unitestIsOn == True:
                # 仅用于unitest测试程序
                def reader():
                    for (X, y) in getdata(self, batch_size):
                        yield (X, y)
                        break

                return reader()
            else:
                return getdata(self, batch_size)
        return wapper


    @getdataForUnittest.__get__(object)
    def getTrainData(self, batch_size):
        return self.train_iter


    @getdataForUnittest.__get__(object)
    def getTestData(self, batch_size):
        return self.test_iter


    @getdataForUnittest.__get__(object)
    def getValidData(self, batch_size):  # ,batch_size,time_steps):
        pass


    def endProcess(self):
        pass


