import re
import os
import matplotlib.pyplot as plt

numeric_types = (int,float)

#数据读写处理的基类
class getdataBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.dataset_name = self.get_dataset_name(self.gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],self.dataset_name)
        self.working_directory = os.path.join(self.gConfig['working_directory'], 'docparser',
                                              self.get_dataset_name(gConfig))
        self.logging_directory = self.gConfig['logging_directory']
        self.data_directory = self.gConfig['data_directory']
        self.logging_directory = os.path.join(self.logging_directory, 'docparser', self.get_dataset_name(gConfig))
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        if os.path.exists(self.data_path) == False:
            os.makedirs(self.data_path)

    def load_data(self,*args):
        pass

    def get_dataset_name(self,gConfig):
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
                    pass
                return reader()
            else:
                return getdata(self,batch_size)
        return wapper

    @getdataForUnittest.__get__(object)
    def getTrainData(self,batch_size):
        pass


