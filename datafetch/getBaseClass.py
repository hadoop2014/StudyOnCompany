import re
import os

numeric_types = (int,float)

#数据读写处理的基类
class getdataBase():
    def __init__(self,gConfig):
        self.gConfig = gConfig
        self.dataset_name = self.get_dataset_name(self.gConfig)
        self.data_path = os.path.join(self.gConfig['data_directory'],self.dataset_name)
        self.rawshape = self.get_rawshape(self.gConfig)
        self.resizedshape = self.rawshape
        self.classnum = self.get_classnum(self.gConfig)  #每个数据集的类别数
        self.classes = []
        self.cpu_num = self.gConfig['cpu_num']
        self.train_iter=None
        self.test_iter=None
        self.unitestIsOn = self.gConfig['unittestIsOn'.lower()]
        self.ctx = self.gConfig['ctx']
        if os.path.exists(self.data_path) == False:
            os.makedirs(self.data_path)


    def load_data(self,*args):
        pass


    def get_classnum(self,gConfig):
        is_find = False
        dataset_name = self.get_dataset_name(gConfig)
        #for key in gConfig:
        #    if key.find('.') >= 0:
        #        dataset_key = re.findall('(.*)\.', key).pop().lower()
        #        if dataset_key == dataset_name:
        #            is_find = True
        #            break
        if dataset_name not in self.gConfig.keys():
        #if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             % (dataset_name, gConfig['datasetlist']))
        #return gConfig[dataset_name+'.classnum']
        return gConfig[dataset_name]['classnum']


    def get_rawshape(self,gConfig):
        is_find = False
        dataset_name = self.get_dataset_name(gConfig)
        #for key in gConfig:
        #    if key.find('.') >= 0:
        #        dataset_key = re.findall('(.*)\.',key).pop().lower()
        #        if dataset_key == dataset_name:
        #            is_find = True
        if dataset_name not in self.gConfig:
        #if is_find == False:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        #return [gConfig[dataset_name+'.channels'],gConfig[dataset_name+'.dim_x'],gConfig[dataset_name+'.dim_y']]
        return [gConfig[dataset_name]['channels'],gConfig[dataset_name]['dim_x'],gConfig[dataset_name]['dim_y']]


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
                    for (X, y) in getdata(self,batch_size):
                        yield (X, y)
                        break
                return reader()
            else:
                return getdata(self,batch_size)
        return wapper


    @getdataForUnittest.__get__(object)
    def getTrainData(self,batch_size):
        return self.train_iter


    @getdataForUnittest.__get__(object)
    def getTestData(self,batch_size):
        return self.test_iter


    @getdataForUnittest.__get__(object)
    def getValidData(self,batch_size):  # ,batch_size,time_steps):
        pass


    def endProcess(self):
        pass

