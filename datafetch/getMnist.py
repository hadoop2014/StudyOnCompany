from datafetch.getBaseClassM import  *
from datafetch.getBaseClassH import  *
#from datafetch.getBaseClassK import  *


class getMnistDataH(getdataBaseH):
    def __init__(self,gConfig):
        super(getMnistDataH,self).__init__(gConfig)
        self.load_data(resize=self.resize, root=self.data_path)

    def get_rawshape(self,gConfig):
        dataset_name = self.get_dataset_name(gConfig)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name]['channels'],gConfig[dataset_name]['dim_x'],gConfig[dataset_name]['dim_y']]


class getMnistDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getMnistDataM, self).__init__(gConfig)
        #self.load_data(resize=self.resize, root=self.data_path)

    def get_rawshape(self,gConfig):
        dataset_name = self.get_dataset_name(gConfig)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name]['channels'],gConfig[dataset_name]['dim_x'],gConfig[dataset_name]['dim_y']]

#class getMnistDataK(getdataBaseK):
#    def __init__(self,gConfig):
#        super(getMnistDataK, self).__init__(gConfig)


class_selector = {
    "mxnet":getMnistDataM,
    "tensorflow":getMnistDataH,
    "pytorch":getMnistDataH
#    "paddle":getMnistDataM,
#    'keras':getMnistDataK
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
