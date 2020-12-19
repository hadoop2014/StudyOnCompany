from datafetch.getBaseClassM import  *
from datafetch.getBaseClassH import  *
#from datafetch.getBaseClassK import  *


class getMnistDataH(getdataBaseH):
    def __init__(self,gConfig):
        super(getMnistDataH,self).__init__(gConfig)


class getMnistDataM(getdataBaseM):
    def __init__(self,gConfig):
        super(getMnistDataM, self).__init__(gConfig)

#class getMnistDataK(getdataBaseK):
#    def __init__(self,gConfig):
#        super(getMnistDataK, self).__init__(gConfig)


class_selector = {
    "mxnet":getMnistDataM,
#    "tensorflow":getMnistDataM,
    "pytorch":getMnistDataH
#    "paddle":getMnistDataM,
#    'keras':getMnistDataK
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
