from datafetch.getBaseClassH import  *

class getMnistDataH(getdataBaseH):
    def __init__(self,gConfig):
        super(getMnistDataH,self).__init__(gConfig)


class_selector = {
    "pytorch":getMnistDataH
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
