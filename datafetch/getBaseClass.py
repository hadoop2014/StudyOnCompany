from baseClass import *

numeric_types = (int,float)

#数据读写处理的基类
class getdataBase(BaseClass):
    def __init__(self,gConfig):
        super(getdataBase, self).__init__(gConfig)
        #self.gConfig = gConfig
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
        dataset_name = self.get_dataset_name(gConfig)
        if dataset_name not in self.gConfig.keys():
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             % (dataset_name, gConfig['datasetlist']))
        return gConfig[dataset_name]['classnum']


    def get_rawshape(self,gConfig):
        ...

    '''
    def get_rawshape(self,gConfig):
        dataset_name = self.get_dataset_name(gConfig)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name]['channels'],gConfig[dataset_name]['dim_x'],gConfig[dataset_name]['dim_y']]
    '''

    def get_dataset_name(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        return dataset_name


    # Ｋ折交叉验证
    def get_k_fold_data(self, k, features):
        assert k > 1, 'k折交叉验证算法中，必须满足条件ｋ>1'
        fold_size = len(features) // k
        X_train, y_train = None, None
        X_valid, y_valid = None, None
        i = np.random.randint(k)
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part = features[idx.start:idx.stop]
            if j == i:
                X_valid = X_part
            elif X_train is None:
                X_train = X_part
            else:
                # X_train = nd.concat(X_train, X_part, dim=0)
                X_train.extend(X_part)
                # y_train = nd.concat(y_train, y_part, dim=0)
        return X_train, X_valid


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

