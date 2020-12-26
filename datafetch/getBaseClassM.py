from datafetch.getBaseClass import  *
import os
import sys

class getdataBaseM(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseM,self).__init__(gConfig)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        import mxnet.gluon.data as gdata
        self.dataset_selector = {
            'mnist': gdata.vision.MNIST,
            'fashionmnist': gdata.vision.FashionMNIST,
            'cifar10': gdata.vision.CIFAR10
            #'hotdog':HOTDOG,
            #'pikachu':PIKACHU
        }
        self.load_data(root=self.data_path)


    def get_transformer(self,train=True):
        import mxnet.gluon.data as gdata
        #默认情况下train = True和train=False使用的变换相同
        transformer = []
        if self.resize is not None and self.resize != 0:
            transformer += [gdata.vision.transforms.Resize(self.resize)]
            self.resizedshape = [self.rawshape[0], self.resize, self.resize]
        transformer += [gdata.vision.transforms.ToTensor()]
        transformer = gdata.vision.transforms.Compose(transformer)
        return transformer


    def load_data(self,root=""):
        from mxnet.gluon import data as gdata
        root = os.path.expanduser(root)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=False)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        transformer = self.get_transformer(train=True)
        self.train_iter = gdata.DataLoader(train_data.transform_first(transformer),
                                           self.batch_size, shuffle=True,
                                           num_workers=num_workers)
        transformer = self.get_transformer(train=False)
        self.test_iter = gdata.DataLoader(test_data.transform_first(transformer),
                                          self.batch_size, shuffle=False,
                                          num_workers=num_workers)



