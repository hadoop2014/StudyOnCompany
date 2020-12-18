from datafetch.getBaseClass import *
import sys

class getdataBaseH(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseH,self).__init__(gConfig)
        #self.dataset_name = self.get_dataset_name(self.gConfig)
        #self.data_path = os.path.join(self.gConfig['data_directory'], self.dataset_name)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        from torchvision import datasets
        self.dataset_selector={
            'mnist':datasets.MNIST,
            'fashionmnist':datasets.FashionMNIST,
            'cifar10':datasets.CIFAR10
        }
        self.load_data(resize=self.resize, root=self.data_path)


    def load_data(self,resize=None,root=""):
        from torch import utils
        from torchvision import transforms
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [transforms.ToTensor()]
        transformer += [transforms.Normalize((0.1307,),(0.3081,))]
        transformer = transforms.Compose(transformer)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.ctx == 'gpu' else {'num_workers':num_workers}
        self.train_iter = utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,**kwargs)
        self.test_iter = utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=True,**kwargs)
