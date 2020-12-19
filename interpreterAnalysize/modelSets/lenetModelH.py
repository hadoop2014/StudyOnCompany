from interpreterAnalysize.modelSets.modelBaseClassH import *
from torch import nn
import torch.nn.functional as F


class lenet(nn.Module):
    def __init__(self,gConfig,input_channels,activation,input_dim_x,input_dim_y,classnum,compute_dim_xy):
        super(lenet,self).__init__()
        self.activation =activation # sigmoid
        conv1_channels = gConfig['conv1_channels']  # 6
        conv1_kernel_size = gConfig['conv1_kernel_size']  # 5
        conv1_strides = gConfig['conv1_strides']  # 1
        conv1_padding = gConfig['conv1_padding']  # 0
        pool1_size = gConfig['pool1_size']  # 2
        pool1_strides = gConfig['pool2_size']  # 2
        pool1_padding = gConfig['pool1_padding']  # 0
        conv2_channels = gConfig['conv2_channels']  # 16
        conv2_kernel_size = gConfig['conv2_kernel_size']  # 5
        conv2_strides = gConfig['conv2_striders']  # 1
        conv2_padding = gConfig['conv2_padding']  # 0
        pool2_size = gConfig['pool2_size']  # 2
        pool2_strides = gConfig['pool2_strides']  # 2
        pool2_padding = gConfig['pool2_padding']  # 0
        dense1_hiddens = gConfig['dense1_hiddens']  # 120
        dense2_hiddens = gConfig['dense2_hiddens']  # 84
        dense3_hiddens = gConfig['dense3_hiddens']  # 10
        dense3_hiddens = classnum
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=conv1_channels,
                               kernel_size=conv1_kernel_size,stride=conv1_strides,
                               padding=conv1_padding)
        out_dim_x,out_dim_y = compute_dim_xy(input_dim_x,input_dim_y,conv1_kernel_size,conv1_strides,conv1_padding)
        #out_dim_x = np.floor((input_dim_x - conv1_kernel_size + 2*conv1_padding)/conv1_strides) + 1
        #out_dim_y = np.floor((input_dim_y - conv1_kernel_size + 2*conv1_padding)/conv1_strides) + 1
        self.pool1 = partial(F.max_pool2d,kernel_size=pool1_size,stride=pool1_strides,
                         padding=pool1_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,pool1_size,pool1_strides,pool1_padding)
        #out_dim_x = np.floor((out_dim_x - pool1_size + 2*pool1_padding)/pool1_strides) + 1
        #out_dim_y = np.floor((out_dim_y - pool1_size + 2*pool1_padding)/pool1_strides) + 1
        self.conv2 = nn.Conv2d(in_channels=conv1_channels,out_channels=conv2_channels,
                               kernel_size=conv2_kernel_size,stride=conv2_strides,
                               padding=conv2_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,conv2_kernel_size,conv2_strides,conv2_padding)
        #out_dim_x = np.floor((out_dim_x - conv2_kernel_size + 2*conv2_padding)/conv2_strides) + 1
        #out_dim_y = np.floor((out_dim_y - conv2_kernel_size + 2*conv2_padding)/conv2_strides) + 1
        self.pool2 = partial(F.max_pool2d,kernel_size=pool2_size,stride=pool2_strides,
                         padding=pool2_padding)
        out_dim_x,out_dim_y = compute_dim_xy(out_dim_x,out_dim_y,pool2_size,pool2_strides,pool2_padding)
        #out_dim_x = np.floor((out_dim_x - pool2_size + 2*pool2_padding)/pool2_strides) + 1
        #out_dim_y = np.floor((out_dim_y - pool2_size + 2*pool2_padding)/pool2_strides) + 1
        in_features = int(out_dim_x*out_dim_y*conv2_channels)
        self.dense1 = nn.Linear(in_features=in_features,out_features=dense1_hiddens)
        self.dense2 = nn.Linear(in_features=dense1_hiddens,out_features=dense2_hiddens)
        self.dense3 = nn.Linear(in_features=dense2_hiddens,out_features=dense3_hiddens)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool2(x)
        pool_out_dim = int(np.prod(x.size()[1:]))
        x = x.view(-1,pool_out_dim)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        return x


class lenetModel(ModelBaseH):
    def __init__(self,gConfig):
        super(lenetModel,self).__init__(gConfig)
        #self.loss = nn.CrossEntropyLoss().to(self.ctx)
        #self.resizedshape = getdataClass.resizedshape
        #self.classnum = getdataClass.classnum
        #self.get_net()
        #self.optimizer = self.get_optimizer(self.gConfig['optimizer'],self.net.parameters())
        #self.input_shape = (self.batch_size,*self.resizedshape)


    def _init_parameters(self):
        super(lenetModel, self)._init_parameters()
        self.loss = nn.CrossEntropyLoss().to(self.ctx)
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.input_shape = (self.batch_size, *self.resizedshape)


    def get_net(self):
        activation = self.gConfig['activation']#sigmoid
        activation = self.get_activation(activation)
        input_channels, input_dim_x, input_dim_y = self.resizedshape
        self.net = lenet(self.gConfig,input_channels,activation,input_dim_x,input_dim_y,self.classnum,
                         ModelBaseH.compute_dim_xy)


    def run_train_loss_acc(self,X,y):
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info()
        self.optimizer.step()
        loss = loss.item()
        acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat = self.net(X)
        acc  = (y_hat.argmax(dim=1) == y).sum().item()
        loss = self.loss(y_hat, y).sum().item()
        return loss,acc


    def get_input_shape(self):
        return self.input_shape


    #def initialize(self,dictParameter = None):
    #    assert dictParameter is not None,"dictParameter must not be None!"
    #    getdataClass = dictParameter['getdataClass']
    #    self.resizedshape = getdataClass.resizedshape
    #    self.classnum = getdataClass.classnum
    #    super(lenetModel, self).initialize(dictParameter)


def create_object(gConfig):
    #用cnnModel实例化一个对象model
    #ckpt_used = gConfig['ckpt_used']
    #getdataClass = gConfig['getdataClass']
    model=lenetModel(gConfig=gConfig)#,getdataClass=getdataClass)
    #model.initialize(ckpt_used)
    return model