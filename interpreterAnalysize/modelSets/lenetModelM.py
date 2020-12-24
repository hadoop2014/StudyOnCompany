from mxnet.gluon import nn,loss as gloss
from interpreterAnalysize.modelSets.modelBaseClassM import *


class lenetModel(ModelBaseM):
    def __init__(self,gConfig):
        super(lenetModel,self).__init__(gConfig)


    def _init_parameters(self):
        super(lenetModel, self)._init_parameters()
        self.optimizer = self.gConfig['optimizer']
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.classnum = getdataClass.classnum
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), self.optimizer,
                                 {'learning_rate': self.learning_rate})
        self.input_shape = (self.batch_size, *self.resizedshape)


    def get_net(self):
        activation = self.gConfig['activation']#sigmoid
        conv1_channels = self.gConfig['conv1_channels'] #6
        conv1_kernel_size = self.gConfig['conv1_kernel_size']#5
        conv1_strides = self.gConfig['conv1_strides']#1
        conv1_padding = self.gConfig['conv1_padding'] #0
        pool1_size = self.gConfig['pool1_size']#2
        pool1_strides = self.gConfig['pool2_size']#2
        conv2_channels = self.gConfig['conv2_channels']#16
        conv2_kernel_size = self.gConfig['conv2_kernel_size']#5
        conv2_strides = self.gConfig['conv2_striders']#1
        conv2_padding = self.gConfig['conv2_padding'] #0
        pool2_size = self.gConfig['pool2_size']#2
        pool2_strides = self.gConfig['pool2_strides']#2
        dense1_hiddens = self.gConfig['dense1_hiddens']#120
        dense2_hiddens = self.gConfig['dense2_hiddens']#84
        dense3_hiddens = self.gConfig['dense3_hiddens']#10
        dense3_hiddens = self.classnum
        self.net.add(nn.Conv2D(channels=conv1_channels,kernel_size=conv1_kernel_size,
                               strides=conv1_strides,activation=self.get_activation(activation),
                               bias_initializer=init.Constant(self.init_bias)),
                nn.MaxPool2D(pool_size=pool1_size,strides=pool1_strides),
                nn.Conv2D(channels=conv2_channels,kernel_size=conv2_kernel_size,strides=conv2_strides,
                          activation=self.get_activation(activation),bias_initializer=init.Constant(self.init_bias)),
                nn.MaxPool2D(pool_size=pool2_size,strides=pool2_strides),
                #Dense会默认将（批量大小，通道，高，宽)形状的输入转换成（批量大小，通道＊高＊宽）的形状的输入
                nn.Dense(dense1_hiddens,activation=self.get_activation(activation),
                         bias_initializer=init.Constant(self.init_bias)),
                nn.Dense(dense2_hiddens,activation=self.get_activation(activation),
                         bias_initializer=init.Constant(self.init_bias)),
                nn.Dense(dense3_hiddens,bias_initializer=init.Constant(self.init_bias)))


    def run_train_loss_acc(self,X,y):
        with autograd.record():
            y_hat = self.net(X)
            loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step.asscalar() == 0:
            self.debug_info()
        self.trainer.step(self.batch_size)
        loss = loss.asscalar()
        y = y.astype('float32')
        acc= (y_hat.argmax(axis=1) == y).sum().asscalar()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        y_hat = self.net(X)
        acc  = (y_hat.argmax(axis=1) == y).sum().asscalar()
        loss = self.loss(y_hat, y).sum().asscalar()
        return loss,acc


    def get_input_shape(self):
        return self.input_shape


    def get_classes(self):
        return self.classnum


def create_object(gConfig):
    model=lenetModel(gConfig=gConfig)
    #model.initialize(ckpt_used)
    return model