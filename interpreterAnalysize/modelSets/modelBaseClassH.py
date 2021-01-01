from interpreterAnalysize.interpreterBaseClass import *
from torch import optim,nn
from functools import partial
from torch.nn.utils.rnn import PackedSequence
import torch
from torchvision import models
from tensorboardX import SummaryWriter
from torchsummary import summary # 该1.8版本在pytorch 1.2.0版本下使用, add_graph函数存在异常
#import visdom
import re

#深度学习模型的基类
class ModelBaseH(InterpreterBase):
    def __init__(self,gConfig):
        super(ModelBaseH, self).__init__(gConfig)
        self.net = nn.Module()


    def _init_parameters(self):
        super(ModelBaseH, self)._init_parameters()
        self.learning_rate = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.viewIsOn = self.gConfig['viewIsOn']
        self.max_to_keep = self.gConfig['max_to_keep']
        self.ctx = self.get_ctx(self.gConfig['ctx'])
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.momentum = self.gConfig['momentum']
        self.initializer = self.gConfig['initializer']
        self.max_queue = self.gConfig['max_queue']
        self.weight_initializer = self.get_initializer(self.initializer)
        self.bias_initializer = self.get_initializer('constant')
        self.global_step = torch.tensor(0, dtype=torch.int64, device=self.ctx)
        #self.set_default_tensor_type()  #设置默认的tensor在ｇｐｕ还是在ｃｐｕ上运算


    def _get_class_name(self, gConfig):
        model_name = re.findall('(.*)Model', self.__class__.__name__).pop().lower()
        assert model_name in gConfig['modellist'], \
            'modellist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['modellist'], model_name, self.__class__.__name__)
        return model_name


    def get_net(self):
        return


    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = torch.device(type='cuda',index=0) #,index=0)
        else:
            ctx = torch.device(type='cpu')#,index=0)
        return ctx


    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return partial(nn.init.normal_,std=self.init_sigma)
        elif initializer == 'xavier':
            return partial(nn.init.xavier_uniform_)
        elif initializer == 'kaiming':
            #何凯明初始化法
            return partial(nn.init.kaiming_uniform_,mode='fan_in')
        elif initializer == 'constant':
            return partial(nn.init.constant_,val=self.init_bias)
        else:
            return None


    def params_initialize(self, module):
        if type(module) == nn.Linear or type(module) == nn.Conv2d:
            #print(self.weight_initializer,'\tnow initializer %s'%module)
            self.weight_initializer(module.weight.data)
            self.bias_initializer(module.bias.data)
        elif type(module) == nn.LSTM or type(module) == nn.GRU or type(module) == nn.RNN:
            self.weight_initializer(module.weight_hh_l0.data)
            self.weight_initializer(module.weight_ih_l0.data)
            self.bias_initializer(module.bias_hh_l0.data)
            self.bias_initializer(module.bias_ih_l0.data)


    def get_optimizer(self,optimizer,parameters):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        if optimizer == 'sgd':
            return optim.SGD(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adadelta':
            return optim.Adagrad(params=parameters,lr=self.learning_rate)
        elif optimizer == 'rmsprop':
            return optim.RMSprop(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adam':
            return optim.Adam(params=parameters,lr=self.learning_rate)
        elif optimizer == 'adagrad':
            return optim.Adagrad
        elif optimizer == 'momentum':
            return optim.SGD(params=parameters,lr=self.learning_rate,momentum=self.momentum)
        return None


    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        if activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif activation == 'relu':
            return torch.nn.ReLU()
        elif activation == 'tanh':
            return torch.nn.Tanh()


    def get_nonlinearity(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        return activation


    def get_context(self):
        return self.ctx


    def get_learningrate(self):
        return self.learning_rate


    def get_globalstep(self):
        return self.global_step.item()


    def get_input_shape(self):
        return tuple()


    def init_state(self):
        pass


    def saveCheckpoint(self):
        torch.save(self.net.state_dict(),self.model_savefile)
        torch.save(self.global_step,self.symbol_savefile)


    def getSaveFile(self):
        if self.model_savefile == '':
            self.model_savefile = None
            return None
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile)== False:
               return None
                #文件不存在
        return self.model_savefile


    def removeSaveFile(self):
        if self.model_savefile is not None:
            filename = os.path.join(os.getcwd() , self.model_savefile)
            if os.path.exists(filename):
                os.remove(filename)
        if self.symbol_savefile is not None:
            filename = os.path.join(os.getcwd(),self.symbol_savefile)
            if os.path.exists(filename):
                os.remove(filename)


    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass,epoch)
        self.writer.close()

        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test


    def debug_info(self,info = None):
        if self.debugIsOn == False:
            return
        if info is not None:
            print('debug:%s'%info)
            return
        for (key,layer) in self.net.named_children():
            self.debug(layer,':'.join([key,layer._get_name()]))
        print('\n')
        return


    def debug(self,layer,name=''):
        for (key,block) in layer.named_children():
            self.debug(block,":".join([name,key,block._get_name()]))
        for (key,parameter) in layer.named_parameters():
            print('\tdebug:%s(%s)' % (name, key),
                  '\tshape=', parameter.shape,
                  '\tdata.mean=%.6f' % parameter.data.mean().item(),
                  '\tgrad.mean=%.6f' % parameter.grad.mean().item(),
                  '\tdata.std=%.6f' % parameter.data.std(),
                  '\tgrad.std=%.6f' % parameter.grad.std())


    def image_record(self,global_step,tag,input_image):
        if global_step < self.gConfig['num_samples']:
            self.writer.add_image(tag,input_image,global_step)


    def run_train_loss_acc(self,X,y):
        loss,acc = None,None
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        loss,acc = None,None
        return loss,acc


    def get_batch_size(self,y):
        return y.size()[0]


    def train_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.init_state()  # 仅用于RNN,LSTM等
        self.net.train()
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                if not isinstance(X,PackedSequence):
                    X = np.array(X)
                y = np.array(y)
            self.image_record(self.global_step.item(), 'input/image', X[0])
            if not isinstance(X,PackedSequence):
                X = torch.tensor(X, device=self.ctx)
            #y = torch.tensor(y, device=self.ctx, dtype=torch.long)
            y = torch.tensor(y, device=self.ctx)
            loss, acc = self.run_train_loss_acc(X, y)
            loss_sum += loss
            acc_sum += acc
            #n += y.size()[0]
            n += self.get_batch_size(y)
            self.writer.add_scalar('train/loss', loss, self.global_step.item())
            self.writer.add_scalar('train/accuracy', acc, self.global_step.item())
            self.global_step += 1
        return loss_sum / n, acc_sum / n


    def evaluate_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.net.eval()
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                y = y.asnumpy()
            except:
                if not isinstance(X,PackedSequence):
                    X = np.array(X)
                y = np.array(y)
            if not isinstance(X,PackedSequence):
                X = torch.tensor(X,device=self.ctx)
            #y = torch.tensor(y,device=self.ctx,dtype=torch.long)
            y = torch.tensor(y, device=self.ctx)
            loss,acc = self.run_eval_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += self.get_batch_size(y)
        return loss_sum / n, acc_sum / n


    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        loss_train,acc_train = self.train_loss_acc(train_iter)
        #nd.waitall()
        if epoch % epoch_per_print == 0:
            # print(features.shape,labels.shape)
            loss_test,acc_test = self.evaluate_loss_acc(test_iter)
            self.writer.add_scalar('test/loss',loss_test,self.global_step.item())
            self.writer.add_scalar('test/accuracy',acc_test,self.global_step.item())
            self.run_matrix(loss_train, loss_test)  # 仅用于rnn,lstm,ssd等
            self.predict(self.net)  # 仅用于rnn,lstm,ssd等
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test


    def predict(self, net):
        pass


    def run_matrix(self, loss_train, loss_test):
        pass


    def grad_clipping(self,params, theta, device):
        norm = torch.tensor([0.0], device=device)
        for param in params:
            norm += (param.grad.data ** 2).sum()
        norm = norm.sqrt().item()
        if norm > theta:
            for param in params:
                param.grad.data *= (theta / norm)


    def set_default_tensor_type(self):
        if self.ctx == torch.device(type='cuda',index=0):
            # 如果设置为cuda的类型，则所有操作都在GPU上进行
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)


    @staticmethod
    def compute_dim_xy(input_dim_x,input_dim_y,kernel_size,strides,padding):
        out_dim_x = np.floor((input_dim_x - kernel_size + 2 * padding) / strides) + 1
        out_dim_y = np.floor((input_dim_y - kernel_size + 2 * padding) / strides) + 1
        return out_dim_x,out_dim_y


    def summary(self,net):
        summary(net, input_size=self.get_input_shape()[1:], batch_size=self.get_input_shape()[0],
                device=re.findall(r'(\w*)', self.ctx.__str__())[0])


    def add_graph(self,net):
        dummy_input = torch.zeros(*self.get_input_shape(),device=self.ctx)
        self.writer.add_graph(net,dummy_input)


    def initialize(self,dictParameter = None):
        assert dictParameter is not None,'dictParameter must not be None!'

        self.gConfig.update(dictParameter)
        self._init_parameters()

        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)
        self.clear_logging_directory(self.logging_directory)

        self.writer = SummaryWriter(logdir=self.logging_directory,max_queue=self.max_queue)
        #self.vis = visdom.Visdom(env='test')
        if 'pretrained' in self.gConfig:
            pass

        ckpt = self.getSaveFile()
        ckpt_used = self.gConfig['ckpt_used']
        if ckpt and ckpt_used:
            print("Reading model parameters from %s" % ckpt)
            self.net.load_state_dict(torch.load(ckpt))
            self.net.to(device=self.ctx)
            self.global_step = torch.load(self.symbol_savefile,map_location=self.ctx)
        else:
            print("Created model with fresh parameters.")
            self.net.to(device=self.ctx)
            self.net.apply(self.params_initialize)
            self.global_step = torch.tensor(0,dtype=torch.int64,device=self.ctx)
        self.debug_info(self.net)
        self.summary(self.net)
        self.add_graph(self.net)
