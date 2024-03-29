import time

import utile
from interpreterAnalysize.interpreterBaseClass import *
from torch import optim,nn
from torch.nn.utils.rnn import PackedSequence
import torch
#from torchvision import models
from tensorboardX import SummaryWriter
from torchsummary import summary # 该1.8版本在pytorch 1.2.0版本下使用, add_graph函数存在异常
import re
from functools import partial
from typing import Callable, Any

class PreprocessH():
    """
    用于装饰器, 用在:
    1) CriteriaBaseH类的__call__方法,在计算前先对 y_hat, y做转换, 针对y_hat为tuple情况(意味着多输出),只取最后一纬
    args:
        function - 被装饰的函数
    """
    def __init__(self, function):
        self.function = function

    def __get__(self,instance,own):
        """
        被装饰的函数在调用时,首先会调用__get__函数
        args:
            instance - 为被装饰函数所属的对象
            own - 被装饰函数所属的类
        reutrn:
            wrap - 装饰器的内层函数
        """
        def wrap(y_hat,y):
            if CriteriaBaseH.__subclasscheck__(own) or ModelBaseH.__subclasscheck__(own):
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[-1]
                    y = y[:,-1]
                y = y.long()
            return self.function(instance,y_hat, y)
        return wrap


class LossBaseH():
    def __init__(self,ctx):
        self.ctx = ctx
        self.loss = nn.CrossEntropyLoss().to(self.ctx)

    #__call__ : Callable[..., Any] = forward
    def __call__(self, y_hat, y):
        return self.forward(y_hat, y)

    def forward(self,y_hat,y):
        return self.loss(y_hat,y)


class CriteriaBaseH():
    def __init__(self):
        ...

    #@PreprocessH
    def __call__(self, y_hat, y):
        return self.forward(y_hat,y)

    def forward(self,y_hat,y):
        criteria = (y_hat.argmax(dim=1) == y).sum().item()
        return criteria


class CheckpointModelH(CheckpointModelBase):
    def save_model(self, net: nn.Module, optimizer: optim.Optimizer):
        self.model_savefile = utile.construct_filename(self.directory, self.prefix_modelfile, self.suffix_modelfile)
        stateSaved = {'model':net.state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(stateSaved,self.model_savefile)
        self.logger.info('Success to save model to file %s' % self.model_savefile)

    def load_model(self, net: nn.Module, get_optimizer : Callable, ctx):
        if self.is_modelfile_exist():
            stateLoaded = torch.load(self.model_savefile, map_location=ctx)
            net.load_state_dict(stateLoaded['model'])
            net.to(ctx)
            optimizer = get_optimizer(net.parameters())
            optimizer.load_state_dict(stateLoaded['optimizer'])
            self.logger.info("Success to load model from file %s" % self.model_savefile)
        else:
            self.logger.error(
                "failed to load the mode file(%s),it is not exists, you must train it first!" % self.model_savefile)
            raise ValueError(
                "failed to load the mode file(%s),it is not exists, you must train it first!" % self.model_savefile)
        return net, optimizer

    @classmethod
    def processing_checkpoint(cls,func):
        @functools.wraps(func)
        def wrap(self, *args):
            start_time = time.time()
            result = func(self, *args)
            self.checkpoint.save_model(self.net, self.optimizer)
            total_time = time.time() - start_time
            taskResult = []
            taskResult.append(os.path.split(self.checkpoint.model_savefile)[-1])
            taskResult.append(utile.get_time_now())
            taskResult.append(f"{total_time:10.2f}")
            taskResult.append(f"{self.acces_train[-1]:10.4f}")
            taskResult.append(f"{self.losses_train[-1]:10.4f}")
            taskResult.append(f"{self.acces_test[-1]:10.4f}")
            taskResult.append(f"{self.losses_test[-1]:10.4f}")
            taskResult.append(f"{self.get_learningrate():10.6f}")
            taskResult.append(f"{self.get_global_step():10d}")
            taskResult.append(str(self.get_context()))
            taskResult.append(NULLSTR)
            content = ','.join(taskResult)
            self.checkpoint.save(content)
            return result
        return wrap

#深度学习模型的基类
class ModelBaseH(InterpreterBase):
    def __init__(self,gConfig):
        super(ModelBaseH, self).__init__(gConfig)
        self.net = nn.Module()


    def _init_parameters(self):
        super(ModelBaseH, self)._init_parameters()
        self.learning_rate = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.learning_rate_decay_step = self.gConfig['learning_rate_decay_step']
        self.viewIsOn = self.gConfig['viewIsOn']
        self.ctx = self.get_ctx(self.gConfig['ctx'])
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.momentum = self.gConfig['momentum']
        self.initializer = self.gConfig['initializer']
        self.max_queue = self.gConfig['max_queue']
        self.weight_initializer = self.get_initializer(self.initializer)
        self.bias_initializer = self.get_initializer('constant')
        self.checkpoint = self.create_space(CheckpointModelH
                                            , max_keep_models = self.gConfig['max_keep_models']
                                            , prefix_modelfile=self._get_module_name())
        #self.set_default_tensor_type()  #设置默认的tensor在ｇｐｕ还是在ｃｐｕ上运算


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


    def get_global_step(self):
        return 0


    def get_input_shape(self):
        return tuple()


    def init_state(self):
        pass


    @CheckpointModelH.processing_checkpoint
    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass,epoch)
        #self.checkpoint.save_model(self.net, self.optimizer)
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
        acc_sum, loss_sum, n = 0, 0, 0
        self.init_state()  # 仅用于RNN,LSTM等
        self.net.train()
        for X, y in data_iter:
            try:
                #X = X.asnumpy()
                #y = y.asnumpy()
                #if not isinstance(X, PackedSequence):
                #X = torch.tensor(X, device=self.ctx)
                # y = torch.tensor(y, device=self.ctx, dtype=torch.long)
                #y = torch.tensor(y, device=self.ctx)
                X = X.to(self.ctx)
                y = y.to(self.ctx)
            except:
                X = torch.tensor(X, device=self.ctx)
                y = torch.tensor(y, device=self.ctx)
                #if not isinstance(X,PackedSequence):
                    #X = np.array(X)
                #    X = X.numpy()
                #y = np.array(y)
                #y = y.numpy()
            self.image_record(self.get_global_step(), 'input/image', X[0])
            #if not isinstance(X,PackedSequence):
            #    X = torch.tensor(X, device=self.ctx)
            #y = torch.tensor(y, device=self.ctx, dtype=torch.long)
            #y = torch.tensor(y, device=self.ctx)
            loss, acc = self.run_train_loss_acc(X, y)
            loss_sum += loss
            acc_sum += acc
            #n += y.size()[0]
            n += self.get_batch_size(y)
            self.writer.add_scalar('train/loss', loss, self.get_global_step())
            self.writer.add_scalar('train/accuracy', acc, self.get_global_step())
        if n != 0:
            loss_sum /= n
            acc_sum /= n
        return loss_sum, acc_sum


    def evaluate_loss_acc(self, data_iter):
        acc_sum, loss_sum, n = 0, 0, 0
        self.net.eval()
        for X, y in data_iter:
            try:
                #X = X.asnumpy()
                #y = y.asnumpy()
                #if not isinstance(X, PackedSequence):
                #X = torch.tensor(X, device=self.ctx)
                # y = torch.tensor(y,device=self.ctx,dtype=torch.long)
                #y = torch.tensor(y, device=self.ctx)
                X = X.to(self.ctx)
                y = y.to(self.ctx)
            except:
                X = torch.tensor(X, device=self.ctx)
                y = torch.tensor(y, device=self.ctx)
                #if not isinstance(X,PackedSequence):
                    #X = np.array(X)
                #    X = X.numpy()
                #y = np.array(y)
                #y = y.numpy()
            #if not isinstance(X,PackedSequence):
            #    X = torch.tensor(X,device=self.ctx)
            #y = torch.tensor(y,device=self.ctx,dtype=torch.long)
            #y = torch.tensor(y, device=self.ctx)
            loss,acc = self.run_eval_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += self.get_batch_size(y)
        if n != 0:
            loss_sum /= n
            acc_sum /= n
        return loss_sum, acc_sum


    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        loss_train,acc_train = self.train_loss_acc(train_iter)
        if epoch % epoch_per_print == 0:
            loss_test,acc_test = self.evaluate_loss_acc(test_iter)
            self.writer.add_scalar('test/loss', loss_test, self.get_global_step())
            self.writer.add_scalar('test/accuracy', acc_test, self.get_global_step())
            self.run_matrix(loss_train, loss_test)  # 仅用于rnn,lstm,ssd等
            self.predict(self.net)  # 仅用于rnn,lstm,ssd等
        if epoch % self.epochs_per_checkpoint == 0:
            self.checkpoint.save_model(self.net, self.optimizer)
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test


    def predict(self,net):
        pass


    def apply_model(self, net):
        getdataClass = self.gConfig['getdataClass']
        keyfields_iter,valid_iter = getdataClass.getValidData(self.batch_size)
        mergedFields = []
        acc_sum, loss_sum, n = 0, 0, 0
        net.eval()
        for (X,y),keyfields in zip(keyfields_iter,valid_iter):
            try:
                #X = X.asnumpy()
                #y = y.asnumpy()
                #if not isinstance(X, PackedSequence):
                X = X.to(self.ctx)
                y = y.to(self.ctx)
            except:
                X = torch.tensor(X, device=self.ctx)
                y = torch.tensor(y, device=self.ctx)
            #    if not isinstance(X,PackedSequence):
                    #X = np.array(X)
            #        X = X.numpy()
                #y = np.array(y)
            #    y = y.numpy()
            #if not isinstance(X,PackedSequence):
            #    X = torch.tensor(X,device=self.ctx)
            #y = torch.tensor(y, device=self.ctx)
            loss, acc, dataFrame = self.predict_with_keyfileds(net,X,y,keyfields)
            mergedFields += [dataFrame]
            acc_sum += acc
            loss_sum += loss
            n += self.get_batch_size(y)
        mergedDataFrame = pd.concat(mergedFields, axis=0)
        mergedDataFrame = mergedDataFrame.dropna(axis=0).reset_index(drop=True)
        self.process_write_to_sqlite3(mergedDataFrame)
        if n != 0:
            loss_sum /= n
            acc_sum /= n
        return loss_sum, acc_sum


    def process_write_to_sqlite3(self, mergedDataFrame):
        for reportType in self.gConfig['报告类型']:
            tablePrefix = self.standard._get_tableprefix_by_report_type(reportType)
            tableName = tablePrefix + self.gConfig['tableName']
            self.database._write_to_sqlite3(mergedDataFrame, self.commonFields,tableName)
            self.logger.info('success to apply model(%s) and write to predicted data to sqlite3: %s'
                             %(self.gConfig['model'], tableName))


    def predict_with_keyfileds(self,net,keyfields,X,y):
        return (None, None, None)


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

    '''
    def set_default_tensor_type(self):
        if self.ctx == torch.device(type='cuda',index=0):
            # 如果设置为cuda的类型，则所有操作都在GPU上进行
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
    '''


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

        self.loggingspace.clear_directory(self.loggingspace.directory)

        self.writer = SummaryWriter(logdir=self.loggingspace.directory,max_queue=self.max_queue)
        #self.vis = visdom.Visdom(env='test')
        assert self.gConfig['mode'] in self.gConfig['modelist']\
            ,"mode(%s) must be in modelist: %s'(self.gConfig['mode'],self.gConfig['modelist']"
        if self.gConfig['mode'] == 'apply':
            self.net, self.optimizer = self.checkpoint.load_model(self.net
                                                                  , partial(self.get_optimizer,self.gConfig['optimizer'])
                                                                  , self.ctx)
        elif self.gConfig['mode'] == 'pretrain':
            pass
        else:
            ckpt_used = self.gConfig['ckpt_used']
            if self.checkpoint.is_modelfile_exist() and ckpt_used:
                self.net, self.optimizer = self.checkpoint.load_model(self.net
                                                                      , partial(self.get_optimizer,self.gConfig['optimizer'])
                                                                      , self.ctx)
            else:
                print("Created model with fresh parameters.")
                self.net.to(device=self.ctx)
                self.net.apply(self.params_initialize)
        self.debug_info(self.net)
        self.summary(self.net)
        self.add_graph(self.net)
