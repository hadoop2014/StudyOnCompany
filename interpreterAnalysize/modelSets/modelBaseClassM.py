from mxnet.gluon import nn
from mxnet import nd, gluon,autograd,init
import mxnet as mx
import numpy as np
#from modelBaseClass import  *
from interpreterAnalysize.interpreterBaseClass import *
import datafetch.commFunction as commFunc
import mxnet.gluon.model_zoo.vision

#深度学习模型的基类
class ModelBaseM(InterpreterBase):
    def __init__(self,gConfig):
        super(ModelBaseM, self).__init__(gConfig)
        self.net = nn.HybridSequential()


    def _init_parameters(self):
        super(ModelBaseM, self)._init_parameters()
        self.learning_rate = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.lr_mult = self.gConfig['lr_mult']  # learning_rate的乘数，加速学习，用于预训练模型
        self.viewIsOn = self.gConfig['viewIsOn']
        self.max_to_keep = self.gConfig['max_to_keep']
        self.ctx = self.get_ctx(self.gConfig['ctx'])
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'])
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.weight_initializer = self.get_initializer(self.gConfig['initializer'])
        self.bias_initializer = self.get_initializer('constant')
        self.global_step = nd.array([0], self.ctx)
        self.state = None  # 用于rnn,lstm等
        self.ssd_image_size = 1  # 仅用于ssd 的pretrain模式,默认情况下设置为１


    def _get_class_name(self, gConfig):
        model_name = re.findall('(.*)Model', self.__class__.__name__).pop().lower()
        assert model_name in gConfig['tasknamelist'], \
            'tasknamelist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['tasknamelist'], model_name, self.__class__.__name__)
        return model_name


    def get_net(self):
        return


    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu(0)
        return ctx


    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return init.Normal(sigma=self.init_sigma)
        elif initializer == 'xavier':
            return init.Xavier()
        elif initializer == 'kaiming':
            #何凯明初始化法
            return init.Xavier(rnd_type='uniform',factor_type='in',magnitude=np.sqrt(2))
        elif initializer == 'constant':
            return init.Constant(self.init_bias)
        elif initializer == 'uniform':
            return init.Uniform()
        else:
            return None


    def get_optimizer(self,optimizer):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        return optimizer


    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        return activation


    def get_context(self):
        return self.ctx


    def get_learningrate(self):
        return self.learning_rate


    def get_globalstep(self):
        return self.global_step.asscalar()


    def get_input_shape(self):
        pass


    def init_state(self):
        pass


    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        #title = self.gConfig['taskname']
        title = self.gConfig['model']
        input_symbol = mx.symbol.Variable('input_data')
        net = self.net(input_symbol)
        mx.viz.plot_network(net, title=title, save_format='png', hide_weights=False,
                            shape=input_shape) \
                .view(directory=self.logging_directory)
        return


    def saveCheckpoint(self):
        self.net.save_parameters(self.model_savefile)
        nd.save(self.symbol_savefile,self.global_step)


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
        self.saveCheckpoint()
        self.predict_cv(self.net)
        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test


    def debug_info(self, info=None):
        if self.debugIsOn == False:
            return
        if info is not None:
            print('debug:%s' % info)
            return
        self.debug(self.net)
        print('\n')
        return


    def debug(self, layer, name=''):
        if len(layer._children) != 0:
            for block in layer._children.values():
                self.debug(block, layer.name)
        for param in layer.params:
            parameter = layer.params[param]
            if parameter.grad_req != 'null':
                print('\tdebug:%s(%s)' % (name, param),
                      '\tshape=', parameter.shape,
                      '\tdata.mean=%f' % parameter.data().mean().asscalar(),
                      '\tgrad.mean=%f' % parameter.grad().mean().asscalar(),
                      '\tdata.std=%.6f' % parameter.data().asnumpy().std(),
                      '\tgrad.std=%.6f' % parameter.grad().asnumpy().std())


    def predict_nlp(self, model):
        #仅用于rnn网络的句子预测
        pass


    def predict_cv(self,model):
        #用于图像相关的任务的预测
        pass


    def image_record(self,global_step,tag,input_image):
        input_image = nd.transpose(input_image,axes=[0,2,3,1])
        if global_step < self.gConfig['num_samples']:
            hotdogs = [input_image[i] for i in range(8)]
            not_hotdogs = [input_image[-i - 1] for i in range(8)]
            commFunc.show_images(hotdogs+not_hotdogs, 2, 8, scale=1.4)


    def run_train_loss_acc(self,X,y):
        loss,acc = None,None
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        loss,acc = None,None
        return loss,acc


    def run_matrix(self, loss_train, loss_test):
        pass


    def train_loss_acc(self,data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.init_state()  # 仅用于RNN,LSTM等
        for X, y in data_iter:
            X = nd.array(X, ctx=self.ctx)
            y = nd.array(y, ctx=self.ctx)
            y = y.astype('float32')
            #self.image_record(self.global_step.asscalar(), 'input/image', X) #仅调试时使用，正常运行时不要使用
            loss, acc = self.run_train_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += y.shape[0]
            self.global_step += nd.array([1],ctx=self.ctx)
        return loss_sum / n, acc_sum / n


    def evaluate_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum = 0
        n = 0
        self.init_state()  #仅用于RNN,LSTM等
        for X, y in data_iter:
            X = nd.array(X,ctx=self.ctx)
            y = nd.array(y,ctx=self.ctx)
            y = y.astype('float32')
            loss,acc = self.run_eval_loss_acc(X, y)
            acc_sum += acc
            loss_sum += loss
            n += y.shape[0]
        return loss_sum / n, acc_sum / n


    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        loss_train,acc_train = self.train_loss_acc(train_iter)
        if epoch % epoch_per_print == 0:
            loss_test, acc_test = self.evaluate_loss_acc(test_iter)
            self.run_matrix(loss_train, loss_test)   #仅用于rnn,lstm,ssd等
            self.predict_nlp(self.net)    #仅用于rnn,lstm,ssd等
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test


    def summary(self):
        self.net.summary(nd.zeros(shape=self.get_input_shape(),ctx=self.ctx))


    def hybridize(self):
        self.net.hybridize()


    def get_pretrain_model(self,**kwargs):
        moduleName = self.check_book[self.gConfig['taskName']][self.gConfig['framework']]['pretrain']['model']
        moduleNameList = moduleName.split('.')
        assert len(moduleNameList) >=3 ,'the len of moduleNameList must be lager then 3!'
        index = moduleNameList.index('model_zoo')  #处理gluoncv.model_zoo.ssd_512_mobilenet1.0_voc的情况
        assert index == 1 , 'module name must be xxxx.model_zoo!'
        firstName = moduleNameList[0]
        moduleName = '.'.join(moduleNameList[:(index + 1)])
        className = '.'.join(moduleNameList[(index + 1):])
        module = __import__(moduleName,fromlist=(moduleName.split('.')[-1]))
        if firstName == 'gluoncv':
            model = getattr(module,'get_model')(name=className,**kwargs)
            try:
                self.ssd_image_size=int(re.findall('ssd_([0-9]*)_',className).pop())
            except:
                print('image_size is only used in ssd task of pretrain mode,other task(%s) has just ignored!',
                      self.gConfig['taskname'])
        else:
            model = getattr(module,className)(**kwargs)
        return model


    def get_classes(self):
        pass


    def transfer_learning(self):
        pretrained_net = self.get_pretrain_model(pretrained=True, ctx=self.ctx,
                                                 root=self.working_directory, classes=self.get_classes())
        net = self.get_pretrain_model(classes=self.get_classes())
        net.features = pretrained_net.features
        #net.features.collect_params().setattr('grad_req', 'null') #所有的features的梯度不再更新
        net.output.initialize(self.weight_initializer, ctx=self.ctx)
        net.output.collect_params().setattr('lr_mult', self.lr_mult)
        # weight = pretrained_net.output.weight
        # hotdog_w = nd.split(weight.data(), 1000, axis=0)[713]  ＃713即为imagenet中hotdog的分类
        # self.net.output.weight.data()[1]= hotdog_w
        return net


    def initialize(self,dictParameter = None):
        assert dictParameter is not None,'dictParameter must not be None!'

        self.gConfig.update(dictParameter)
        self._init_parameters()

        if os.path.exists(self.logging_directory) == False:
            os.makedirs(self.logging_directory)
        if os.path.exists(self.working_directory) == False:
            os.makedirs(self.working_directory)

        ckpt = self.getSaveFile()
        ckpt_used = self.gConfig['ckpt_used']
        if self.gConfig['mode'] == 'pretrain' :
            self.net = self.transfer_learning()
            self.trainer = gluon.Trainer(self.net.collect_params(), self.optimizer,
                                         {'learning_rate': self.learning_rate})
            self.global_step = nd.array([0], ctx=self.ctx)
            self.debug_info(self.net)
            self.summary()
        elif ckpt and ckpt_used:
            print("Reading model parameters from %s" % ckpt)
            self.net.load_parameters(ckpt, ctx=self.ctx)
            self.global_step = nd.load(self.symbol_savefile)[0]
        else:
            print("Created model with fresh parameters.")
            self.net.initialize(self.weight_initializer, force_reinit=True, ctx=self.ctx)
            self.global_step = nd.array([0], ctx=self.ctx)
            self.debug_info(self.weight_initializer.dumps())
            self.debug_info(self.net)
            self.summary()
        self.show_net(input_shape={'input_data': self.get_input_shape()})
        self.hybridize()
