#使用tensorflow2.0, 采用keras框架
import tensorflow as tf
from tensorflow import keras
from interpreterAnalysize.interpreterBaseClass import *


class CheckpointModelT(CheckpointModelBase):
    def save_model(self, net: keras.Sequential, optimizer):
        self.checkpoint = tf.train.Checkpoint(model=net, optimizer=optimizer)  # varible=self.global_step)
        self.model_savefile = utile.construct_filename(self.directory, self.prefix_modelfile, self.suffix_modelfile)
        self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.directory,
                                                  checkpoint_name=self.model_savefile,
                                                  max_to_keep=self.max_keep_models)
        self.manager.save()
        self.logger.info('Success to save model to file %s' % self.model_savefile)

    def load_model(self, net: keras.Sequential, optimizer , ctx):
        self.checkpoint = tf.train.Checkpoint(model=net, optimizer=optimizer)  # varible=self.global_step)
        #self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.workingspace.directory,
        #                                          checkpoint_name=self.checkpoint_filename,
        #                                          max_to_keep=self.max_keep_models)

        # if 'pretrained_model' in self.gConfig:
        #    self.saver.restore(self.session, self.gConfig['pretrained_model'])
        # ckpt = tf.train.get_checkpoint_state(self.gConfig['working_directory'])
        ckpt = tf.train.latest_checkpoint(self.directory)
        # if ckpt and ckpt.model_checkpoint_path and ckpt_used:
        if ckpt:
            self.logger.info("success to load model from %s" % ckpt)
            # self.net.load_models(self.model_savefile)
            # tf.train.Checkpoint(x=self.global_step).restore(tf.train.latest_checkpoint(self.working_directory))
            # self.manager.restore(ckpt)
            self.checkpoint.restore(ckpt)
            # self.net.load_weights(ckpt)
            #self.global_step = self.optimizer.iterations
        else:
            self.logger.error(
                "failed to load the mode file(%s),it is not exists, you must train it first!" % ckpt)
            raise ValueError(
                "failed to load the mode file(%s),it is not exists, you must train it first!" % ckpt)
        return net, optimizer

    def is_modelfile_exist(self):
        isModelfileExist = False
        ckpt = tf.train.latest_checkpoint(self.directory)
        if ckpt:
               isModelfileExist = True
        return isModelfileExist

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
class modelBaseT(InterpreterBase):
    def __init__(self,gConfig):
        super(modelBaseT, self).__init__(gConfig)
        self.net = keras.Sequential()

    def _init_parameters(self):
        super(modelBaseT, self)._init_parameters()
        self.learning_rate_value = self.gConfig['learning_rate']
        self.learning_rate_decay_factor = self.gConfig['learning_rate_decay_factor']
        self.decay_steps = self.gConfig['decay_steps']
        self.init_sigma = self.gConfig['init_sigma']
        self.init_bias = self.gConfig['init_bias']
        self.momentum = self.gConfig['momentum']
        self.max_keep_models = self.gConfig['max_keep_models']
        self.max_queue = self.gConfig['max_queue']
        self.initializer = self.gConfig['initializer']
        self.optimizer = self.gConfig['optimizer']
        #self.optimizer = self.get_optimizer(self.gConfig['optimizer'])
        self.tfdbgIsOn = self.gConfig['tfdbgIsOn'.lower()]
        self.keeps = 1 - self.gConfig['dropout']
        self.init_framework()
        #self.log_savefile='.'.join([self.get_model_name(gConfig),'log'])
        self.global_step = tf.Variable(0, trainable=False, name='global_step',dtype=tf.int64)
        with tf.name_scope('learning_rate'):
            self.learning_rate = tf.Variable(float(self.learning_rate_value), trainable=False, name='learning_rate')
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate *
                                                                    self.learning_rate_decay_factor,
                                                                    name='learning_rate_decay')
            #self.learning_rate = tf.train.exponential_decay(self.learning_rate_value,self.global_step,self.decay_steps,
            #                                                self.learning_rate_decay_factor,staircase=True)

        self.writer = tf.summary.create_file_writer(self.loggingspace.directory)
        with self.writer.as_default():
            tf.summary.trace_on()
        self.checkpoint = self.create_space(CheckpointModelT
                                            , max_keep_models = self.gConfig['max_keep_models']
                                            , prefix_modelfile=self._get_module_name())
        #self.net = keras.Sequential()

    def get_net(self):
        return

    def init_framework(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                print(e)

    def get_initializer(self, initializer):
        assert initializer in self.gConfig['initializerlist'],'initializer(%s) is invalid,it must one of %s' % \
                                                    (initializer, self.gConfig['initializerlist'])
        if initializer == 'normal':
            return keras.initializers.TruncatedNormal(stddev=self.init_sigma)
        elif initializer == 'xavier':
            return keras.initializers.GlorotNormal()
        elif initializer == 'kaiming':
            #何凯明初始化法
            return keras.initializers.he_uniform()
        elif initializer == 'constant':
            return keras.initializers.Constant(self.init_bias)
        else:
            return None

    def get_optimizer(self,optimizer):
        assert optimizer in self.gConfig['optimizerlist'], 'optimizer(%s) is invalid,it must one of %s' % \
                                                               (optimizer, self.gConfig['optimizerlist'])
        if optimizer == 'sgd':
            return keras.optimizers.SGD(self.learning_rate)
        elif optimizer == 'adadelta':
            return keras.optimizers.Adadelta(self.learning_rate)
        elif optimizer == 'rmsprop':
            return keras.optimizers.RMSprop(self.learning_rate)
        elif optimizer == 'adam':
            return keras.optimizers.Adam(self.learning_rate)
        elif optimizer == 'adagrad':
            return keras.optimizers.Adagrad(self.learning_rate)
        elif optimizer == 'momentum':
            return keras.optimizers.SGD(learning_rate=self.learning_rate,momentum=self.momentum)
        return None

    def get_padding(self):
        #把padding数字转化为tensorflow.nn.conv2d的函数入参
        def get_padding_iner(padding):
            if padding == 0:
                return 'VALID'
            else:
                return 'SAME'
        return get_padding_iner

    def get_activation(self, activation='relu'):
        assert activation in self.gConfig['activationlist'], 'activation(%s) is invalid,it must one of %s' % \
                                                    (activation, self.gConfig['activationlist'])
        if activation == 'sigmoid':
            #return tf.nn.sigmoid
            return keras.layers.Activation('sigmoid')
        elif activation == 'relu':
            #return tf.nn.relu
            return keras.layers.Activation('relu')

    def get_context(self):
        return self.global_step.device
        '''
        if tf.test.is_gpu_available():
            devices = tf.test.gpu_device_name()
        else:
            devices = '/device:CPU'
        return devices'''

    def get_learningrate(self):
        return self.learning_rate.numpy()


    def get_globalstep(self):
        #return tf.train.get_or_create_global_step()
        return self.global_step

    '''
    def saveCheckpoint(self):
        #checkpoint_path = os.path.join(self.working_directory, self.checkpoint_filename)
        #self.net.save(self.model_savefile)
        #tf.train.Checkpoint(x=self.global_step).save(self.working_directory)
        self.manager.save()
    '''

    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        for epoch in range(num_epochs):
            self.run_epoch(getdataClass, epoch)

        with self.writer.as_default():
            tf.summary.trace_export(name='graph',step=self.optimizer.iterations,profiler_outdir=self.workingspace.directory)
        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test

    def debug_info(self,*args):
        if self.debugIsOn == False:
            return
        if len(args) == 0:
            return
        if len(args) == 1:
            print('debug:%s'%args[0])
            return
        (net,grads) = args
        self.debug(net,grads)
        print('\n')
        return

    def debug(self,net,grads):
        for trainable_var,grad in zip(net.trainable_variables,grads):
            print('\tdebug(%d):%s:%s'%(self.global_step,net.name,trainable_var.name),
                  '\tshape=',trainable_var.shape,
                  '\tdata.mean=%.6f'%trainable_var.numpy().mean(),
                  '\tgrad.mean=%.6f' % grad.numpy().mean(),
                  '\tdata.std=%.6f'%trainable_var.numpy().std(),
                  '\tgrad.std=%.6f'%grad.numpy().std())

    def run_train_loss_acc(self,X,y,keeps):
        loss,acc = None,None
        return loss,acc

    def run_eval_loss_acc(self, X, y, keeps=1.0):
        loss,acc = None,None
        return loss,acc

    def run_learning_rate_decay(self,acc_train):
        if len(self.acces_train) > 2 and acc_train < min(self.acces_train[-3:]):
            #self.session.run(self.learning_rate_decay_op)
            self.learning_rate = self.learning_rate * self.learning_rate_decay_factor
            print("\n execute learning rate decay(learning_rate=%f)\n" % self.learning_rate.eval())

    def image_record(self,global_step,tag,input_image):
        if global_step < self.gConfig['num_samples']:
            with self.writer.as_default():
                tf.summary.image(name=tag,data=input_image,step=global_step)

    def train_loss_acc(self,data_iter):
        acc_sum = 0
        loss_sum =0
        num = 0
        for X, y in data_iter:
            try:
                X = X.asnumpy()
                X = np.transpose(X, axes=[0, 2, 3, 1])
                y = y.asnumpy()
            except:
                #X = np.array(X)
                #y = np.array(y)
                pass
            self.image_record(self.global_step, 'input/image', X)
            loss, acc = self.run_train_loss_acc(X, y,self.keeps)
            with self.writer.as_default():
                tf.summary.scalar(name='train/loss',data=loss,step=self.optimizer.iterations)
                tf.summary.scalar(name='train/accuracy',data=acc,step=self.optimizer.iterations)
            batch_size = y.shape[0]
            loss_sum += loss * batch_size
            acc_sum += acc * batch_size
            num += batch_size
            self.global_step = self.global_step + 1
        return loss_sum / num, acc_sum / num

    def evaluate_loss_acc(self, data_iter):
        acc_sum = 0
        loss_sum =0
        num = 0
        for X, y in data_iter:
            try:
                #由mxnet框架读入的数据通过该方法转换
                X = X.asnumpy()
                X = np.transpose(X, axes=[0, 2, 3, 1])
                y = y.asnumpy()
            except:
                #由其他框架读入的数据通过numpy转换
                #X = np.array(X)
                #y = np.array(y)
                pass
            loss,acc = self.run_eval_loss_acc(X, y)
            batch_size = y.shape[0]
            loss_sum += loss * batch_size
            acc_sum += acc * batch_size
            num += batch_size
        return loss_sum / num, acc_sum / num

    def run_step(self,epoch,train_iter,valid_iter,test_iter, epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=0,0,None,None,0,0
        loss_train,acc_train = self.train_loss_acc(train_iter)
        if epoch % epoch_per_print == 0:
            self.run_learning_rate_decay(acc_train)
            loss_test,acc_test = self.evaluate_loss_acc(test_iter)
            with self.writer.as_default():
                tf.summary.scalar(name='test/loss',data=loss_test,step=self.optimizer.iterations)
                tf.summary.scalar(name='test/accuracy',data=acc_test,step=self.optimizer.iterations)
        if epoch % self.epochs_per_checkpoint == 0:
            self.checkpoint.save_model(self.net, self.optimizer)
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test

    def get_input_shape(self):
        pass

    def initialize(self,dictParameter = None):
        assert dictParameter is not None,'dictParameter must not be None!'

        self.gConfig.update(dictParameter)
        self._init_parameters()
        self.loggingspace.clear_directory(self.loggingspace.directory)

        #self.checkpoint = tf.train.Checkpoint(model=self.net,optimizer = self.optimizer)#varible=self.global_step)
        #self.manager = tf.train.CheckpointManager(self.checkpoint, directory=self.workingspace.directory,
        #                                          checkpoint_name=self.checkpoint_filename, max_to_keep=self.max_keep_models)

        #if 'pretrained_model' in self.gConfig:
        #    self.saver.restore(self.session, self.gConfig['pretrained_model'])
        #ckpt = tf.train.get_checkpoint_state(self.gConfig['working_directory'])
        #ckpt = tf.train.latest_checkpoint(self.workingspace.directory)
        #if ckpt and ckpt.model_checkpoint_path and ckpt_used:
        assert self.gConfig['mode'] in self.gConfig['modelist'] \
            , "mode(%s) must be in modelist: %s'(self.gConfig['mode'],self.gConfig['modelist']"
        if self.gConfig['mode'] == 'apply':
            self.net, self.optimizer = self.checkpoint.load_model(self.net,self.optimizer)
        elif self.gConfig['mode'] == 'pretrain':
            pass
        else:
            ckpt_used = self.gConfig['ckpt_used']
            if self.checkpoint.is_modelfile_exist() and ckpt_used:
                self.net, self.optimizer = self.checkpoint.load_model(self.net,self.optimizer)
                self.global_step = self.optimizer.iterations
            else:
                print("Created model with fresh parameters.")
                print("Created model with fresh parameters.")
                self.global_step = tf.constant([0], dtype=tf.int64)
                self.net.build(input_shape=self.get_input_shape())

        #if ckpt and ckpt_used:
        #    print("Reading model parameters from %s" % ckpt)
            #self.net.load_models(self.model_savefile)
            #tf.train.Checkpoint(x=self.global_step).restore(tf.train.latest_checkpoint(self.working_directory))
            #self.manager.restore(ckpt)
        #    self.checkpoint.restore(ckpt)
            #self.net.load_weights(ckpt)
        #    self.global_step = self.optimizer.iterations
        #    self.net.build(input_shape=self.get_input_shape())
            #self.net = keras.models.load_model(ckpt)
        #else:
        #    print("Created model with fresh parameters.")
        #    self.global_step = tf.constant([0],dtype=tf.int64)
        #    self.net.build(input_shape=self.get_input_shape())
        self.net.summary()


