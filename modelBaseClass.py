# -*- coding: utf-8 -*-
# @Time    : 9/25/2019 5:03 PM
# @Author  : wuhao
# @File    : modelBaseClass.py
import json
from baseClass import *

class CheckpointModelBase(CheckpointBase):
    '''
    explian: 用于存放模型的基础类
    args:
         modelfile_basic - 模型的基本文件名, 在派生类CheckpointModelVisual中使用, 把模型结果数据拷贝到该文件中
    '''
    modelfile_basic = NULLSTR
    def __init__(self, working_directory, logger, checkpointfile, checkpointIsOn, max_keep_checkpoint, max_keep_models
                 ,copy_file = False,prefix_modelfile = NULLSTR, suffix_modelfile = 'model', modelfile = NULLSTR):
        super(CheckpointModelBase, self).__init__(working_directory, logger, checkpointfile, checkpointIsOn
                                                  , max_keep_checkpoint)
        self.prefix_modelfile = prefix_modelfile
        self.suffix_modelfile = suffix_modelfile
        CheckpointModelBase.modelfile_basic = os.path.join(self.directory, modelfile)
        if modelfile != NULLSTR and isinstance(modelfile,str):
            self.prefix_modelfile = modelfile.split('.')[0]
            self.suffix_modelfile = modelfile.split('.')[-1]
        self.max_keep_models = max_keep_models
        self.model_savefile = self._check_max_checkpoints(self.directory
                                                          , self.prefix_modelfile
                                                          , self.suffix_modelfile
                                                          , self.max_keep_models
                                                          , copy_file=copy_file)
        self._init_modelfile(self.model_savefile)

    def _init_modelfile(self, modelfile):
        ...

    @property
    def model_filename(self):
        return self.model_savefile

    def is_modelfile_exist(self):
        isModelfileExist = False
        if self.model_savefile is not None:
            if os.path.exists(self.model_savefile):
               isModelfileExist = True
        return isModelfileExist

#深度学习模型的基类
class ModelBase(BaseClass):
    def __init__(self,gConfig):
        super(ModelBase, self).__init__(gConfig)
        self.start_time = time.time()
        #self.debugIsOn = self.gConfig['debugIsOn'.lower()]
        self.check_book = self.get_check_book()
        self.symbol_savefile = os.path.join(self.workingspace.directory,
                                            self._get_module_name() + '.symbol')# + self.gConfig['framework'])
        self.losses_train = []
        self.acces_train = []
        self.losses_valid = []
        self.acces_valid = []
        self.losses_test = []
        self.acces_test = []


    def _init_parameters(self):
        self.epoch_per_print = self.gConfig['epoch_per_print']
        self.debug_per_steps = self.gConfig['debug_per_steps']
        self.epochs_per_checkpoint = self.gConfig['epochs_per_checkpoint']
        self.batch_size = self.gConfig['batch_size']
        self.debugIsOn = self.gConfig['debugIsOn']


    def get_check_book(self):
        check_file = os.path.join(self.gConfig['config_directory'], self.gConfig['check_file'])
        #check_book = None
        if os.path.exists(check_file):
            with open(check_file, encoding='utf-8') as check_f:
                check_book = json.load(check_f)
        else:
            raise ValueError("%s is not exist,you must create first!" % check_file)
        return check_book


    def get_net(self):
        pass


    def get_context(self):
        pass


    def get_learningrate(self):
        pass


    def get_global_step(self):
        pass


    def run_step(self,epoch,train_iter,valid_iter,test_iter,epoch_per_print):
        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test=None,None,None,None,None,None
        return loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test


    def run_epoch(self,getdataClass, epoch, output_log=True):
        train_iter = getdataClass.getTrainData(self.batch_size)
        test_iter = getdataClass.getTestData(self.batch_size)
        valid_iter = getdataClass.getValidData(self.batch_size)

        loss_train, acc_train,loss_valid,acc_valid,loss_test,acc_test = \
            self.run_step(epoch,train_iter,valid_iter,test_iter,self.epoch_per_print)

        if epoch % self.epoch_per_print == 0:
            self.losses_train.append(loss_train)
            self.acces_train.append(acc_train)

            self.losses_valid.append(loss_valid)
            self.acces_valid.append(acc_valid)

            self.losses_test.append(loss_test)
            self.acces_test.append((acc_test))

            check_time = time.time()
            if loss_valid is None:
                print("\nepoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.6f" % loss_train,
                      "\t acc_test = %.4f" % acc_test, "\t loss_test = %.6f" % loss_test,
                      #"\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.4f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_global_step(),
                      "  context:%s" % self.get_context())
            elif loss_test is None:
                print("\nepoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.6f" % loss_train,
                      #"\t acc_test = %.4f" % acc_test, "\t loss_test = %.4f" % loss_test,
                      "\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.6f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_global_step(),
                      "  context:%s" % self.get_context())
            else:
                print("\nepoch %d:" % (epoch), "train_time(%depochs)" % self.gConfig['epoch_per_print'],
                      "=%.2f" % (check_time - self.start_time),
                      "\t acc_train = %.4f" % acc_train, "\t loss_train = %.6f" % loss_train,
                      "\t acc_test = %.4f" % acc_test, "\t loss_test = %.6f" % loss_test,
                      "\t acc_valid = %.4f" % acc_valid, "\t loss_valid = %.6f" % loss_valid,
                      "\t learning_rate = %.6f" % self.get_learningrate(),
                      '\t global_step = %d' % self.get_global_step(),
                      "  context:%s" % self.get_context())
            self.start_time = check_time
            self.debug_info()  #tensorflow框架下有无效打印，需要修改
        #if epoch % self.epochs_per_checkpoint == 0:
            #self.saveCheckpoint()
        #    self.checkpoint.save_model(self.net, self.optimizer)
        return


    def train(self,model_eval,getdataClass,gConfig,num_epochs):
        return self.losses_train,self.acces_train,self.losses_valid,self.acces_valid,\
               self.losses_test,self.acces_test


    def debug_info(self,*args):
        pass

