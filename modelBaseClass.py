# -*- coding: utf-8 -*-
# @Time    : 9/25/2019 5:03 PM
# @Author  : wuhao
# @File    : modelBaseClass.py
import json
from baseClass import *

class CheckpointModelBase(CheckpointBase):
    def __init__(self, working_directory, logger, checkpointfile, checkpointIsOn, moduleName, max_to_keep):
        super(CheckpointModelBase, self).__init__(working_directory, logger, checkpointfile, checkpointIsOn)
        self.moduleName = moduleName
        self.suffix_modelfile = 'model'
        self.model_savefile = self._check_max_modelfiles(self.directory, moduleName, max_to_keep)

    def _check_max_modelfiles(self, directory, moduleName, max_to_keep):
        """
        explain: 检查保持的模型数量.
            如果directory目录下的模型数据超过max_to_keep,则删除最老的模型
        args:
            directory - 模型文件所在的目录
            moduleName - 模型文件的前缀名
            max_to_keep - 能保留的最大模型文件个数
        reutrn:
            model_savefile - 模型所保存的文件名
            1) 取directory目录下,日期最新的一个文件名
            2) 如果directory目录下, 模型文件为空, 则构造一个日期为当天的模型文件名
        """
        files = os.listdir(directory)
        files = [file for file in files if self._is_file_needed(file, moduleName)]
        files = sorted(files, reverse=True)
        if len(files) > 0:
            model_savefile = os.path.join(directory, files[0])
            if len(files) > max_to_keep:
                files_discard = files[max_to_keep:]
                for file in files_discard:
                    os.remove(os.path.join(directory, file))
        else:
            model_savefile =  self._construct_modelfile()
        return model_savefile

    def _is_file_needed(self,fileName, moduleName):
        isFileNeeded = False
        if moduleName != NULLSTR and fileName != NULLSTR:
            fileName = os.path.split(fileName)[-1]
            suffix = fileName.split('.')[-1]
            if utile._is_matched(moduleName, fileName) and suffix == self.suffix_modelfile:
                isFileNeeded = True
        return isFileNeeded

    def _construct_modelfile(self):
        modelfile =  self.moduleName + utile.get_today() + '.' + self.suffix_modelfile
        return os.path.join(self.directory, modelfile)

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
        self.model_savefile = os.path.join(self.workingspace.directory,
                                           self._get_module_name() + '.model')# + self.gConfig['framework'])
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
        check_book = None
        if os.path.exists(check_file):
            with open(check_file, encoding='utf-8') as check_f:
                check_book = json.load(check_f)
        else:
            raise ValueError("%s is not exist,you must create first!" % check_file)
        return check_book

    '''
    def _get_class_name(self, gConfig):
        return "ModelBase"
    '''

    def get_net(self):
        pass


    def get_context(self):
        pass


    def get_learningrate(self):
        pass


    def get_global_step(self):
        pass

    '''
    def saveCheckpoint(self):
        pass
    '''

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

