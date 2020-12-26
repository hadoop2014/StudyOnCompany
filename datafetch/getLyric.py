from  datafetch.getBaseClass import *
from mxnet import  nd
import mxnet as mx
import zipfile
import random
import numpy as np


class getLyricDataM(getdataBase):
    def __init__(self,gConfig):
        super(getLyricDataM,self).__init__(gConfig)
        self.data_path = self.gConfig['data_directory']
        self.filename = self.gConfig['lybric_filename']
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.time_steps = self.gConfig['time_steps']
        self.ctx = self.get_ctx(gConfig['ctx'])
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.k = self.gConfig['k']
        self.load_data(self.filename,root=self.data_path)


    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = mx.gpu(0)
        else:
            ctx = mx.cpu(0)
        return ctx


    def load_data(self,filename,root):
        root = os.path.expanduser(root)
        with zipfile.ZipFile(os.path.join(root,filename)) as zin:
            filename = re.findall('(.*).zip',filename).pop()
            with zin.open(filename) as f:
                corpus_chars = f.read().decode('utf-8')
        corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
        corpus_chars = corpus_chars[0:10000]
        self.idx_to_char = list(set(corpus_chars))
        self.char_to_idx = dict([(char, i) for i, char in enumerate(self.idx_to_char)])
        self.vocab_size = len(self.char_to_idx)
        self.corpus_indices = [self.char_to_idx[char] for char in corpus_chars]

        self.transformers = [self.fn_onehot]
        self.resizedshape = [self.time_steps,self.vocab_size]


    def fn_onehot(self,x):
        x = nd.transpose(x)
        return nd.one_hot(x,self.vocab_size)


    def transform(self,reader,transformers):
        """
        Create a batched reader.

        :param reader: the data reader to read from.
        :type reader: callable
        :param transformer: a list of transformer.
        :type transformer: list
        :return: the transformed reader.
        :rtype: callable
        """
        def transform_reader():
            for (X,y) in reader:
                for transformer in transformers:
                    X = transformer(X)
                yield (X,y)
        return transform_reader


    # Ｋ折交叉验证
    def get_k_fold_data(self, k, features):
        assert k > 1, 'k折交叉验证算法中，必须满足条件ｋ>1'
        fold_size = len(features)// k
        X_train, y_train = None, None
        X_valid, y_valid = None, None
        i = np.random.randint(k)
        for j in range(k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            X_part = features[idx.start:idx.stop]
            if j == i:
                X_valid = X_part
            elif X_train is None:
                X_train = X_part
            else:
                #X_train = nd.concat(X_train, X_part, dim=0)
                X_train.extend(X_part)
                #y_train = nd.concat(y_train, y_part, dim=0)
        return X_train,  X_valid


    def data_iter_random(self, corpus_indices, batch_size, time_steps, ctx=None):
        # 减1是因为输出的索引是相应输入的索引加1
        num_examples = (len(corpus_indices) - 1) // time_steps
        epoch_size = num_examples // batch_size
        example_indices = list(range(num_examples))
        random.shuffle(example_indices)

        # 返回从pos开始的长为time_steps的序列
        def _data(pos):
            return corpus_indices[pos: pos + time_steps]

        for i in range(epoch_size):
            # 每次读取batch_size个随机样本
            i = i * batch_size
            batch_indices = example_indices[i: i + batch_size]
            X = [_data(j * time_steps) for j in batch_indices]
            Y = [_data(j * time_steps + 1) for j in batch_indices]
            yield nd.array(X, ctx), nd.array(Y, ctx)


    def data_iter_consecutive(self, corpus_indices, batch_size, time_steps, ctx=None):
        corpus_indices = nd.array(corpus_indices, ctx=ctx)
        data_len = len(corpus_indices)
        batch_len = data_len // batch_size
        indices = corpus_indices[0: batch_size * batch_len].reshape((
            batch_size, batch_len))
        epoch_size = (batch_len - 1) // time_steps
        for i in range(epoch_size):
            i = i * time_steps
            X = indices[:, i: i + time_steps]
            Y = indices[:, i + 1: i + time_steps + 1]
            yield X, Y


    @getdataBase.getdataForUnittest
    def getTrainData(self,batch_size):
        self.train_data,self.test_data = self.get_k_fold_data(self.k,self.corpus_indices)
        if self.randomIterIsOn == True:
            train_iter = self.data_iter_random(self.train_data,self.batch_size,self.time_steps,self.ctx)
        else:
            train_iter = self.data_iter_consecutive(self.train_data, self.batch_size, self.time_steps, self.ctx)
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()


    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size):
        if self.randomIterIsOn == True:
            test_iter = self.data_iter_random(self.test_data,self.batch_size,self.time_steps,self.ctx)
        else:
            test_iter = self.data_iter_consecutive(self.test_data, self.batch_size, self.time_steps, self.ctx)
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()


    def get_rawshape(self,gConfig):
        is_find = False
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'],dataset_name,self.__class__.__name__)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name]['dim']]


    def get_classnum(self,gConfig):
        #非分类用数据集，需要重写该函数，返回None
        return None


class_selector = {
    "mxnet":getLyricDataM,
    #"tensorflow":getLyricDataM,
    "pytorch":getLyricDataM,
    #"paddle":getLyricDataM,
    #"keras": getLyricDataM
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
