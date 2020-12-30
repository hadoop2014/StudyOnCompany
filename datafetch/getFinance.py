from datafetch.getBaseClassH import *
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader


class FinanceDataSet(Dataset):
    def __init__(self,features): #self参数必须，其他参数及其形式随程序需要而不同，比如(self,*inputs)
        self._data = features

    def __len__(self):
        return len(self._data)

    def __getitem__(self,idx):
        data = self._data[idx]
        return data


def pad_tensor(vec, pad):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to

    return:
        a new tensor padded to 'pad'
    """
    if vec.dim() == 1:
        result =  torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.float)], dim=0).data.numpy()
    else:
        result = torch.cat([vec, torch.zeros(pad - len(vec),*vec.shape[1:], dtype=torch.float)], dim=0).data.numpy()
    return result

class Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self):
        pass

    def _collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' before padding like:
                '''
                [tensor([1,2,3,4]),
                 tensor([1,2]),
                 tensor([1,2,3,4,5])]
                '''
            ys - a LongTensor of all labels in batch like:
                '''
                [1,0,1]
                '''
        """
        xs = [torch.FloatTensor(v[:,:-1]) for v in batch] #获取特征, T * input_dim
        ys = [torch.FloatTensor(v[:,-1]) for v in batch] #获取标签, T * 1
        #ys = torch.LongTensor([v[:,-1] for v in batch])  #获取标签, T * 1
        # 获得每个样本的序列长度
        seq_lengths = torch.LongTensor([v for v in map(len, xs)])
        max_len = max([len(v) for v in xs])
        # 每个样本都padding到当前batch的最大长度
        xs = torch.FloatTensor([pad_tensor(v, max_len) for v in xs])
        ys = torch.FloatTensor([pad_tensor(v, max_len) for v in ys])
        # 把xs和ys按照序列长度从大到小排序
        seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        xs = xs[perm_idx]
        ys = ys[perm_idx]
        return xs, seq_lengths, ys

    def __call__(self, batch):
        return self._collate(batch)


class getFinanceDataH(getdataBase):
    def __init__(self,gConfig):
        super(getFinanceDataH,self).__init__(gConfig)
        self.data_path = self.gConfig['data_directory']
        self.filename = self.gConfig['lybric_filename']
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.time_steps = self.gConfig['time_steps']
        self.ctx = self.get_ctx(gConfig['ctx'])
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.k = self.gConfig['k']
        self.load_data()


    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = torch.device(type='cuda',index=0) #,index=0)
        else:
            ctx = torch.device(type='cpu')#,index=0)
        return ctx


    def load_data(self,*args):
        dictSourceData = self.get_dictSourceData(self.gConfig)
        tableName = dictSourceData['tableName']
        sql = ''
        sql = sql + 'select * from {}'.format(tableName)
        dataFrame = pd.read_sql(sql,self._get_connect())
        dataGroups = dataFrame.groupby(dataFrame['公司代码'])
        features = []
        labels = []
        fieldStart = dictSourceData['fieldStart']
        for _,group in dataGroups:
            group = group.sort_values(by=['报告时间'], ascending=True)
            features += [torch.from_numpy(np.array(group.iloc[:,fieldStart:],dtype=np.float32))]
            #labels += [torch.from_numpy(np.array(group.iloc[:,-1],dtype=np.float32))]
        self.features = FinanceDataSet(features)
        #self.labels = FinanceDataSet(labels)
        self.input_dim = len(dataFrame.columns) - fieldStart - 1
        self.transformers = [self.fn_transpose]
        self.resizedshape = [self.time_steps,self.input_dim]
        #train_data_X = DataLoader(dataset=self.features, batch_size=self.batch_size, num_workers=self.cpu_num
        #                        , collate_fn=Collate())

        #for X in train_data_X:
        #    shape = X.shape

        #torch.nn.utils.rnn.pack_padded_sequence()

    '''
    def fn_onehot(self,x):
        x = nd.transpose(x)
        return nd.one_hot(x,self.vocab_size)
    '''

    def fn_transpose(self,x):
        x = torch.transpose(x,0,1)
        return x


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


    '''
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
    '''

    '''
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
   '''

    def data_iter(self,features,labels,batch_size,time_steps,ctx):
        features = torch.nn.utils.rnn.pack_padded_sequence(features,time_steps)
        labels = torch.nn.utils.rnn.pack_padded_sequence(labels)
        data_len = len(features)
        batch_len = data_len // batch_size
        indicesX = features[0: batch_size * batch_len].reshape((batch_size, batch_len, -1))
        indicesY = labels[0: batch_size * batch_len].reshape((batch_size, batch_len, -1))
        epoch_size = (batch_len - 1) // time_steps
        for i in range(epoch_size):
            i = i * time_steps
            X = indicesX[:, i: i + time_steps,:]
            Y = indicesY[:, i: i + time_steps,:]
            yield X, Y


    @getdataBase.getdataForUnittest
    def getTrainData(self,batch_size):
        self.train_data,self.test_data = self.get_k_fold_data(self.k,self.features)
        #if self.randomIterIsOn == True:
        #    train_iter = self.data_iter_random(self.train_data,self.batch_size,self.time_steps,self.ctx)
        #else:
        #    train_iter = self.data_iter_consecutive(self.train_data, self.batch_size, self.time_steps, self.ctx)
        #train_iter = self.data_iter(self.train_data_X,self.train_data_y,self.batch_size,self.time_steps,self.ctx)
        train_iter = DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.cpu_num
                               ,collate_fn=Collate())
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()


    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size):
        #if self.randomIterIsOn == True:
        #    test_iter = self.data_iter_random(self.test_data,self.batch_size,self.time_steps,self.ctx)
        #else:
        #    test_iter = self.data_iter_consecutive(self.test_data, self.batch_size, self.time_steps, self.ctx)
        test_iter = DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.cpu_num
                              ,collate_fn=Collate())
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()


    def get_dictSourceData(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return gConfig[dataset_name]


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
    "mxnet":getFinanceDataH,
    "pytorch":getFinanceDataH
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
