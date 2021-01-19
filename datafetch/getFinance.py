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
    if isinstance(vec, torch.Tensor):
        if vec.dim() == 1:
            result =  torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.float)], dim=0).data.numpy()
        else:
            result = torch.cat([vec, torch.zeros(pad - len(vec),*vec.shape[1:], dtype=torch.float)], dim=0).data.numpy()
    elif isinstance(vec, pd.DataFrame):
        if vec.ndim == 1:
            nanRow = np.array([np.nan] * (pad - len(vec)))
            result = vec.append(pd.DataFrame(nanRow))
        else:
            nanRow = np.array([np.nan] * (pad - len(vec)) * np.prod(vec.shape[1:])).reshape(pad - len(vec),*vec.shape[1:])
            nanFrame = pd.DataFrame(nanRow,columns=vec.columns)
            result = vec.append(nanFrame)
    else:
        raise  ValueError('type of vec is error,it must be torch.Tensor or DataFrame')
    return result


class Collate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self,ctx,time_steps,fieldEnd):
        self.ctx = ctx
        self.time_steps = time_steps
        self.fieldEnd = fieldEnd

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
        def cut_tensor(v):
            resLength = len(v) - self.time_steps
            if resLength > 0:
                if isinstance(v, torch.Tensor):
                    return v[resLength:]
                elif isinstance(v, pd.DataFrame):
                    return v.loc[resLength:]
                else:
                    raise ValueError('type(%s) of v is not supported, it must be torch.Tensor or pd.DataFrame' % type(v))
            else:
                return v

        if isinstance(batch[0], torch.Tensor):
            xs = [torch.FloatTensor(v[:,:self.fieldEnd]) for v in batch] #获取特征, T * input_dim
            ys = [torch.FloatTensor(v[:,self.fieldEnd]) for v in batch] #获取标签, T * 1
            max_len = max([len(v) for v in xs])
            if max_len > self.time_steps:
                # 如果最大长度超出 time_steps,则把早期超出time_steps部分的数据删除掉
                xs = list(map(cut_tensor, xs))
                ys = list(map(cut_tensor, ys))
                max_len = self.time_steps
            # 获得每个样本的序列长度
            seq_lengths = torch.LongTensor([v for v in map(len, xs)])
            # 每个样本都padding到当前batch的最大长度
            xs = torch.FloatTensor([pad_tensor(v, max_len) for v in xs])
            ys = torch.FloatTensor([pad_tensor(v, max_len) for v in ys])
            # 把xs和ys按照序列长度从大到小排序
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            xs = xs[perm_idx].to(self.ctx)
            ys = ys[perm_idx]
            return xs, seq_lengths, ys
        elif isinstance(batch[0], pd.DataFrame):
            xs = batch
            max_len = max([len(v) for v in xs])
            if max_len > self.time_steps:
                # 如果最大长度超出 time_steps,则把早期超出time_steps部分的数据删除掉
                xs = list(map(cut_tensor, xs))
                max_len = self.time_steps
            # 获得每个样本的序列长度
            seq_lengths = torch.LongTensor([v for v in map(len, xs)])
            # 每个样本都padding到当前batch的最大长度
            xs = [pad_tensor(v, max_len) for v in xs]
            # 把xs和ys按照序列长度从大到小排序
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            xs = [xs[i] for i in perm_idx]
            return xs, seq_lengths
        else:
            raise ValueError('type(%s) of batch items is not supported, it must be torch.Tensor or pd.DataFrame!' % type(batch[0]))

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
        self.dictSourceData = self.get_dictSourceData(self.gConfig)
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
        tableName = self.dictSourceData['tableName']
        sql = ''
        sql = sql + 'select * from {}'.format(tableName)
        dataFrame = pd.read_sql(sql,self._get_connect())
        dataFrameNa = dataFrame[dataFrame.isna().any(axis=1)]
        if len(dataFrameNa) > 0:
            self.logger.error('NaN found in input tensor:%s'%dataFrameNa.values)
            raise ValueError("Nan found in dataFrame,please repaire it:\n %s" % dataFrameNa)
        dataFrameTest = dataFrame
        dataFrameValid = dataFrame
        # 最后一个年度的间隔时长在训练时不为1, 在预测时设置为1, 表示预测一整年的增长率
        dataFrameValid.loc[dataFrameValid['训练标识'] == 0,'间隔时长'] = 1.0
        fieldStart = self.dictSourceData['fieldStart']
        fieldEnd = self.dictSourceData['fieldEnd']
        # 把最后一个年度的数据排除在训练数据之外,因为这个时候还没有标签数据
        dataFrame = dataFrame[dataFrame['训练标识'] == 1]
        self.keyfields,self.features = self._get_keyfields_features(dataFrame,fieldStart)
        self.keyfieldsTest,self.featuresTest = self._get_keyfields_features(dataFrameTest, fieldStart)
        self.keyfieldsValid,self.featuresValid = self._get_keyfields_features(dataFrameValid,fieldStart)
        self.columns = dataFrame.columns.values
        self.fieldStart = fieldStart
        self.fieldEnd = fieldEnd
        self.input_dim = len(dataFrame.columns) - fieldStart + fieldEnd
        self.transformers = [self.fn_transpose]
        self.resizedshape = [self.time_steps,self.input_dim]


    def _get_keyfields_features(self,dataFrame,fieldStart):
        dataGroups = dataFrame.groupby(dataFrame['公司代码'])
        keyfields = []
        features = []
        for _, group in dataGroups:
            group = group.sort_values(by=['报告时间'], ascending=True)
            if len(group) <= 1:
                pass
            keyfields += [group.iloc[:, :fieldStart]]
            features += [torch.from_numpy(np.array(group.iloc[:, fieldStart:], dtype=np.float32))]
        return keyfields,features


    def fn_transpose(self,X,seq_lengths):
        X = torch.transpose(X,0,1)
        X = torch.nn.utils.rnn.pack_padded_sequence(X,seq_lengths)
        return X


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
            for (X,seq_lengths,y) in reader:
                for transformer in transformers:
                    X = transformer(X,seq_lengths)
                yield (X,y)
        return transform_reader


    @getdataBase.getdataForUnittest
    def getTrainData(self,batch_size):
        self.train_data,self.test_data = self.get_k_fold_data(self.k,self.features)
        train_iter = DataLoader(dataset=self.train_data, batch_size=self.batch_size, num_workers=self.cpu_num
                               ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd']))
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()


    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size):
        self.test_data = self.featuresTest
        #_, self.test_data = self.get_k_fold_data(self.k, self.featuresTest)
        test_iter = DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.cpu_num
                              ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd']))
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()


    @getdataBase.getdataForUnittest
    def getValidData(self,batch_size):
        keyfields_iter = DataLoader(dataset=self.keyfieldsValid, batch_size=self.batch_size, num_workers=self.cpu_num
                                    ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd']))
        valid_iter = DataLoader(dataset=self.featuresValid, batch_size=self.batch_size, num_workers=self.cpu_num
                              ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd']))
        self.valid_iter = self.transform(valid_iter,self.transformers)
        return self.valid_iter(),keyfields_iter


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


    def get_keyfields_columns(self):
        return self.columns[:self.fieldStart]


    def get_X_columns(self):
        return self.columns[self.fieldStart: self.fieldEnd]


    def get_y_columns(self):
        return [self.columns[self.fieldEnd]]


    def get_y_predict_columns(self):
        return [self.dictSourceData['predictColumnsName']]


class_selector = {
    "mxnet":getFinanceDataH,
    "pytorch":getFinanceDataH
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
