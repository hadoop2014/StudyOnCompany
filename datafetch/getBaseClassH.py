from datafetch.getBaseClass import *
import sys
import torch
import numpy as np
import pandas as pd
import copy
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
            result =  torch.cat([vec, torch.zeros(pad - len(vec), dtype=torch.float)], dim=0)#.data.numpy()
        else:
            result = torch.cat([vec, torch.zeros(pad - len(vec),*vec.shape[1:], dtype=torch.float)], dim=0)#.data.numpy()
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

    def __init__(self,ctx,time_steps,fieldEnd,batch_first):
        self.ctx = ctx
        self.time_steps = time_steps
        self.fieldEnd = fieldEnd
        self.batch_first = batch_first

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
                    return v.iloc[resLength:]
                else:
                    raise ValueError('type(%s) of v is not supported, it must be torch.Tensor or pd.DataFrame' % type(v))
            else:
                return v

        if isinstance(batch[0], torch.Tensor):
            #xs = [torch.FloatTensor(v[:,:self.fieldEnd]) for v in batch] #获取特征, T * input_dim
            #ys = [torch.FloatTensor(v[:,self.fieldEnd]) for v in batch] #获取标签, T * 1
            xs = [v[:, :self.fieldEnd] for v in batch]  # 获取特征, T * input_dim
            ys = [v[:, self.fieldEnd:] for v in batch]  # 获取标签, T * 1
            max_len = max([len(v) for v in xs])
            if max_len > self.time_steps:
                # 如果最大长度超出 time_steps,则把早期超出time_steps部分的数据删除掉
                xs = list(map(cut_tensor, xs))
                ys = list(map(cut_tensor, ys))
                max_len = self.time_steps
            # 获得每个样本的序列长度,必须放在CPU上
            seq_lengths = torch.LongTensor([v for v in map(len, xs)])
            # 每个样本都padding到当前batch的最大长度
            xs = torch.stack([pad_tensor(v, max_len) for v in xs],dim=0)
            ys = torch.stack([pad_tensor(v, max_len) for v in ys],dim=0)
            # 把xs和ys按照序列长度从大到小排序
            seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            xs = xs[perm_idx].to(self.ctx)
            ys = ys[perm_idx].to(self.ctx)
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


class getdataBaseH(getdataBase):
    def __init__(self,gConfig):
        super(getdataBaseH,self).__init__(gConfig)
        self.resize = self.gConfig['resize']
        self.test_percent = self.gConfig['test_percent']
        self.batch_size = self.gConfig['batch_size']
        self.batch_first = self.gConfig['batch_first'],
        from torchvision import datasets
        self.dataset_selector={
            'mnist':datasets.MNIST,
            'fashionmnist':datasets.FashionMNIST,
            'cifar10':datasets.CIFAR10
        }
        #self.load_data(resize=self.resize, root=self.data_path)


    def load_data(self,resize=None,root="",*args):
        #from torch import utils
        from torchvision import transforms
        import torch.utils.data
        root = os.path.expanduser(root)
        transformer = []
        if resize is not None and resize != 0:
            transformer += [transforms.Resize(resize)]
            self.resizedshape = [self.rawshape[0],resize,resize]
        transformer += [transforms.ToTensor()]
        transformer += [transforms.Normalize((0.1307,),(0.3081,))]
        transformer = transforms.Compose(transformer)
        train_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        test_data = self.dataset_selector[self.dataset_name](root=root,train=True,download=True,transform=transformer)
        num_workers = 0 if sys.platform.startswith('win32') else self.cpu_num
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.ctx == 'gpu' else {'num_workers':num_workers}
        self.train_iter = torch.utils.data.DataLoader(train_data,batch_size=self.batch_size,shuffle=True,**kwargs)
        self.test_iter = torch.utils.data.DataLoader(test_data,batch_size=self.batch_size,shuffle=True,**kwargs)


    def _table_to_dataFrame(self,tableName, prefixNeeded = True):
        assert isinstance(self.gConfig['报告类型'],list) and self.gConfig['报告类型'][0] == '年度报告'\
            ,"报告类型(%s) != 年度报告,当前不支持该类型!"%self.gConfig['报告类型']
        if prefixNeeded:
            tableNamePrefix = self._get_tableprefix_by_report_type(self.gConfig['报告类型'][0])
        else:
            tableNamePrefix = NULLSTR
        tableName = tableNamePrefix + tableName
        sql = ''
        sql = sql + 'select * from {}'.format(tableName)
        dataFrame = pd.read_sql(sql, self._get_connect())
        dataFrameNa = dataFrame[dataFrame.isna().any(axis=1)]
        if len(dataFrameNa) > 0:
            self.logger.error('NaN found in input tensor:%s' % dataFrameNa.values)
            raise ValueError("Nan found in dataFrame,please repaire it:\n %s" % dataFrameNa)
        return dataFrame


    def _preprocessing(self,dataFrame:DataFrame,tableName,fieldInsert):
        """
        args:
            dataFrame - 从公司价格分析表中读取的数据
            tableName - 数据库的表名: 公司价格分析表
        reutrn:
            dataFrameResult - 经过预处理后的数据, 采用sklearn.preprocessing.MaxAbsScaler做归一化处理,处理原则:
            '''
            1) 读取 公司价格分析表 中'preprocessing'中的参数, 将dependent为空的字段 排序放到前面,这些字段用fit_transform先进行处理,
               将fit后的scaler记录在dictScaler中, 而dependent不为空的字段, 根据dependent指示的字段中从dictScaler中读取scaler,
               调用transform处理. 如 报告周指数 采用MaxAbsSaler.fit_transform处理,而起始周指数则采用报告周指数的MaxAbsScaler.transform处理
            2) 针对dictTablePreprocessing[key]['scale'] == 'group'场景, 按公司分组进行归一化处理,如营业收入字段,每个公司都要进行单独归一化,
            3) 针对dictTablePreprocessing[key]['scale'] == 'whole'场景,按字段进行全局归一化,如 报告周指数 字段, 因为对所有公司都是取的同一个数据.
            4) 将归一化后的字段覆盖原dataFrame中字段,返回.
        """
        dataFrameResult = dataFrame
        dictTablePreprocessing = self.dictTables[tableName]['preprocessing']
        # 将dictTablePreprocessin的字段 进行排序, 如果dependent为空, 则优先处理, dependent不为空的字段,因为要用到其指向的字段的scaler,只能后处理.
        sortedField = sorted(dictTablePreprocessing.keys(), key=lambda x: len(dictTablePreprocessing[x]['dependent']))
        modulePreprocessing = __import__('sklearn.preprocessing',fromlist=['preprocessing'])
        modulePandas = __import__('pandas')
        orderBy = self.dictTables[tableName]["orderBy"]
        dataFrame = dataFrame.sort_values(by=orderBy,ascending=True) # 公司价格分析表采用的是 ['公司代码','报告时间'], 指数趋势分析表用的是 ["公司代码","趋势简称","报告时间"],
        groupBy = self.dictTables[tableName]['groupBy']
        dataGroups = dataFrame.groupby(by = groupBy,as_index=False) # 公司价格分析表采用的是 ['公司代码'], 指数趋势分析表用的是 ["公司代码","趋势简称"],
        dictFieldScaler = dict()
        dictScaler = dict()
        for key in sortedField:
            if dictTablePreprocessing[key]['scaler'] == 'pandas':
                method = getattr(modulePandas,dictTablePreprocessing[key]['method'])
                dataFrameScaled = method(dataFrameResult[key])
                columnsScaled = dataFrameScaled.columns.tolist()
                columns = dataFrameResult.columns.tolist()
                for column in columnsScaled:
                    columns.insert(fieldInsert, column)
                dataFrameResult = dataFrameResult.reindex(columns = columns)
                dataFrameResult[columnsScaled] = dataFrameScaled[columnsScaled]
            else:
                if dictTablePreprocessing[key]['dependent'] == NULLSTR:
                    scaler = getattr(modulePreprocessing,dictTablePreprocessing[key]['scaler'])() # 默认copy=True, 必须为该值
                    if dictTablePreprocessing[key]['scale'] == 'group':
                        dictScaler = dict([(key, copy.deepcopy(scaler)) for key in dataGroups.groups.keys()])
                    elif dictTablePreprocessing[key]['scale'] == 'whole':
                        dictScaler = copy.deepcopy(scaler)
                    dictFieldScaler.update({key: dictScaler})
                else:
                    dictScaler = dictFieldScaler[dictTablePreprocessing[key]['dependent']]
                method = dictTablePreprocessing[key]['method']
                if dictTablePreprocessing[key]['scale'] == 'group':
                    dataFrameScaled = dataGroups[key].apply(lambda group : pd.Series(
                        np.squeeze(getattr(dictScaler[group.name],method)(np.expand_dims(group,axis=1))),index=group.index
                    ))
                    dataFrameScaled = dataFrameScaled.swaplevel(0,-1).droplevel(-1)
                    dataFrameResult[key] = dataFrameScaled
                elif dictTablePreprocessing[key]['scale'] == 'whole':
                    dataFrameScaled = dataFrame[key].transform(lambda group : pd.Series(
                        np.squeeze(getattr(dictScaler,method)(np.expand_dims(group,axis=1))),index=group.index
                    ))
                    dataFrameResult[key] = dataFrameScaled
            dataFrameResult = dataFrameResult.sort_values(by=orderBy, ascending=True) # 公司价格分析表采用的是 ['公司代码','报告时间'], 指数趋势分析表用的是 ["公司代码","趋势简称","报告时间"],
        return dataFrameResult


    def _get_keyfields_features(self,dataFrame:DataFrame,fieldStart,tableName):
        groupBy = self.dictTables[tableName]['groupBy']
        dataGroups = dataFrame.groupby(by=groupBy) #
        keyfields = []
        features = []
        sortBy = self.dictTables[tableName]['orderBy']
        for _, group in dataGroups:
            group = group.sort_values(by=sortBy[-1], ascending=True) # 公司价格分析表和指数趋势分析表,都用的是 ['报告时间']
            if len(group) <= 1:
                pass
            keyfields += [group.iloc[:, :fieldStart]]
            features += [torch.tensor(np.array(group.iloc[:, fieldStart:]), dtype=torch.float)]
        return keyfields,features


    def get_ctx(self,ctx):
        assert ctx in self.gConfig['ctxlist'], 'ctx(%s) is invalid,it must one of %s' % \
                                                               (ctx, self.gConfig['ctxlist'])
        if ctx == 'gpu':
            ctx = torch.device(type='cuda',index=0) #,index=0)
        else:
            ctx = torch.device(type='cpu')#,index=0)
        return ctx


    def get_array(self,x,ctx):
        return torch.tensor(x, device = ctx)


    def get_rawshape(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'],dataset_name,self.__class__.__name__)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return [gConfig[dataset_name]['dim']]


    def get_dictSourceData(self,gConfig):
        dataset_name = re.findall('get(.*)Data', self.__class__.__name__).pop().lower()
        assert dataset_name in gConfig['datasetlist'], \
            'datasetlist(%s) is invalid,one of it must be a substring (%s) of class name(%s)' % \
            (gConfig['datasetlist'], dataset_name, self.__class__.__name__)
        if dataset_name not in self.gConfig:
            raise ValueError('dataset(%s) has not be configed in datasetlist(%s)'
                             %(dataset_name,gConfig['datasetlist']))
        return gConfig[dataset_name]