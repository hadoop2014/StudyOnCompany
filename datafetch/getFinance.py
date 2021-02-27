#!/usr/bin/env Python
# coding   : utf-8
# @Time    : 1/16/2021 5:03 PM
# @Author  : wu.hao
# @File    : getFinance.py
# @Note    : 用于获取公司财务数据和股票价格数据, 最终用于股票价格预测
from datafetch.getBaseClassH import *


class getFinanceDataH(getdataBaseH):
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


    def load_data(self,*args):
        tableName = self.dictSourceData['tableName']
        dataFrame = self._table_to_dataFrame(tableName)
        dataFrame = self._preprocessing(dataFrame,tableName)
        dataFrameTest = dataFrame
        dataFrameValid = dataFrame
        # 最后一个年度的间隔时长在训练时不为1, 在预测时设置为1, 表示预测一整年的增长率
        dataFrameValid.loc[dataFrameValid['训练标识'] == 0,'间隔时长'] = 1.0
        # 把最后一个年度的数据排除在训练数据之外,因为这个时候还没有标签数据
        dataFrame = dataFrame[dataFrame['训练标识'] == 1]
        fieldStart = self.dictSourceData['fieldStart']
        fieldEnd = self.dictSourceData['fieldEnd']
        self.keyfields,self.features = self._get_keyfields_features(dataFrame,fieldStart,tableName)
        self.keyfieldsTest,self.featuresTest = self._get_keyfields_features(dataFrameTest, fieldStart,tableName)
        self.keyfieldsValid,self.featuresValid = self._get_keyfields_features(dataFrameValid,fieldStart,tableName)
        self.columns = dataFrame.columns.values
        self.fieldStart = fieldStart
        self.fieldEnd = fieldEnd
        self.input_dim = len(dataFrame.columns) - fieldStart + fieldEnd
        self.transformers = [self.fn_transpose]
        self.resizedshape = [self.time_steps,self.input_dim]

    '''
    def _preprocessing(self,dataFrame:DataFrame,tableName):
        """
        args:
            dataFrame - 从公司价格分析表中读取的数据
            tableName - 数据库的表名: 公司价格分析表
        reutrn:
            dataFrameResult - 经过预处理后的数据, 采用sklearn.preprocessing.MaxAbsScaler做归一化处理,处理原则:
            1) 读取 公司价格分析表 中preprocessing中的参数, 将dependent为空的字段 排序放到前面,这些字段用fit_transform先进行处理,
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
        dataFrame = dataFrame.sort_values(by=['公司代码','报告时间'],ascending=True)
        dataGroups = dataFrame.groupby(dataFrame['公司代码'],as_index=False)
        dictFieldScaler = dict()
        dictScaler = dict()
        for key in sortedField:
            if dictTablePreprocessing[key]['dependent'] == NULLSTR:
                scaler = getattr(modulePreprocessing,dictTablePreprocessing[key]['scaler'])() # 默认copy=True,必须的
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
            dataFrameResult = dataFrameResult.sort_values(by=['公司代码', '报告时间'], ascending=True)
        return dataFrameResult
    '''

    '''
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
            features += [torch.tensor(np.array(group.iloc[:, fieldStart:], dtype=np.float32))]
        return keyfields,features
    '''


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
        #self.train_data = self.features
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
