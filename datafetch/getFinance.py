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
        fieldStart = self.dictSourceData['fieldStart']
        fieldEnd = self.dictSourceData['fieldEnd']
        dataFrame = self._table_to_dataFrame(tableName)
        dataFrame = self._preprocessing(dataFrame,tableName,fieldEnd)
        #dataFrameTest = dataFrame
        dataFrameValid = dataFrame
        # 最后一个年度的间隔时长在训练时不为1, 在预测时设置为1, 表示预测一整年的增长率
        dataFrameValid.loc[dataFrameValid['训练标识'] == 0,'间隔时长'] = 1.0
        dataFrameValid = dataFrameValid.drop(columns=['训练标识'], axis=1)
        dataFrameTest = dataFrameValid
        # 把最后一个年度的数据排除在训练数据之外,因为这个时候还没有标签数据
        dataFrame = dataFrame[dataFrame['训练标识'] == 1]
        dataFrame = dataFrame.drop(columns=['训练标识'], axis=1)
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
                                ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd'],self.batch_first))
        self.train_iter = self.transform(train_iter,self.transformers)
        return self.train_iter()


    @getdataBase.getdataForUnittest
    def getTestData(self,batch_size):
        self.test_data = self.featuresTest
        #_, self.test_data = self.get_k_fold_data(self.k, self.featuresTest)
        test_iter = DataLoader(dataset=self.test_data, batch_size=self.batch_size, num_workers=self.cpu_num
                               ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd'],self.batch_first))
        self.test_iter = self.transform(test_iter,self.transformers)
        return self.test_iter()


    @getdataBase.getdataForUnittest
    def getValidData(self,batch_size):
        keyfields_iter = DataLoader(dataset=self.keyfieldsValid, batch_size=self.batch_size, num_workers=self.cpu_num
                                    ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd'],self.batch_first))
        valid_iter = DataLoader(dataset=self.featuresValid, batch_size=self.batch_size, num_workers=self.cpu_num
                                ,collate_fn=Collate(self.ctx,self.time_steps,self.dictSourceData['fieldEnd'],self.batch_first))
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
        return self.dictSourceData['predictColumnsName']


class_selector = {
    "mxnet":getFinanceDataH,
    "pytorch":getFinanceDataH
}


def create_model(gConfig):
    getdataClass = class_selector[gConfig['framework']](gConfig)
    return getdataClass
