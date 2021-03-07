#from interpreterAnalysize.modelSets.modelBaseClassH import *
from interpreterAnalysize.modelSets.rnnBaseModelH import *
from torch import nn


class Loss(LossBaseH):
    def __init__(self,ctx, output_weight):
        super(Loss,self).__init__(ctx)
        self.output_weight = output_weight
        self.lossMain = nn.CrossEntropyLoss().to(self.ctx)
        self.lossSecond = nn.MSELoss().to(self.ctx)
        self.lossThird = nn.MSELoss().to(self.ctx)

    def forward(self,y_hat,y):
        if isinstance(y_hat, tuple):
            loss = self.lossSecond(y_hat[0], y[:,0]) * self.output_weight[0]
            loss += self.lossThird(y_hat[1], y[:,1]) * self.output_weight[1]
            loss += self.lossMain(y_hat[2], y[:,2].long()) * self.output_weight[2]
        else:
            y = y.long()
            loss = self.lossMain(y_hat,y)
        return loss


class Criteria(CriteriaBaseH):
    def __init__(self):
        super(Criteria,self).__init__()

    def forward(self,y_hat, y):
        if isinstance(y_hat, tuple):
            criteria = (y_hat[-1].argmax(dim=1) == y[:, -1].long()).sum().item()
        else:
            #y = y.long()
            criteria = (y_hat.argmax(dim=1) == y.long()).sum().item()
        return  criteria


class RNNMultiOutput(RNN):
    def __init__(self,rnn_layer,output_dim,dropout,ctx):
        super(RNNMultiOutput,self).__init__(rnn_layer,output_dim,dropout,ctx)
        #self.ctx = ctx
        #self.rnn = rnn_layer
        #self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        #self.dropout = nn.Dropout(dropout)
        #self.dense = nn.Linear(self.hidden_size, output_dim)
        #self.state = None
        self.output_dim = output_dim
        if isinstance(self.output_dim,list):
            dims = len(self.output_dim)
        else:
            dims = 1
        for i in range(dims):
            setattr(self, 'dense%d'%i, nn.Linear(self.hidden_size,self.output_dim[i]))


    def forward(self, X, state=None): # inputs: (batch, seq_len)
        Y, self.state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        if isinstance(Y,PackedSequence):
            Y,seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(Y)
        Y = Y.reshape(-1,Y.shape[-1])
        Y = self.dropout(Y)
        if isinstance(self.output_dim,list):
            output = tuple([getattr(self,'dense%d'%i)(Y) for i in range(len(self.output_dim))])
        else:
            output = self.dense(Y)
        return output, self.state


class rnndetectionModel(rnnBaseModelH):
    def __init__(self,gConfig):
        super(rnndetectionModel, self).__init__(gConfig)


    def get_loss(self):
        output_weight = self.gConfig['output_weight']
        return Loss(self.ctx, output_weight)


    def get_criteria(self):
        return Criteria()


    def get_net(self):
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.input_dim = getdataClass.input_dim
        self.clip_gradient = self.gConfig['clip_gradient']
        self.time_steps = self.resizedshape[0]
        self.rnn_hiddens = self.gConfig['rnn_hiddens']  # 256
        self.num_layers = self.gConfig['num_layers']
        self.output_dim = self.gConfig['output_dim']
        self.output_weight = self.gConfig['output_weight']
        self.dropout = self.gConfig['dropout']
        self.bidirectional = self.gConfig['bidirectional']
        self.activation = self.get_activation(self.gConfig['activation'])
        self.nonlinearity = self.get_nonlinearity(self.gConfig['activation'])
        self.cell = self.get_cell(self.gConfig['cell'])
        self.cell_selector = {
            'rnn': nn.RNN(input_size = self.input_dim,hidden_size=self.rnn_hiddens, num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity,bidirectional=self.bidirectional),
            'gru': nn.GRU(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers,
                          bidirectional=self.bidirectional),
            'lstm': nn.LSTM(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers,
                            bidirectional=self.bidirectional)
        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.input_shape = (self.time_steps, self.batch_size, self.resizedshape[1])
        cell = self.get_cell(self.gConfig['cell'])
        rnn_layer = self.cell_selector[cell]
        net = RNN(rnn_layer,self.output_dim,self.dropout,self.ctx)
        return net


    def predict_with_keyfileds(self,net,X,y,keyfields):
        self.init_state()
        with torch.no_grad():
            # 解决GPU　out memory问题
            y_hat, self.state = net(X, self.state)
        mergedDataFrame = self.merged_fields(keyfields, X, y, y_hat)
        #y = torch.transpose(y, 0, 1).contiguous().view(-1)
        #y_hat = y_hat.squeeze()
        if isinstance(y_hat, tuple):
            # rnndetection模型中,输出的y_hat是一个tuple型, 不能执行squeeze
            y = torch.transpose(y, 0, 1).contiguous().view(-1, y.shape[-1])
            y_hat = tuple(y.squeeze() for y in y_hat)
        else:
            y = torch.transpose(y, 0, 1).contiguous().view(-1)
            y = y.long()
            y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y)
        loss = loss.item() * y.shape[0]
        #acc = self.get_acc(y_hat,y)
        acc = self.criteria(y_hat,y)
        self.output_info(y_hat, y)
        return loss, acc, mergedDataFrame


    def merged_fields(self, keyfields, X, y, y_predict):
        # 用keyfields, X, y, y_predict拼接出原始数据 , 加上 预测市值增长率
        keyfields, seq_lengths_key = keyfields
        X_raw, seq_lengths_X =  torch.nn.utils.rnn.pad_packed_sequence(X)
        X_raw = torch.transpose(X_raw, 0, 1)
        batch_size = len(seq_lengths_X)
        if isinstance(y_predict,tuple):
            #y_predict = y_predict[-1]
            #y = y[:,-1].long()
            y_predict = [y_hat.squeeze() if y_hat.shape[-1] == 1 else y_hat.argmax(axis=1) for y_hat in y_predict]
            y_predict = torch.stack(y_predict)
            #y_predict = y_predict.reshape(-1, batch_size)
            y_predict = torch.transpose(y_predict, 0, 1)
        else:
            y_predict = y_predict.argmax(axis=1)
            y_predict = y_predict.reshape(-1, batch_size)
            y_predict = torch.transpose(y_predict, 0, 1)
        #y_predict = y_predict.reshape(-1, batch_size)
        #y_predict = torch.transpose(y_predict, 0, 1)
        getdataClass = self.gConfig['getdataClass']
        keyfields_columns = getdataClass.get_keyfields_columns()
        X_columns = getdataClass.get_X_columns()
        y_columns = getdataClass.get_y_columns()
        y_predict_columns = getdataClass.get_y_predict_columns()
        dataFrame_keyfields = pd.concat(keyfields, axis=0).reset_index(drop=True)
        dataFrame_X = pd.DataFrame(X_raw.cpu().numpy().reshape(-1, X_raw.shape[-1]).tolist(), columns=X_columns)
        dataFrame_y = pd.DataFrame(y.cpu().numpy().reshape(-1,y.shape[-1]).tolist(), columns=y_columns)
        dataFrame_y_predict = pd.DataFrame(y_predict.cpu().numpy().reshape(-1,y_predict.shape[-1]).tolist(), columns=y_predict_columns)
        dataFrame_merged = pd.concat([dataFrame_keyfields, dataFrame_X, dataFrame_y, dataFrame_y_predict], axis=1)
        return dataFrame_merged


    def process_write_to_sqlite3(self, mergedDataFrame):
        tableName = self.gConfig['tableName']
        self._write_to_sqlite3(mergedDataFrame, tableName)
        self.logger.info('success to apply model(%s) and write to predicted data to sqlite3: %s'
                             %(self.gConfig['model'], tableName))


    def run_matrix(self, loss_train, loss_test):
        return 0.0, 0.0


    #def get_acc(self,y_hat, y):
    #    if isinstance(y_hat,tuple):
    #        acc =  (y_hat[2].argmax(dim=1) == y[:,2].long()).sum().item()
    #    else:
    #        acc = (y_hat.argmax(dim=1) == y).sum().item()
    #    return acc


    def output_info(self,y_hat,y):
        if isinstance(y_hat, tuple):
            y_hat = y_hat[2].argmax(axis=1)
        else:
            y_hat = y_hat.argmax(axis=1)
        if y_hat.dim() == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        combine = list(zip(y_hat.cpu().numpy(), y.cpu().numpy()))
        end = min(len(combine), self.time_steps)
        print(f"(y_hat, y):{combine[:end]}")


    def get_input_shape(self):
        return self.input_shape


def create_object(gConfig):
    #用cnnModel实例化一个对象model
    model=rnndetectionModel(gConfig=gConfig)
    return model