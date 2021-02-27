#from interpreterAnalysize.modelSets.modelBaseClassH import *
from interpreterAnalysize.modelSets.rnnBaseModelH import *
from torch import nn


class rnndetectionModel(rnnBaseModelH):
    def __init__(self,gConfig):
        super(rnndetectionModel, self).__init__(gConfig)


    def get_loss(self):
        return nn.CrossEntropyLoss().to(self.ctx)


    def get_net(self):
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.input_dim = getdataClass.input_dim
        self.clip_gradient = self.gConfig['clip_gradient']
        self.time_steps = self.resizedshape[0]
        self.rnn_hiddens = self.gConfig['rnn_hiddens']  # 256
        self.num_layers = self.gConfig['num_layers']
        self.output_dim = self.gConfig['output_dim']
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
        #self.get_net()
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
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y)
        loss = loss.item() * y.shape[0]
        acc = self.get_acc(y_hat,y)
        self.output_info(y_hat, y)
        #if y_hat.dim() == 0:
        #    y_hat = y_hat.unsqueeze(dim=0)
        #combine = list(zip(y_hat.cpu().numpy(), y.cpu().numpy()))
        #end = min(len(combine), self.time_steps)
        #print("(y_hat, y):",combine[:end])
        return loss, acc, mergedDataFrame


    def merged_fields(self, keyfields, X, y, y_predict):
        y_predict = y_predict.argmax(axis=1)
        # 用keyfields, X, y, y_predict拼接出原始数据 , 加上 预测市值增长率
        keyfields, seq_lengths_key = keyfields
        X_raw, seq_lengths_X =  torch.nn.utils.rnn.pad_packed_sequence(X)
        X_raw = torch.transpose(X_raw, 0, 1)
        batch_size = len(seq_lengths_X)
        y_predict = y_predict.reshape(-1, batch_size)
        y_predict = torch.transpose(y_predict, 0, 1)
        getdataClass = self.gConfig['getdataClass']
        keyfields_columns = getdataClass.get_keyfields_columns()
        X_columns = getdataClass.get_X_columns()
        y_columns = getdataClass.get_y_columns()
        y_predict_columns = getdataClass.get_y_predict_columns()
        dataFrame_keyfields = pd.concat(keyfields, axis=0).reset_index(drop=True)
        dataFrame_X = pd.DataFrame(X_raw.cpu().numpy().reshape(-1, X_raw.shape[-1]).tolist(), columns=X_columns)
        dataFrame_y = pd.DataFrame(y.cpu().numpy().reshape(-1,1).tolist(), columns=y_columns)
        dataFrame_y_predict = pd.DataFrame(y_predict.cpu().numpy().reshape(-1,1).tolist(), columns=y_predict_columns)
        dataFrame_merged = pd.concat([dataFrame_keyfields, dataFrame_X, dataFrame_y, dataFrame_y_predict], axis=1)
        return dataFrame_merged


    def process_write_to_sqlite3(self, mergedDataFrame):
        tableName = self.gConfig['tableName']
        self._write_to_sqlite3(mergedDataFrame, tableName)
        self.logger.info('success to apply model(%s) and write to predicted data to sqlite3: %s'
                             %(self.gConfig['model'], tableName))


    def run_matrix(self, loss_train, loss_test):
        return 0.0, 0.0


    def get_acc(self,y_hat, y):
        return (y_hat.argmax(dim=1) == y).sum().item()


    def output_info(self,y_hat,y):
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