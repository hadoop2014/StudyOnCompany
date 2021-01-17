from interpreterAnalysize.modelSets.modelBaseClassH import *
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self,rnn_layer,output_dim,ctx):
        super(RNN,self).__init__()
        self.ctx = ctx
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.dense = nn.Linear(self.hidden_size, output_dim)
        self.state = None


    def forward(self, X, state=None): # inputs: (batch, seq_len)
        Y, self.state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        if isinstance(Y,PackedSequence):
            Y,seq_lengths = torch.nn.utils.rnn.pad_packed_sequence(Y)
        Y = Y.reshape(-1,Y.shape[-1])
        output = self.dense(Y)
        return output, self.state


    def begin_state(self,batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)


class rnnModel(ModelBaseH):
    def __init__(self,gConfig):
        super(rnnModel,self).__init__(gConfig)


    def _init_parameters(self):
        super(rnnModel, self)._init_parameters()
        #self.loss = nn.CrossEntropyLoss().to(self.ctx)
        self.loss = nn.MSELoss().to(self.ctx)
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.input_dim = getdataClass.input_dim
        #self.idx_to_char = getdataClass.idx_to_char
        #self.char_to_idx = getdataClass.char_to_idx
        self.clip_gradient = self.gConfig['clip_gradient']
        #self.prefixes = self.gConfig['prefixes']
        #self.predict_length = self.gConfig['predict_length']
        self.time_steps = self.resizedshape[0]
        self.rnn_hiddens = self.gConfig['rnn_hiddens']  # 256
        self.num_layers = self.gConfig['num_layers']
        #self.input_dim = self.vocab_size
        self.output_dim = self.gConfig['output_dim']
        self.activation = self.get_activation(self.gConfig['activation'])
        self.nonlinearity = self.get_nonlinearity(self.gConfig['activation'])
        self.cell = self.get_cell(self.gConfig['cell'])
        #self.scratchIsOn = self.gConfig['scratchIsOn']
        #self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.cell_selector = {
            'rnn': nn.RNN(input_size = self.input_dim,hidden_size=self.rnn_hiddens, num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity),
            'gru': nn.GRU(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers),
            'lstm': nn.LSTM(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers)
        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=self.learning_rate_decay_step
                                                         ,gamma=self.learning_rate_decay_factor)
        self.input_shape = (self.time_steps, self.batch_size, self.resizedshape[1])


    def get_cell(self,cell):
        assert cell in self.gConfig['celllist'], 'cell(%s) is invalid,it must one of %s' % \
                                                               (cell, self.gConfig['celllist'])
        return cell


    def get_batch_size(self,y):
        return y.size()[0] * y.size()[1]


    def get_net(self):
        cell = self.get_cell(self.gConfig['cell'])
        rnn_layer = self.cell_selector[cell]
        self.net = RNN(rnn_layer,self.output_dim,self.ctx)


    def run_train_loss_acc(self,X,y):
        self.init_state()
        if self.state is not None:
            # 使用detach函数从计算图分离隐藏状态, 这是为了
            # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
            if isinstance(self.state, tuple):  # LSTM, state:(h, c)
                self.state = (self.state[0].detach(), self.state[1].detach())
            else:
                self.state = self.state.detach()
        # outputs有num_steps个形状为(batch_size, vocab_size)的矩阵
        (y_hat, self.state) = self.net(X, self.state)
        # Y的形状是(batch_size, num_steps)，转置后再变成长度为
        # batch * num_steps 的向量，这样跟输出的行一一对应
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_clipping(self.net.parameters(), self.clip_gradient, self.ctx)
        self.scheduler.step()
        self.optimizer.step()
        #if self.global_step == 0 or self.global_step == 1:
        if self.get_global_step() == 0 or self.get_global_step() == 1:
            self.debug_info()
        loss = loss.item() * y.shape[0]
        acc = 0
        #acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        self.init_state()
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat, self.state = self.net(X, self.state)
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y)
        loss = loss.item() * y.shape[0]
        #acc  = (y_hat.argmax(dim=1) == y).sum().item()
        acc = 0
        print("(y_hat, y):",list(zip(y_hat.cpu().numpy(), y.cpu().numpy()))[:6])
        return loss,acc


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
        acc = 0
        print("(y_hat, y):",list(zip(y_hat.cpu().numpy(), y.cpu().numpy()))[:6])
        return loss, acc, mergedDataFrame


    def merged_fields(self, keyfields, X, y, y_predict):
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
        dataFrame_X = pd.DataFrame(X_raw.cpu().numpy().reshape(-1, X_raw.shape[-1]), columns=X_columns)
        dataFrame_y = pd.DataFrame(y.cpu().numpy().reshape(-1,1), columns=y_columns)
        dataFrame_y_predict = pd.DataFrame(y_predict.cpu().numpy().reshape(-1,1), columns=y_predict_columns)
        dataFrame_merged = pd.concat([dataFrame_keyfields, dataFrame_X, dataFrame_y, dataFrame_y_predict], axis=1)
        return dataFrame_merged


    def run_matrix(self, loss_train, loss_test):
        return 0.0, 0.0


    def get_learningrate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']


    def get_global_step(self):
        global_step = 0
        if len(self.optimizer.state_dict()['state']) > 0:
            global_step = self.optimizer.state_dict()['state'][0]['step']
        return global_step


    def init_state(self):
        self.state = None
        #self.state = self.net.begin_state(batch_size=self.batch_size,num_hiddens=self.rnn_hiddens, device=self.ctx)


    def get_optim_state(self):
        return self.optimizer.state_dict()


    def load_optim_state(self,state_dict):
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.optimizer.load_state_dict(state_dict)


    def get_input_shape(self):
        return self.input_shape


    def image_record(self,global_step,tag,input_image):
        # 在RNN,LSTM,GRU中使得该函数失效
        pass


    def summary(self,net):
        summary(net, input_size=self.get_input_shape()[1:], batch_size=self.batch_size,
                device=re.findall(r'(\w*)', self.ctx.__str__())[0])


def create_object(gConfig):
    #用cnnModel实例化一个对象model
    model=rnnModel(gConfig=gConfig)
    return model