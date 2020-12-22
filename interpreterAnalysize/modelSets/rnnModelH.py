from interpreterAnalysize.modelSets.modelBaseClassH import *
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self,rnn_layer,batch_size,input_dim,rnn_hiddens,output_dim,num_layers,activation,ctx):
        super(RNN,self).__init__()
        self.ctx = ctx
        self.activation =activation # sigmoid
        self.rnn = rnn_layer
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_size = rnn_hiddens
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dense = nn.Linear(self.hidden_size,self.output_dim)


    #def forward(self, x, state):
    #    Y, state = self.rnn(x,state)
        # 全连接层会首先将Y的形状变成(time_steps * batch_size, num_hiddens)，它的输出
        # 形状为(time_steps * batch_size, vocab_size)
        # output = self.dense(Y.reshape((-1, Y.shape[-1])))
    #    output = self.dense(Y.reshape((-1, self.hidden_size)))
    #    return output, state


    def forward(self,input):
        hidden = torch.zeros(self.num_layers, 2, self.hidden_size,device=self.ctx)
        out, _ = self.rnn(input, hidden)  # out: seg_len * batch_size * hidden_size
        #return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵
        return out.reshape(-1, self.hidden_size)


    def begin_state(self, *args, **kwargs):
        #return self.rnn.begin_state(*args,**kwargs)
        return torch.zeros(self.batch_size,self.hidden_size)


class rnnModel(ModelBaseH):
    def __init__(self,gConfig):
        super(rnnModel,self).__init__(gConfig)


    def _init_parameters(self):
        super(rnnModel, self)._init_parameters()
        self.loss = nn.CrossEntropyLoss().to(self.ctx)
        getdataClass = self.gConfig['getdataClass']
        self.resizedshape = getdataClass.resizedshape
        self.vocab_size = getdataClass.vocab_size
        self.idx_to_char = getdataClass.idx_to_char
        self.char_to_idx = getdataClass.char_to_idx
        self.clip_gradient = self.gConfig['clip_gradient']
        self.prefixes = self.gConfig['prefixes']
        self.predict_length = self.gConfig['predict_length']
        self.time_steps = self.resizedshape[0]
        self.rnn_hiddens = self.gConfig['rnn_hiddens']  # 256
        self.num_layers = self.gConfig['num_layers']
        self.input_dim = self.vocab_size
        self.output_dim = self.vocab_size
        self.activation = self.get_activation(self.gConfig['activation'])
        self.nonlinearity = self.get_nonlinearity(self.gConfig['activation'])
        self.cell = self.get_cell(self.gConfig['cell'])
        self.scratchIsOn = self.gConfig['scratchIsOn']
        self.cell_selector = {
            'rnn': nn.RNN(input_size = self.input_dim,hidden_size=self.rnn_hiddens, num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity,batch_first=True),
                           #i2h_weight_initializer=self.weight_initializer,
                           #h2h_weight_initializer=self.weight_initializer,
                           #i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'gru': nn.GRU(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers,
                          batch_first=True),
                          # i2h_weight_initializer=self.weight_initializer,
                          # h2h_weight_initializer=self.weight_initializer,
                          # i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'lstm': nn.LSTM(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers,
                            batch_first=True)
                          #   i2h_weight_initializer=self.weight_initializer,
                          #   h2h_weight_initializer=self.weight_initializer,
                          #   i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer)
        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.input_shape = (self.batch_size, *self.resizedshape)


    def get_cell(self,cell):
        assert cell in self.gConfig['celllist'], 'cell(%s) is invalid,it must one of %s' % \
                                                               (cell, self.gConfig['celllist'])
        return cell


    def get_net(self):
        #batch_size = self.batch_size
        #input_dim = self.input_dim
        #rnn_hiddens =self.rnn_hiddens
        #output_dim = self.output_dim
        activation = self.get_activation(self.gConfig['activation'])
        cell = self.get_cell(self.gConfig['cell'])

        #if self.scratchIsOn == True:
        #    self.net = self.scratch_selector[cell]
        #else:
        rnn_layer = self.cell_selector[cell]
        self.net = RNN(rnn_layer,self.batch_size,self.input_dim,self.rnn_hiddens,self.output_dim,self.num_layers
                       ,activation,self.ctx)


    def run_train_loss_acc(self,X,y):
        self.optimizer.zero_grad()
        y_hat = self.net(X)
        loss = self.loss(y_hat, y).sum()
        loss.backward()
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info()
        self.optimizer.step()
        loss = loss.item()
        acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat = self.net(X)
        acc  = (y_hat.argmax(dim=1) == y).sum().item()
        loss = self.loss(y_hat, y).sum().item()
        return loss,acc


    def get_input_shape(self):
        return self.input_shape


def create_object(gConfig):
    #用cnnModel实例化一个对象model
    model=rnnModel(gConfig=gConfig)
    return model