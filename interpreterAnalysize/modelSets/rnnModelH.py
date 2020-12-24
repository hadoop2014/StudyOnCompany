from interpreterAnalysize.modelSets.modelBaseClassH import *
from torch import nn
import math
from torch.autograd import Variable
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self,rnn_layer,batch_size,input_dim,rnn_hiddens,output_dim,num_layers,activation,ctx):
        super(RNN,self).__init__()
        self.ctx = ctx
        #self.activation =activation # sigmoid
        '''
        self.rnn = rnn_layer
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_size = rnn_hiddens
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dense = nn.Linear(self.hidden_size,self.output_dim)
        '''
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        #self.vocab_size = input_dim
        self.dense = nn.Linear(self.hidden_size, output_dim)
        self.state = None


    #def forward(self, x, state):
    #    Y, state = self.rnn(x,state)
        # 全连接层会首先将Y的形状变成(time_steps * batch_size, num_hiddens)，它的输出
        # 形状为(time_steps * batch_size, vocab_size)
        # output = self.dense(Y.reshape((-1, Y.shape[-1])))
    #    output = self.dense(Y.reshape((-1, self.hidden_size)))
    #    return output, state


    def forward(self, X, state=None): # inputs: (batch, seq_len)
        #Y, self.state = self.rnn(torch.stack(X), state)
        Y, self.state = self.rnn(X, state)
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        Y = Y.reshape(-1,Y.shape[-1])
        output = self.dense(Y)
        return output, self.state

    '''
    def forward(self,input):
        hidden = torch.zeros(self.num_layers, 2, self.hidden_size,device=self.ctx)
        out, _ = self.rnn(input, hidden)  # out: seg_len * batch_size * hidden_size
        #return out.view(-1, self.hidden_size)  # return: seg_len * batch_size 返回一个矩阵
        return out.reshape(-1, self.hidden_size)
    '''

    '''
    def begin_state(self, *args, **kwargs):
        #return self.rnn.begin_state(*args,**kwargs)
        return torch.zeros(self.batch_size,self.hidden_size)
    '''

    def begin_state(self,batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)


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
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.cell_selector = {
            'rnn': nn.RNN(input_size = self.input_dim,hidden_size=self.rnn_hiddens, num_layers=self.num_layers,
                          nonlinearity=self.nonlinearity),
                           #i2h_weight_initializer=self.weight_initializer,
                           #h2h_weight_initializer=self.weight_initializer,
                           #i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'gru': nn.GRU(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers),
                          # i2h_weight_initializer=self.weight_initializer,
                          # h2h_weight_initializer=self.weight_initializer,
                          # i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'lstm': nn.LSTM(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers)
                          #   i2h_weight_initializer=self.weight_initializer,
                          #   h2h_weight_initializer=self.weight_initializer,
                          #   i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer)
        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.input_shape = (self.time_steps, self.batch_size, self.resizedshape[1])


    def get_cell(self,cell):
        assert cell in self.gConfig['celllist'], 'cell(%s) is invalid,it must one of %s' % \
                                                               (cell, self.gConfig['celllist'])
        return cell


    def get_batch_size(self,y):
        return y.size()[0] * y.size()[1]


    def get_net(self):
        activation = self.get_activation(self.gConfig['activation'])
        cell = self.get_cell(self.gConfig['cell'])
        rnn_layer = self.cell_selector[cell]
        self.net = RNN(rnn_layer,self.batch_size,self.input_dim,self.rnn_hiddens,self.output_dim,self.num_layers
                       ,activation,self.ctx)


    def run_train_loss_acc(self,X,y):
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
        loss = self.loss(y_hat, y.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_clipping(self.net.parameters(), self.clip_gradient, self.ctx)
        self.optimizer.step()
        if self.global_step == 0 or self.global_step == 1:
            self.debug_info()
        loss = loss.item() * y.shape[0]
        acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        self.init_state()
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat, self.state = self.net(X, self.state)
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        loss = self.loss(y_hat, y.long())
        loss = loss.item() * y.shape[0]
        acc  = (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def predict_nlp(self, model):
        for prefix in self.prefixes:
            print(' -', self.predict_rnn(
                prefix, self.predict_length, model, self.vocab_size, self.ctx, self.idx_to_char,
                self.char_to_idx))


    def predict_rnn(self,prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                          char_to_idx):
        state = None
        output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
        for t in range(num_chars + len(prefix) - 1):
            X = torch.tensor([output[-1]], device=ctx).view(1, 1)
            #X = d2l.to_onehot(inputs, vocab_size)  # X是个list
            X = self.to_onehot(X,vocab_size)
            X = torch.stack(X)
            if state is not None:
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].to(ctx), state[1].to(ctx))
                else:
                    state = state.to(ctx)
            (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(dim=1).item()))
        return ''.join([idx_to_char[i] for i in output])


    def to_onehot(self,X, n_class):
        # X shape: (batch, seq_len), output: seq_len elements of (batch, n_class)
        return [self.one_hot(X[:, i], n_class) for i in range(X.shape[1])]


    def one_hot(self,x, n_class, dtype=torch.float32):
        # X shape: (batch), output shape: (batch, n_class)
        x = x.long()
        res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
        res.scatter_(1, x.view(-1, 1), 1)
        return res


    def run_matrix(self, loss_train, loss_test):
        #rnn中用perplexity取代accuracy
        perplexity_train = math.exp(loss_train)
        perplexity_test = math.exp(loss_test)
        print('global_step %d, perplexity_train %.6f,perplexity_test %f.6'%
              (self.global_step, perplexity_train,perplexity_test))
        return perplexity_train,perplexity_test


    def init_state(self):
        self.state = None
        #self.state = self.net.begin_state(batch_size=self.batch_size,num_hiddens=self.rnn_hiddens, device=self.ctx)


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