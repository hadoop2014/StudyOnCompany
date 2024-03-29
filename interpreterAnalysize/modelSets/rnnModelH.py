#from interpreterAnalysize.modelSets.modelBaseClassH import *
from interpreterAnalysize.modelSets.rnnBaseModelH import *
from torch import nn
import math


class rnnModel(rnnBaseModelH):
    def __init__(self,gConfig):
        super(rnnModel,self).__init__(gConfig)


    def get_loss(self):
        loss = nn.CrossEntropyLoss().to(self.ctx)
        return loss


    def get_net(self):
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
        self.dropout = self.gConfig['dropout']
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
            'gru': nn.GRU(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers),
            'lstm': nn.LSTM(input_size=self.input_dim,hidden_size=self.rnn_hiddens,num_layers=self.num_layers)
        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.input_shape = (self.time_steps, self.batch_size, self.resizedshape[1])
        #self.get_net()
        cell = self.get_cell(self.gConfig['cell'])
        rnn_layer = self.cell_selector[cell]
        net = RNN(rnn_layer,self.output_dim,self.dropout,self.ctx)
        return net


    #def get_acc(self,y_hat,y):
    #    return (y_hat.argmax(dim=1) == y).sum().item()

    '''
    def run_train_loss_acc(self,X,y):
        if self.randomIterIsOn:
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
        loss = self.loss(y_hat, y.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_clipping(self.net.parameters(), self.clip_gradient, self.ctx)
        self.scheduler.step()
        self.optimizer.step()
        #if self.global_step == 0 or self.global_step == 1:
        if self.get_global_step()==0 or self.get_global_step() == 1:
            self.debug_info()
        loss = loss.item() * y.shape[0]
        acc= (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        if self.randomIterIsOn:
            self.init_state()
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat, self.state = self.net(X, self.state)
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        loss = self.loss(y_hat, y.long())
        loss = loss.item() * y.shape[0]
        acc  = (y_hat.argmax(dim=1) == y).sum().item()
        return loss,acc
    '''

    def predict(self, model):
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
        print('global_step %d, perplexity_train %.6f,perplexity_test %f.6' %
              (self.get_global_step(), perplexity_train, perplexity_test))
        return perplexity_train,perplexity_test


    def apply_model(self, net):
        self.logger.info('rnnModel.apply_model has not be implement')


    def output_info(self,y_hat,y):
        pass


    def get_input_shape(self):
        return self.input_shape


def create_object(gConfig):
    #用cnnModel实例化一个对象model
    model=rnnModel(gConfig=gConfig)
    return model