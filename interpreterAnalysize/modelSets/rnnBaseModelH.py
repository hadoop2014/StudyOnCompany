from interpreterAnalysize.modelSets.modelBaseClassH import *
from torch import nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self,rnn_layer,output_dim,dropout,ctx):
        super(RNN,self).__init__()
        self.ctx = ctx
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim
        if isinstance(self.output_dim,list):
            for i in range(len(self.output_dim)):
                setattr(self, 'dense%d' % i, nn.Linear(self.hidden_size, self.output_dim[i]))
        else:
            self.dense = nn.Linear(self.hidden_size, self.output_dim)
        self.state = None


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
        #output = self.dense(Y)
        return output, self.state


    def begin_state(self,batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device),)


class rnnBaseModelH(ModelBaseH):
    def __init__(self,gConfig):
        super(rnnBaseModelH, self).__init__(gConfig)


    def _init_parameters(self):
        super(rnnBaseModelH, self)._init_parameters()
        self.time_steps = self.gConfig['time_steps']
        self.clip_gradient = self.gConfig['clip_gradient']
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.batch_first = self.gConfig['batch_first']
        self.loss = self.get_loss()
        self.net = self.get_net()
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.scheduler = self.get_scheduler(self.optimizer,self.learning_rate_decay_step, self.learning_rate_decay_factor)


    def get_cell(self,cell):
        assert cell in self.gConfig['celllist'], 'cell(%s) is invalid,it must one of %s' % \
                                                               (cell, self.gConfig['celllist'])
        return cell


    def get_batch_size(self,y):
        return y.size()[0] * y.size()[1]


    def get_scheduler(self, optimizer, step_size, gamma):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size#self.learning_rate_decay_step
                                        , gamma=gamma) #self.learning_rate_decay_factor)
        return scheduler


    def get_loss(self):
        raise NotImplementedError('you should implement it!')


    def get_net(self):
        raise NotImplementedError('you should implement it!')


    def run_train_loss_acc(self,X,y):
        if self.randomIterIsOn == True:
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
        acc = self.get_acc(y_hat,y)
        return loss,acc


    def run_eval_loss_acc(self, X, y):
        if self.randomIterIsOn == True:
            self.init_state()
        with torch.no_grad():
            #解决GPU　out memory问题
            y_hat, self.state = self.net(X, self.state)
        y = torch.transpose(y, 0, 1).contiguous().view(-1)
        y_hat = y_hat.squeeze()
        loss = self.loss(y_hat, y)
        loss = loss.item() * y.shape[0]
        acc = self.get_acc(y_hat,y)
        self.output_info(y_hat,y)
        return loss,acc


    def get_acc(self,y_hat,y):
        raise NotImplementedError('you should implement it!')


    def output_info(self,y_hat,y):
        raise NotImplementedError('you should implement in child class!')


    def predict_with_keyfileds(self,net,X,y,keyfields):
        raise NotImplementedError('you should implement in child class!')


    def run_matrix(self, loss_train, loss_test):
        raise NotImplementedError('you should implement it!')


    def get_learningrate(self):
        return self.optimizer.state_dict()['param_groups'][0]['lr']


    def get_global_step(self):
        global_step = 0
        if len(self.optimizer.state_dict()['state']) > 0:
            global_step = self.optimizer.state_dict()['state'][0]['step']
        return global_step


    def init_state(self):
        self.state = None


    def get_optim_state(self):
        return self.optimizer.state_dict()


    def load_optim_state(self,state_dict):
        self.optimizer = self.get_optimizer(self.gConfig['optimizer'], self.net.parameters())
        self.optimizer.load_state_dict(state_dict)


    def get_input_shape(self):
        raise NotImplementedError('you should implement it!')
        #return self.input_shape


    def image_record(self,global_step,tag,input_image):
        # 在RNN,LSTM,GRU中使得该函数失效
        pass


    def summary(self,net):
        summary(net, input_size=self.get_input_shape()[1:], batch_size=self.batch_size,
                device=re.findall(r'(\w*)', self.ctx.__str__())[0])

