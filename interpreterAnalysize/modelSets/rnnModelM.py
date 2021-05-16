from mxnet.gluon import loss as gloss,nn,rnn
from mxnet import gluon,init,symbol
from interpreterAnalysize.modelSets.modelBaseClassM import *
import math


class Lstm(nn.HybridBlock):
    #该类在net.hybridize()时存在问题
    def __init__(self,input_dim,rnn_hiddens,output_dim,batch_size,ctx,
                 weight_initializer,bias_initializer,**kwargs):
        super(Lstm,self).__init__(**kwargs)
        self.rnn_hiddens = rnn_hiddens
        self.batch_size = batch_size
        self.output_dims = output_dim
        #输入门参数
        self.W_xi = self.params.get('W_xi',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hi = self.params.get('W_hi',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_i = self.params.get('b_i',shape=(rnn_hiddens),init=bias_initializer)
        #遗忘门参数
        self.W_xf = self.params.get('W_xf',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hf = self.params.get('W_hf',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_f = self.params.get('b_f',shape=(rnn_hiddens),init=bias_initializer)
        #输出门参数
        self.W_xo = self.params.get('W_xo',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_ho = self.params.get('W_ho',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_o = self.params.get('b_o',shape=(rnn_hiddens),init=bias_initializer)
        #候选记忆细胞参数
        self.W_xc = self.params.get('W_xc',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hc = self.params.get('W_hc',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_c = self.params.get('b_c',shape=(rnn_hiddens),init=bias_initializer)
        #输出参数
        self.W_hq = self.params.get('W_hq',shape=(rnn_hiddens,output_dim),init=weight_initializer)
        self.b_q = self.params.get('b_q',shape=(output_dim),init=bias_initializer)


    def hybrid_forward(self, F, X, *args, **kwargs):
        W_xi = kwargs['W_xi']
        W_hi = kwargs['W_hi']
        b_i = kwargs['b_i']

        W_xf = kwargs['W_xf']
        W_hf = kwargs['W_hf']
        b_f = kwargs['b_f']

        W_xo = kwargs['W_xo']
        W_ho = kwargs['W_ho']
        b_o = kwargs['b_o']

        W_xc = kwargs['W_xc']
        W_hc = kwargs['W_hc']
        b_c = kwargs['b_c']

        W_hq = kwargs['W_hq']
        b_q = kwargs['b_q']

        (H,C) = args[0]
        outputs = 0

        for x in X:
            gate_i = F.sigmoid(F.dot(x,W_xi) + F.dot(H,W_hi) + b_i)
            gate_f = F.sigmoid(F.dot(x,W_xf) + F.dot(H,W_hf) + b_f)
            gate_o = F.sigmoid(F.dot(x,W_xo) + F.dot(H,W_ho) + b_o)
            C_tilda = F.tanh(F.dot(x,W_xc) + F.dot(H,W_hc) + b_c)
            C = gate_f * C + gate_i * C_tilda
            H = gate_o * F.tanh(C)
            Y = F.dot(H,W_hq) + b_q
            if type(outputs) == int:
                outputs = Y
            else:
                outputs = F.concat(outputs,Y,dim=0)
        return outputs,(H,C)


    def begin_state(self, *args, **kwargs):
        batch_size = kwargs['batch_size']
        ctx = kwargs['ctx']
        return (nd.zeros(shape=(batch_size, self.rnn_hiddens), ctx=ctx),
                nd.zeros(shape=(batch_size,self.rnn_hiddens),ctx=ctx))


    def get_symbol_state(self):
        return (mx.symbol.Variable('state'),mx.symbol.Variable('cell'))


class Gru(nn.HybridBlock):
    # 该类在net.hybridize()时存在问题
    def __init__(self,input_dim,rnn_hiddens,output_dim,batch_size,ctx,
                 weight_initializer,bias_initializer,**kwargs):
        super(Gru,self).__init__(**kwargs)
        self.rnn_hiddens = rnn_hiddens
        self.batch_size = batch_size
        self.output_dims = output_dim
        #更新门参数
        self.W_xz = self.params.get('W_xz',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hz = self.params.get('W_hz',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_z = self.params.get('b_z',shape=(rnn_hiddens),init=bias_initializer)
        #重置门参数
        self.W_xr = self.params.get('W_xr',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hr = self.params.get('W_hr',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_r = self.params.get('b_r',shape=(rnn_hiddens),init=bias_initializer)
        #候选隐藏状态参数
        self.W_xh =self.params.get('W_xh',shape=(input_dim,rnn_hiddens),init=weight_initializer)
        self.W_hh = self.params.get('W_hh',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)
        self.b_h = self.params.get('b_h',shape=(rnn_hiddens),init=bias_initializer)
        #输出参数
        self.W_hq = self.params.get('W_hq',shape=(rnn_hiddens,output_dim),init=weight_initializer)
        self.b_q = self.params.get('b_q',shape=(output_dim),init=bias_initializer)


    def hybrid_forward(self, F, X, *args, **kwargs):
        W_xz = kwargs['W_xz']
        W_hz = kwargs['W_hz']
        b_z = kwargs['b_z']

        W_xr = kwargs['W_xr']
        W_hr = kwargs['W_hr']
        b_r = kwargs['b_r']

        W_xh = kwargs['W_xh']
        W_hh = kwargs['W_hh']
        b_h = kwargs['b_h']

        W_hq = kwargs['W_hq']
        b_q = kwargs['b_q']

        H = args[0]
        outputs = 0#F.empty(shape=(self.batch_size,self.output_dims))
        for x in X:
            Z = F.sigmoid(F.dot(x,W_xz) + F.dot(H, W_hz) + b_z)
            R = F.sigmoid(F.dot(x,W_xr) + F.dot(H, W_hr) + b_r)
            H_tilda = F.tanh(F.dot(x, W_xh) + F.dot(R * H, W_hh) + b_h)
            H = Z * H +(1 - Z) * H_tilda
            Y = F.dot(H, W_hq) + b_q
            if type(outputs) == int :#F.zeros(0):
                outputs = Y
            else:
                outputs = F.concat(outputs,Y,dim=0)
        return outputs, H


    def begin_state(self, *args, **kwargs):
        batch_size = kwargs['batch_size']
        ctx = kwargs['ctx']
        return nd.zeros(shape=(batch_size, self.rnn_hiddens), ctx=ctx)


    def get_symbol_state(self):
        return mx.symbol.Variable('state')


class Rnn(nn.HybridBlock):
    # 该类在net.hybridize()时存在问题
    def __init__(self,input_dim,rnn_hiddens,output_dim,batch_size,ctx,
                 weight_initializer,bias_initializer,**kwargs):
        super(Rnn,self).__init__(**kwargs)
        self.rnn_hiddens = rnn_hiddens
        self.ctx = ctx
        with self.name_scope():
            # 隐藏层参数
            self.W_xh = self.params.get("W_xh",shape=(input_dim,rnn_hiddens),init=weight_initializer)#_one((num_inputs, num_hiddens))
            self.W_hh = self.params.get('W_hh',shape=(rnn_hiddens,rnn_hiddens),init=weight_initializer)#_one((num_hiddens, num_hiddens))
            self.b_h = self.params.get('b_h',shape=(rnn_hiddens),init=bias_initializer)#nd.zeros(num_hiddens, ctx=ctx)
            # 输出层参数
            self.W_hq =self.params.get('W_hq',shape=(rnn_hiddens,output_dim),init=weight_initializer) #_one((num_hiddens, num_outputs))
            self.b_q = self.params.get('b_q',shape=(output_dim),init=bias_initializer)#nd.zeros(num_outputs, ctx=ctx)


    def hybrid_forward(self, F, X, *args, **kwargs):
        W_xh = kwargs['W_xh']
        W_hh = kwargs['W_hh']
        b_h = kwargs['b_h']
        W_hq = kwargs['W_hq']
        b_q = kwargs['b_q']

        H = args[0]
        outputs = 0
        for x in X:
            H = F.tanh(F.dot(x, W_xh) + F.dot(H, W_hh) + b_h)
            Y = F.dot(H, W_hq) + b_q
            if type(outputs) == int:
                outputs = Y
            else:
                outputs = F.concat(outputs,Y,dim=0)
        return outputs, H


    def begin_state(self, *args, **kwargs):
        batch_size = kwargs['batch_size']
        ctx = kwargs['ctx']
        return nd.zeros(shape=(batch_size, self.rnn_hiddens), ctx=ctx)


    def get_symbol_state(self):
        return mx.symbol.Variable('state')


class RNN(nn.HybridBlock):
    def __init__(self,rnn_layer,rnn_hiddens,vocab_size,**kwargs):
        super(RNN,self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.hidden_size = rnn_hiddens
        self.dense = nn.Dense(self.vocab_size)


    def hybrid_forward(self, F, x, *args, **kwargs):
        state = args[0]
        Y, state = self.rnn(x,state)
        # 全连接层会首先将Y的形状变成(time_steps * batch_size, num_hiddens)，它的输出
        # 形状为(time_steps * batch_size, vocab_size)
        # output = self.dense(Y.reshape((-1, Y.shape[-1])))
        output = self.dense(F.reshape(Y, (-1, self.hidden_size)))
        return output, state


    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args,**kwargs)


class rnnModel(ModelBaseM):
    def __init__(self,gConfig):
        super(rnnModel,self).__init__(gConfig)
        self.loss = gloss.SoftmaxCrossEntropyLoss()


    def _init_parameters(self):
        super(rnnModel,self)._init_parameters()
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
        self.cell = self.get_cell(self.gConfig['cell'])
        self.scratchIsOn = self.gConfig['scratchIsOn']
        self.cell_selector = {
            'rnn': rnn.RNN(hidden_size=self.rnn_hiddens, activation=self.activation,
                           i2h_weight_initializer=self.weight_initializer,
                           h2h_weight_initializer=self.weight_initializer,
                           i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'gru': rnn.GRU(hidden_size=self.rnn_hiddens,
                           i2h_weight_initializer=self.weight_initializer,
                           h2h_weight_initializer=self.weight_initializer,
                           i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer),
            'lstm': rnn.LSTM(hidden_size=self.rnn_hiddens,
                             i2h_weight_initializer=self.weight_initializer,
                             h2h_weight_initializer=self.weight_initializer,
                             i2h_bias_initializer=self.bias_initializer, h2h_bias_initializer=self.bias_initializer)
        }
        self.scratch_selector = {
            'rnn': Rnn(self.input_dim, self.rnn_hiddens, self.output_dim, self.batch_size, self.ctx,
                       self.weight_initializer, self.bias_initializer),
            'gru': Gru(self.input_dim, self.rnn_hiddens, self.output_dim, self.batch_size, self.ctx,
                       self.weight_initializer, self.bias_initializer),
            'lstm': Lstm(self.input_dim, self.rnn_hiddens, self.output_dim, self.batch_size, self.ctx,
                         self.weight_initializer, self.bias_initializer)

        }
        self.randomIterIsOn = self.gConfig['randomIterIsOn']
        self.get_net()
        self.net.initialize(ctx=self.ctx)
        # 该处的clip_gradient没有起到效果
        self.trainer = gluon.Trainer(self.net.collect_params(), self.optimizer,
                                     {'learning_rate': self.learning_rate, 'clip_gradient': self.clip_gradient})
        self.input_shape = (self.resizedshape[0], self.batch_size, self.resizedshape[1])


    def get_net(self):
        input_dim = self.gConfig['input_dim']#1
        input_dim = self.vocab_size
        rnn_hiddens =self.gConfig['rnn_hiddens'] #256
        output_dim = self.gConfig['output_dim']#1
        output_dim = self.vocab_size
        activation = self.get_activation(self.gConfig['activation'])
        cell = self.get_cell(self.gConfig['cell'])

        if self.scratchIsOn == True:
            self.net = self.scratch_selector[cell]
        else:
             rnn_layer = self.cell_selector[cell]
             self.net = RNN(rnn_layer,rnn_hiddens,self.vocab_size)


    def init_state(self):
        self.state = self.net.begin_state(batch_size=self.batch_size, ctx=self.ctx)


    def run_train_loss_acc(self, X, y):
        if self.randomIterIsOn == True:
            self.init_state()
        else:
            for s in self.state:
                s.detach()
        with autograd.record():
            y_hat,self.state = self.net(X,self.state) #self.state在父类中通过init_state来初始化
            batch_size = y.shape[0]
            n = batch_size / y.size
            y = y.T.reshape((-1,)) #y.shape = (batch_size,time_steps),y.T.reshape((-1)).shape=(time_steps*batch_size,)
            loss = self.loss(y_hat, y).mean()
        loss.backward()
        if self.global_step == 0:
            self.debug_info()
        #self.trainer.step(self.batch_size)
        params = [p.data() for p in self.net.collect_params().values()]
        self.grad_clipping(params, self.clip_gradient, self.ctx)
        self.trainer.step(batch_size=1) #在loss采用mean后，batch_size相应的改成１
        loss = loss.asscalar() * batch_size
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        return loss, acc * n


    def run_eval_loss_acc(self, X, y):
        self.init_state() #对于测试来说，因为没有反向传播，每个time_step,batch_size的数据都要初始化状态
        y_hat,self.state = self.net(X,self.state)
        batch_size = y.shape[0]
        n = batch_size / y.size
        y = y.T.reshape((-1,))
        loss = self.loss(y_hat, y).mean()
        loss = loss.asscalar() * batch_size
        acc = (y_hat.argmax(axis=1) == y).sum().asscalar()
        return loss, acc * n


    def run_matrix(self, loss_train, loss_test):
        #rnn中用perplexity取代accuracy
        perplexity_train = math.exp(loss_train)
        perplexity_test = math.exp(loss_test)
        print('global_step %d, perplexity_train %.6f,perplexity_test %f.6'%
              (self.global_step.asscalar(), perplexity_train,perplexity_test))
        return perplexity_train,perplexity_test


    def get_input_shape(self):
        return self.input_shape


    def get_cell(self,cell):
        assert cell in self.gConfig['celllist'], 'cell(%s) is invalid,it must one of %s' % \
                                                               (cell, self.gConfig['celllist'])
        return cell


    def predict_nlp(self, model):
        for prefix in self.prefixes:
            print(' -', self.predict_rnn_gluon(
                prefix, self.predict_length, model, self.vocab_size, self.ctx, self.idx_to_char,
                self.char_to_idx))


    def predict_rnn_gluon(self,prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                          char_to_idx):
        # 使用model的成员函数来初始化隐藏状态
        state = model.begin_state(batch_size=1, ctx=ctx)
        output = [char_to_idx[prefix[0]]]
        for t in range(num_chars + len(prefix) - 1):
            X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
            X = nd.one_hot(X,vocab_size)
            (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
            if t < len(prefix) - 1:
                output.append(char_to_idx[prefix[t + 1]])
            else:
                output.append(int(Y.argmax(axis=1).asscalar()))
        return ''.join([idx_to_char[i] for i in output])


    def show_net(self,input_shape = None):
        if self.viewIsOn == False:
            return
        #print(self.net)
        #title = self.gConfig['taskname']
        title = self.gConfig['model']
        input_symbol = mx.symbol.Variable('input_data')
        if self.scratchIsOn:
            print('采用自定义Rnn,Lstm,Gru模型时，mx.viz.plot_network当前不可用！')
            state_symbol = self.net.get_symbol_state()
            net,state = self.net(input_symbol,state_symbol)
        else:
            state_symbol = mx.symbol.Variable('state')
            net,state = self.net(input_symbol,state_symbol)
            mx.viz.plot_network(net, title=title, save_format='png', hide_weights=False,
                                shape=input_shape) \
                    .view(directory=self.loggingspace.directory)
        return


    def apply_model(self,net):
        self.logger.info('rnnModelM.apply_model has not be implement!')


    def summary(self):
        self.init_state()
        self.net.summary(nd.zeros(shape=self.get_input_shape(), ctx=self.ctx),
                         self.state)


    def hybridize(self):
        if self.scratchIsOn:
            print('当使用自定义模型时，hybridize()函数无法使用')
            pass
        else:
            self.net.hybridize()


    def grad_clipping(self,params, theta, ctx):
        """Clip the gradient."""
        if theta is not None:
            norm = nd.array([0], ctx)
            for param in params:
                norm += (param.grad ** 2).sum()
            norm = norm.sqrt().asscalar()
            if norm > theta:
                for param in params:
                    param.grad[:] *= theta / norm


def create_object(gConfig):
    model=rnnModel(gConfig=gConfig)
    #model.initialize(ckpt_used)
    return model