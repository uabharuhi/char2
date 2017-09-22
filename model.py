import tensorflow as tf

class RNN_Model(object):
    def __init__(self):
        pass
    def build(self,param):
        self.embedding_dim = param.embedding_dim
        self.hidden_num = param.hidden_num
        self.output_dim = param.vocab_size
        self.max_seq_len = param.max_seq_len

        vocab_size = param.vocab_size
                                                #         batch_size
        self.X = tf.placeholder(tf.int32, (None, None))
        self.y = tf.placeholder(tf.int32, (None, None))

        #with tf.variable_scope('embedding', reuse=True):
        self.embedding_matrix = tf.get_variable("W", (vocab_size,self.embedding_dim))
        self.rnn_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.X)
        self.rnn_cell = self.rnn_cell()
        self.initial_state = self.rnn_cell.zero_state()
        # <batch_size, max_time, hidden_num>,<batch_size,hidden_num>
        self.lstm_h,self.final_state = tf.nn.dynamic_rnn( self.rnn_inputs, self.max_seq_len,
            initial_state=self.initial_state)

        #dont reshapes first
        #要輸出 可是維度是三維,本來的解法應該是要把它reshape然後接起來之類的
        self.W_softmax = tf.get_variable("W_softmax",
                (self.max_seq_len, vocab_size ))
        self.b_softmax = tf.get_variable("b_softmax", (vocab_size))
        # each time step output store as list
        self.logits_list  = []

        for i in range(self.max_seq_len):
            time_output = self.lstm_h[:,i,:]
            #<batch_size,vocab_size>
            logits =  tf.matmul(time_output, self.W_softmax) + self.b_softmax
            self.logits_list.append(logits)

        self.lost_list = []

        for i in range(self.max_seq_len):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y[:,i],logits=self.logits_list[i])
            self.lost_list .append( loss)

        self.total_lost = tf.constant(0.0)

        for loss in self.lost_list:
            self.total_lost += loss
        #  optimizition
        self.lr = tf.Variable(0.001, trainable=False)
        self.trainable_vars = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_step = self.optimizer.minimize(self.total_lost)
        #       saver
        #self.global_step = tf.Variable(0, trainable=False)
        #self.saver = tf.train.Saver(tf.all_variables())
        #rnn cell

    def rnn_cell(self):
        cell = tf.nn.BasicLSTMCell(self.hidden_num)
        return cell

    def step(self,sess,batch_X,batch_y,initial_state=None):
        if initial_state==None:
            input_feed = {self.X: batch_X,
                              self.targets: batch_y}
        else:
              input_feed = {self.X: batch_X,
                              self.targets: batch_y,
                              self.initial_state: initial_state}

        output_feed = [self.total_loss,
                       self.final_state,
                       self.logits,
                       self.train_optimizer]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2], outputs[3]


class Parameter(object):
    def __init__(self,datapath):
        self.embedding_dim = 100
        self.hidden_num = 50
        self.max_seq_len = 4
        self.epoch_num = 100
        self.batch_size = 32

    def generate_data(self,path):
        with open(path,"r",encoding="utf-8") as f:
            s = f.read()
            self.vocab = set(s)
            self.id2char = list(self.vocab)
            self.char2id = { c:i  for i,c in enumerate(self.id2char)}

            self.vocab_size = len(self.vocab)

            self.X = [self.char2id[c] for c in s]
            self.y = [None]+self.X

    def generate_batches(self):
        total_char_num = len(self.X)
        batch_num = (total_char_num//self.max_seq_len)//self.batch_size









