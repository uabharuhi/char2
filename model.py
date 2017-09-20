import tensorflow as tf
import numpy as np

class RNN_Model(object):
    def __init__(self,param):
        self.param  = param
        self.hidden_num = hidden_num
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
    def build(self,ids,embedding_dim,vocab_size):
        #with tf.variable_scope('embedding', reuse=True):
        embedding_matrix = tf.get_variable("W", [vocab_size,embedding_dim])
        rnn_inputs = tf.nn.embedding_lookup(embedding_matrix, ids)
        rnn_cell = self.rnn_cell()
        unfold_rnn = tf.nn.dynamic_rnn(inputs, self.max_seq_len, 
            initial_state=rnn_cell.zero_state())

        #rnn cell
    def rnn_cell(self):
        cell = tf.nn.BasicLSTMCell(self.hidden_num)
        return cell 