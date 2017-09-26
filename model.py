import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os

def train_loop(param,model):
    data_manager = DataManager(param)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        writer = tf.summary.FileWriter("log/", graph = sess.graph)
        sess.run(init)



        for i in range(param.epoch_num):

            print('start of epoch  %d '%(i))

            train_generator,num_batch_train = data_manager.train_batches()


            state = None
            loss_list = []
            for X_batch,y_batch in tqdm(train_generator):
                loss, next_state , _ =  model.step(sess,X_batch,y_batch,state)
                loss_list.append(loss)
                state = next_state
            print( 'training loss %.3f'%( np.mean(loss_list) ) )

            if i%param.val_per_epcoh_num == 0:
                #validation
                #做validation好像有點奇怪... 不知道state要不要繼續傳下去= =
                #應該是要吧
                state = None
                loss_list = []

                accuracy_list = []
                val_generator,_ = data_manager.val_batches()

                for X,y in tqdm(val_generator):
                    loss,accu,next_state = model.get_loss_and_accuracy(sess,X,y,state)
                    loss_list.append(loss)
                    accuracy_list .append(accu)
                    state = next_state

                print('validation loss %.3f'%(np.mean(loss_list) ))
                print('validation accuracy %.3f'%(np.mean(accuracy_list) ))


            if  i%param.save_per_epcoh_num == 0:
            #save
                path = os.path.join(param.save_path, "model.ckpt")
                print("Saving the model at Epoch %i." %(i))
                model.saver.save(sess, path,
                global_step=model.global_step)
            #sample...

            if  i%param.sample_per_epcoh_num == 0:
                print("sampling at Epoch %i." %(i))
                samples = model.sample_precess(sess,param.sample_num,param.sample_max_len,param.id2char)
                for si,sample in enumerate(samples):
                    path = os.path.join(param.sample_save_path,"sample_ep%d_%d.txt"%(i, si))
                    with open(path,"w",encoding="utf-8") as f:
                        f. write(sample)








def build_rnn_model(param):
    return RNN_Model(param)


class RNN_Model(object):
    def __init__(self,param):
        self.embedding_dim = param.embedding_dim
        self.hidden_num = param.hidden_num
        self.output_dim = param.vocab_size


        vocab_size = param.vocab_size
                                                #         batch_size,max_len_seq of ids

        self.X = tf.placeholder(tf.int32, (None,None))
        self.y = tf.placeholder(tf.int32, (None, None))

        #with tf.variable_scope('embedding', reuse=True):
        self.embedding_matrix = tf.Variable(tf.random_normal((vocab_size,self.hidden_num)), name = "W")
        # Now Dimension is (batch size ,max_seq_len,embedding_dim)

        '''
         tf.nn.embedding_lookup可以用多維度
         a = tf.constant([[1, 3], [5, 7],[1.1,1.23]])
         b  = tf.nn.embedding_lookup(a,[[1,2],[0,1]])
        with tf.Session() as sess:
        print(sess.run(b))
            [[[ 5.          7.        ]
              [ 1.10000002  1.23000002]]

             [[ 1.          3.        ]
              [ 5.          7.        ]]]
        '''

        #獲得動態的dimention
        batch_size = tf.shape(self.X)[0]
        #tf.shape(self.X)[1] 是tensorf 因為要在loop用所以加上

        self.max_seq_len =  tf.shape(self.X)[1]

        '''
        dynamic rnn的sequence_lengths參數可以讓每一個batch的seqence數量不同
        with tf.Session() as sess:
            t= tf.tile([3],[3])
            print(sess.run(t))
        '''
        sequence_lengths = tf.tile([param.max_seq_len],[batch_size])
        self.rnn_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.X)
        self.rnn_cell = self.rnn_cell()
        self.initial_state = self.rnn_cell.zero_state(batch_size,tf.float32)
        # <batch_size, max_time, hidden_num>,<batch_size,hidden_num>
        self.lstm_h,self.final_state = tf.nn.dynamic_rnn(self.rnn_cell, self.rnn_inputs  ,sequence_length=sequence_lengths,
            initial_state=self.initial_state)



        #dont reshapes first
        #要輸出 可是維度是三維,本來的解法應該是要把它reshape然後接起來之類的
        self.W_softmax =  tf.Variable(tf.random_normal((self.hidden_num, vocab_size)), name = "W_softmax")
        self.b_softmax = tf.Variable(tf.constant( vocab_size,dtype=tf.float32), name = "b_softmax")
        # each time step output store as list
        a =  tf.Variable([[]],dtype=tf.float32)
        self.loss_list =   tf.Variable([],dtype=tf.float32)
        self._prob_list =           tf.reshape(a,(batch_size,0))

        cnt = tf.constant(0)
        cond  = lambda cnt,loss_list,prob_list: tf.less(cnt, self.max_seq_len )

        def loop_body(cnt,loss_list,prob_list):
            #squeeze 把dimension的大小為1的降維
            time_output = tf.squeeze(tf.slice(self.lstm_h, tf.stack([0,cnt, 0]), [-1,1,-1]),[1])
            #<batch_size,vocab_size>
            logits =  tf.matmul(time_output, self.W_softmax) + self.b_softmax
            #  定義 softxmax ..輸出每一個timestep的機率比較方便預測和採樣
            prob= tf.nn.softmax(logits)
            labels= tf.squeeze(tf.slice(self.y, tf.stack([0,cnt]), [-1,1]),[1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
            prob_list =  tf.concat( [prob_list, prob],1)
            loss_list =  tf.concat( [loss_list, loss],0)
            return  cnt+1,loss_list,prob_list
        #tensorflow規定每一個當作參數的tensor的shape在while內不能變動,如果要變動必須要用None
        # TODO
        d = self._prob_list.get_shape().as_list()
        d[1]=None
        magic_shape = tf.TensorShape(d)
        cnt,self.loss_list,self.prob_list = tf.while_loop(cond, loop_body, [cnt,self.loss_list ,self._prob_list ],
                          shape_invariants=[cnt.get_shape(),tf.TensorShape([None]),magic_shape ])
        #https://stackoverflow.com/questions/40432289/tensorflow-reshape-with-variable-length
        self.reshaped_prob_tensor  = tf.reshape(self.prob_list, (batch_size,self.max_seq_len,vocab_size))



        self.prediction = tf.argmax(self.reshaped_prob_tensor , axis=2)
        self.reshaped_loss_list  = tf.reshape(self.loss_list, (batch_size,self.max_seq_len))
        #self.reshaped_loss_list = tf.Print(self.reshaped_loss_list ,[self.reshaped_loss_list ],"losses"
                                           #,summarize=61,first_n=10)
        #全部都一起拿進來算gradient會和本來的每一筆batch加起來之後平均算gradient
        #在反向傳播上應該是有區別的
        #

        self.total_loss = tf.reduce_mean(tf.reduce_sum(self.reshaped_loss_list,axis=1))

        #self.total_loss = tf.Print(self.total_loss ,[self.total_loss ],"loss")
        #print( self.total_loss.get_shape())
        equality = tf.equal(tf.cast(self.prediction, tf.int32), self.y)


        self.accuracy =  tf.reduce_mean(tf.cast(equality, tf.float32))




        #輸出每一個timestep的loss
        '''
            with tf.Session() as sess:
                y = np.array([[1,0],[0,2]])
                logits = np.array([[[1,1,2],[0,0,1]],[[1,2,1],[1,0,0]]],dtype=np.float32)
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y[:,0],logits=logits[:,0])
                print(sess.run(loss))
            [ 1.55144477  1.55144465]
            import math
            p = np.exp(logits[0,0])
            p = p/np.sum(p)
            print(math.log(0.21194156))
            -1.5514447026887888
        '''




        #  optimizition
        self.lr = tf.Variable(param.lr, trainable=False)
        self.trainable_vars = tf.trainable_variables()
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.global_step = tf.Variable(0, trainable=False)
        self.train_step = self.optimizer.minimize(self.total_loss,global_step=self.global_step)
        #       saver
        self.saver = tf.train.Saver(tf.all_variables())


    def rnn_cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_num)
        return cell
    def generate_input_feed(self,X,y=None,state=None):
        feed  = {self.X:X}
        if  y is not None:
            feed[self.y] = y
        if state is not  None:
            feed[self.initial_state] = state
        return feed

    def step(self,sess,batch_X,batch_y,initial_state=None):
        input_feed = self.generate_input_feed(batch_X,batch_y,initial_state)
        output_feed = [self.total_loss,
                       self.final_state,
                       self.train_step]
        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]


    def prob_each_timestep(self,sess,batch_X,state=None):
        input_feed = self.generate_input_feed(batch_X,None,state)
        #
        # shape of prob_each_time_step = sess.run(self.prob_list, input_feed )
        # is <max_seq_len,batch num ,vocab_size>
        # chage to <batch num,max_seq_len,vocab_size>
        #prob_each_time_step = sess.run(, input_feed )

        '''
        import tensorflow as tf

        a = tf.constant(1)
        b = tf.constant(2)
        c = tf.constant(3)

        l = [a,b,c]

        ab = tf.add(a,b)

        with tf.Session() as sess:
            res,g = sess.run((l,ab))
            print(res)
             print(g)

        output -- >
        [1, 2, 3]
        3
        '''
        prob_each_time_step,final_state = sess.run((self.prob_tensor,self.final_state), input_feed )
        return (prob_each_time_step,final_state)

    def predict_output(self,sess,batch_X,state=None):
        '''
        with tf.Session() as sess:
        #get accuracy
        prob = [[[0.1,0.9],[0.7,0.3],[0.2,0.8],[0.51,0.49]],
                [[0.9,0.1],[0.7,0.3],[0.2,0.8],[0.49,0.51]]
               ]
        correct_answer =[[0,0,1,0],[0,0,1,1]]
        prediction = tf.argmax(prob, 2)
        print(sess.run(prediction))
        equality = tf.equal(prediction, correct_answer)
        print(sess.run(equality))
        accuracy = tf.reduce_mean(tf.cast(equality, tf.float32),axis=1)
        print(sess.run(accuracy))

        output-->
        [[1 0 1 0]
         [0 0 1 1]]

        [[False  True  True  True]
         [ True  True  True  True]]

        [ 0.75  1.  ]
        '''
        input_feed = self.generate_input_feed(batch_X,None,state)
        predictions,next_state = sess.run((self.prediction,self.final_state),input_feed)
        return predictions,next_state

    def sample_precess(self,sess,sample_num,sample_len,id2char):
        #random generate a char
        X =  sess.run(tf.random_uniform( (sample_num,1),0,len(id2char),tf.int32))
        state = None
        sample_output = []
        for i  in range(sample_len):
            s,next_state  = self.predict_output(sess,X,state)
            sample_output.append(np.squeeze(s))
            X,state = s,next_state
        samples =np.array(sample_output).T.tolist()


        res = []
        for i,s in enumerate(samples):
            si = []
            for j,c in enumerate(s):
                si.append(id2char[c])
            res.append("".join(si))
        return  res


    #利用這一個function算一個batch的正確率
    #如果要算整個epoch 就把state往下傳 (這是model)的邏輯
    #因為validation set是文章的後面部分 ,一個batch和一個batch之間是有關聯的
    def get_loss_and_accuracy(self,sess,batch_X,batch_y,state=None):
        loss,accuracy,next_state = sess.run((self.total_loss,self.accuracy,self.final_state),
                        self.generate_input_feed(batch_X,batch_y,state))

        #正確率的算法 ,
        # 一個batch 有N(batch_size  有 max_seq_len個要預測的東西 N*max_seq_len
        # 看 N*max_seq_len 裡面中了幾個
        # 正確率就是多少
        # 這樣比較符合邏輯
        #因為之後的train loop 內會把每一個batch的正確率的平均當作全部的正確率
        #而每一個batch的大小又是一樣大的 所以這樣算是合理的
        #用numpy做似乎會降低速度...
        #實驗一下
        '''
        equality = np.equal(preds, batch_y)

        accuracy =  np.mean(equality)
        這個超級久
        '''
        return loss,accuracy,next_state










#如果generator拿來產生train set
def generate_batches(X,y,batch_num,batch_size,max_seq_len):
        def generate_one_batch(batch_idx):
            Xs = []
            ys = []
            start_idx = batch_idx*batch_size*max_seq_len
            for j in range(batch_size):
                Xs.append(X[start_idx+j*max_seq_len:start_idx+(j+1)*max_seq_len])
                ys.append(y[start_idx+j*max_seq_len:start_idx+(j+1)*max_seq_len])
            return Xs,ys

        for i in range(batch_num):
            yield generate_one_batch(i)




class DataManager(object):
    def __init__(self,param,train_rate=0.8,val_rate=0.2):
        self.X = param.X
        self.y = param.y

        char_num = len(param.X)

        self.batch_size = param.batch_size
        self.max_seq_len = param.max_seq_len
        self.batch_num = (char_num//self.max_seq_len)//self.batch_size


        train_char_num = int(char_num*train_rate)
        self.train_batch_num = (train_char_num//self.max_seq_len)//self.batch_size

        val_char_num =   int(char_num*val_rate)
        self.val_batch_num = (val_char_num//self.max_seq_len)//self.batch_size


        self.train_X = np.array(self.X[0:train_char_num])
        self.train_y = np.array(self.y[0:train_char_num])


        self.val_X = np.array(self.X[train_char_num:train_char_num+val_char_num])
        self.val_y = np.array(self.y[train_char_num:train_char_num+val_char_num])




    def train_batches(self):
        return  generate_batches(   self.train_X,
                                                    self.train_y,
                                                    self.train_batch_num,
                                                    self.batch_size,
                                                    self.max_seq_len), self.train_batch_num

    def val_batches(self):
        return  generate_batches(   self.val_X,
                                                    self.val_y,
                                                    self.val_batch_num ,
                                                    self.batch_size,
                                                    self.max_seq_len),self.val_batch_num


class Parameter(object):
    def __init__(self,datapath):
        #model paramter
        self.embedding_dim = 100
        self.hidden_num = 50

        #data parameter
        self.max_seq_len = 4
        self._data_info(datapath)

        #training parameter
        self.epoch_num = 200
        self.batch_size = 32
        self.val_per_epcoh_num = 5
        self.lr  = 0.01

        #save parameter
        self.save_path = './save'
        self.save_per_epcoh_num = 5

        # sample parameter
        self.sample_max_len = 200
        self.sample_num = 3
        self.sample_per_epcoh_num = 5
        self.sample_save_path = './gen'




    def _data_info(self,path):
        with open(path,"r",encoding="utf-8") as f:
            s = f.read()
            self.vocab = set(s)
            print("Vocab size %d"%(len(self.vocab) ) )

            self.id2char = list(self.vocab)
            self.char2id = { c:i  for i,c in enumerate(self.id2char)}

            self.vocab_size = len(self.vocab)

            self.X = [self.char2id[c] for c in s]
            self.y = self.X[1:]+[None]













