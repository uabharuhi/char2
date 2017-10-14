import tensorflow as tf
import model
import numpy as np

DATAPATH = './data/data.txt'
parameter = model.Parameter(DATAPATH)
data_manager = model.DataManager(parameter)
rnn_model = model.RNN_Model(parameter)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state(parameter.save_path)
  if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      #print('reload successfully')
      #train_generator,num_batch_train = data_manager.train_batches()
      #print(parameter.X[0:100])
      #test,testy = next(train_generator)
      #print(test)
      #print(testy)
      #s,next_state = rnn_model.predict_output(sess,test)
      #print('- - - - - - - - - -- - - - - - ')
      #print(s)
      #gg = s==testy
      #print(gg)
      #print(np.mean(s==testy))
      #print('accuracy')
      #print(sess.run(rnn_model.accuracy,{rnn_model.X:test,rnn_model.y:testy}))
      #測試一個一個丟進去和直接丟一個batch內的數量(4個)字元結果是否相同
      #測試結果不同... 因為predict的時候 是拿X當作下個時間點的input..
      #而一個一個是拿上一個是拿上一個時間點的輸出當作input
      #一旦上一個時間點錯誤 基本上就會錯.....
      #
      #X = [[15]]
      #state = None
      #l=[]
      #s,next_state = rnn_model.predict_output(sess,[[1 ,50, 47, 36 ]])
      #print(s)
      #s,next_state = rnn_model.predict_output(sess,[ [ 1 ,49 ,32 ,51]])
      #print(s)
      #strr=parameter.id2char[15]
      #for i in range(100):
      #  s,next_state = rnn_model.predict_output(sess,X,state)
      #  strr+= parameter.id2char[s[0][0]]
      #  X,state = s,next_state
      #print(strr)
      model.sample2file(sess,rnn_model,"GG",parameter,random=True)

#關於batch的問題
#第一個問題
#每一個batch裡面,把它切成很多等份,這些等份是有前後關係的
#但是他們卻是使用前一個batch傳過來的state-->他們之前並不像前後文那樣具有強烈的關係
#所以就算訓練成功 sample出來的東西也很奇怪....
#要成功sample 可能要把batch弄成一排一排的
#本來是 [1,2,3,4],[5,6,7,8]在同一個batch
#現在要弄成把他們拆成[1,2,3,4]在第一個batch [5,6,7,8]在第二個batch

#第二個問題
#第一組的初始state,也就是state=None的時候,那個時候state是zero
#大家的state都是zero,
#而batch裡面 有一些是 [1,2,3,4] 有一些是 [1,7,8,3]
#然後因為第一個state大家都是zero第一個input大家都是1
#LSTM給定這兩個東西之後,基本上只能輸出一種值 當然沒辦法滿足所有可能
#所以而當地二個batch進來的時候 因為dynamic_rnn會傳回一個 <bach_size,hidden_num>維度的state
#每一個batch接收到的state的值不像第一個batch一樣都是相同的
#當model capacity夠大的時候就有辦法滿足所有可能 所以之後正確率才都會是一只有第一個batch沒有1


#First Citizen: 15, 40, 49, 50, 51, 1, 12, 40, 51, 40,
#1 50 47 36 1 50 47 36 0 15 43 43

