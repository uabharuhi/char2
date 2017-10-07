import tensorflow as tf
import model
import numpy as np
import os
DATAPATH = './data/small2.txt'
parameter = model.Parameter(DATAPATH)
sample_num = 3
sample_len = 400
#data_manager = model.DataManager(parameter)

def random_sample_precess(self,sess,sample_num,sample_len,id2char):
  #random generate a char
  X =  sess.run(tf.random_uniform( (sample_num,1),0,len(id2char),tf.int32))
  state = None
  sample_output = []
  for i  in range(sample_len):
      prob,next_state  = self.prob_each_timestep(sess,X,state)
      print(prob.shape)
      X = []
  return []
#      sample_output.append(np.squeeze(s))
#      X,state = s,next_state
#  samples =np.array(sample_output).T.tolist()
#  res = []
#  for i,s in enumerate(samples):
#    si = []
#    for j,c in enumerate(s):
#      si.append(id2char[c])
#      res.append("".join(si))
#  return  res


with tf.Session() as sess:
  meta = tf.train.import_meta_graph('./save/model.ckpt-19380.meta')
  rnn_model = model.RNN_Model(parameter)
  sess.run(tf.global_variables_initializer())
  meta.restore(sess, tf.train.latest_checkpoint('./save'))

  samples = random_sample_precess(rnn_model,sess,sample_num,sample_len,parameter.id2char)
#  for si,sample in enumerate(samples):
#    path = os.path.join(parameter.sample_save_path,"QQ %d.txt"%(si))
#    with open(path,"w",encoding="utf-8") as f:
#      f.write(sample)






#  graph = tf.get_default_graph()
#  X_placeholder = graph.get_operation_by_name('Placeholder')
#  prob = graph.get_operation_by_name('Reshape_1')
#
#  X =  sess.run(tf.random_uniform( (sample_num,1),0,len(parameter.id2char),tf.int32))
#  state = None
#  sample_output = []
#  for i  in range(sample_len):
#      s,next_state  = self.predict_output(sess,X,state)
#      sample_output.append(np.squeeze(s))
#      X,state = s,next_state
#      samples =np.array(sample_output).T.tolist()
#  #sess
#
#
#def random_sample_precess(self,sess,sample_num,sample_len,id2char):
#    #random generate a char
#    X =  sess.run(tf.random_uniform( (sample_num,1),0,len(id2char),tf.int32))
#    state = None
#    sample_output = []
#    for i  in range(sample_len):
#        s,next_state  = self.predict_output(sess,X,state)
#        sample_output.append(np.squeeze(s))
#        X,state = s,next_state
#    samples =np.array(sample_output).T.tolist()
#
#
#    res = []
#    for i,s in enumerate(samples):
#        si = []
#        for j,c in enumerate(s):
#            si.append(id2char[c])
#        res.append("".join(si))
#    return  res
#
