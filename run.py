
import model

DATAPATH = './data/data.txt'
parameter = model.Parameter(DATAPATH)
data_manager = model.DataManager(parameter)
rnn_model = model.RNN_Model(parameter)
model.train_loop(parameter,rnn_model)
