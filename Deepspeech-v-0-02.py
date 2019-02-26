import numpy as np
import tensorflow as tf
import soundfile as sf
import sys
import timit_load_data
import tools


class DeepSpeech(object):

  def __init__(model, h1_size, h2_size, h3_size, h4_size, h5_size, input_size):
    model.h1_size, model.h2_size, model.h3_size, model.h4_size, model.h5_size = h1_size, h2_size, h3_size, h4_size, h5_size
    model.input_size = input_size
    model.relu_clip = 20
    ##Hyper Parameters
    model.learning_rate = 0.001
    model.beta1 = 0.9
    model.beta2 = 0.999
    model.epsilon = 1e-8

  def load_timit_ctc_data(model, filename):
    audiopath_full = "timit/"+filename+".wav"
    phonemepath_full = "timit/"+filename+".phn"
    #print(audiopath_full)
    audio, text_idx = timit_load_data.get_ctc_data(audiopath_full , phonemepath_full)
    audio = tools.window_data(audio)
    #print("text", text)
    #text = tools.text_to_onehot(text,len(timit_load_data.phones))
    #print("text reconstructed", tools.onehot_to_text(text))
    return audio,text_idx


  def init_var(model):

    model.x = tf.placeholder("float", shape=[None,model.input_size])
    print(model.x.shape)
    ##He initialisation
    model.w1 = tf.Variable(tf.random_normal([model.input_size, model.h1_size], stddev = tf.sqrt(2/model.input_size)), dtype=tf.float32)
    model.b1 = tf.Variable(tf.zeros([model.h1_size]), dtype=tf.float32)
    
    model.w2 = tf.Variable(tf.random_normal([model.h1_size, model.h2_size], stddev = tf.sqrt(2/model.h1_size)), dtype=tf.float32)
    model.b2 = tf.Variable(tf.zeros([model.h2_size]), dtype=tf.float32)

    model.w3 = tf.Variable(tf.random_normal([model.h2_size, model.h3_size], stddev = tf.sqrt(2/model.h2_size)), dtype=tf.float32)
    model.b3 = tf.Variable(tf.zeros([model.h3_size]), dtype=tf.float32)
    
    model.w4 = tf.Variable(tf.random_normal([model.h3_size, model.h4_size], stddev = tf.sqrt(2/model.h3_size)), dtype=tf.float32)
    model.b4 = tf.Variable(tf.zeros([model.h4_size]), dtype=tf.float32)

    model.w5 = tf.Variable(tf.random_normal([model.h4_size, model.h5_size], stddev = tf.sqrt(2/model.h4_size)), dtype=tf.float32)
    model.b5 = tf.Variable(tf.zeros([model.h5_size]), dtype=tf.float32)

    #model.y = tf.placeholder("float", shape=[None,model.h5_size])
    model.y = tf.sparse_placeholder(tf.int32)
    model.seq_len = tf.placeholder(shape=[None], dtype=tf.int32)


    

  def BiRnn(model,x, n_steps, batch_size=None, previous_state=None):
    if (batch_size == None):
      batch_size = tf.shape(x)[0]
    
    h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(x, model.w1), model.b1)), model.relu_clip)
    h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1, model.w2), model.b2)), model.relu_clip)
    h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2, model.w3), model.b3)), model.relu_clip)

## debug
##    init = tf.global_variables_initializer()
##    sess = tf.Session()
##    sess.run(init)
##    print(sess.run(tf.shape(h3)))

    LSTM = False
    #lstm = tf.nn.rnn_cell.LSTMCell(10)
    #state = lstm.zero_state(10, dtype=tf.float32)

    #for i in range(seq_length):
    ##hf_output, state = lstm(h3[0],None)   

    if(LSTM == True): ##unrolled biLSTM not working
      print("TRUe")
      fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(2000, reuse=False)
      bw_cell = tf.contrib.rnn.LSTMBlockFusedCell(2000, reuse=False)
      h3_fw = tf.reshape(h3, [n_steps, batch_size, model.h3_size])
      h3_bw = tf.reshape(tf.reverse(h3, axis=0), [n_steps, batch_size, model.h3_size])
      
      ##print(sess.run(tf.shape(h3)))
      hf_output, hf_output_state = fw_cell(inputs=h3_fw, dtype=tf.float32)#, sequence_length=seq_length, initial_state=previous_state)
      ##print(sess.run(tf.shape(hf_output)))

      hb_output, hb_output_state = fb_cell(inputs=h3_bw, dtype=tf.float32)
      ##print(sess.run(tf.shape(hb_output)))

    else: ##dynamic biRNN current choice 
      print("FALSE")
      h3_reshape = tf.reshape(h3, [n_steps, batch_size, model.h3_size])
      cell_fw = tf.nn.rnn_cell.BasicRNNCell(model.h3_size)
      cell_bw = tf.nn.rnn_cell.BasicRNNCell(model.h3_size)
      outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = cell_fw,
        cell_bw = cell_bw,
        dtype=tf.float32,
        inputs = h3_reshape)

      output_fw, output_bw = outputs
      states_fw, states_bw = states
      output_fw = tf.minimum(tf.nn.relu(output_fw), model.relu_clip)
      output_bw = tf.minimum(tf.nn.relu(output_bw), model.relu_clip)
      ##print(sess.run(tf.shape(output_fw)))
      ##print(sess.run(tf.shape(output_bw)))
      h4 = tf.reshape(tf.add(output_fw,output_bw), [batch_size,model.h3_size])

    
    h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h4, model.w5), model.b5)), model.relu_clip)
    h6 = tf.nn.softmax(h5)

    ##print(sess.run(tf.shape(h5)))
    return h5

  

  def train(model, load_data_function, filenames_list):
    ''' trains a single data sample at a time

        load_data_function must return:
        audio input data - batch_size x frame_size
        y - onehot encoded label

        filenames_list must be a list of str valued filepaths
    '''

    model.init_var()
    
    y_pred = model.BiRnn(model.x,n_steps=-1,batch_size=None)
    y_pred = tf.reshape(y_pred, [-1,tf.shape(y_pred)[0],tf.shape(y_pred)[1]])
    print("y", tf.shape(model.y), model.y)
    print("seq_len", tf.shape(y_pred)[-1])
    loss = tf.nn.ctc_loss(labels=model.y,inputs=y_pred, sequence_length=model.seq_len)
    
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
    #                    labels=model.y, logits=y_pred)) 
    print("loss shape",loss.shape)
    optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate,
                                       beta1=model.beta1, beta2=model.beta2, epsilon=model.epsilon).minimize(loss)

    #test_pred = tf.nn.ctc_beam_search_decoder(y_pred, model.seq_len, merge_repeated=True)
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    #print("input data shape", sess.run(tf.shape(input_data)))
    sess.run(init)

    for epoch in range(1):
      for i in range(len(filenames_list)):
        print("i:",i)
        audio_input_data, y = load_data_function(filenames_list[i])
        y = y.reshape(y.shape[0],1)
        values = np.ones_like(y)
        indices = np.concatenate((np.zeros_like(y),np.arange(y.shape[0]).reshape(y.shape[0],1)), axis=1)
        print(indices)
        
##        if(i>=150):
##          print(y.shape)
##          print(audio_input_data.shape)
        _,l = sess.run([optimizer,loss], feed_dict = {model.x: audio_input_data, model.y: tf.SparseTensorValue(indices, values[:,0], np.array([1,y.shape[0]])), model.seq_len : (np.ones(shape=(audio_input_data.shape[0]))*y.shape[0])})
        #_,l = sess.run([optimizer,loss], feed_dict = {model.x: audio_input_data, model.y: tf.SparseTensorValue(y, np.ones_like(y[:,0]), np.array([len(timit_load_data.phones),1])), model.seq_len : [audio_input_data.shape[0]]})
        
      #if (epoch % 10 == 0):
        #print('Step %i: Loss: %f' % (epoch, l))
      print("epoch ", epoch, " Loss ", l)

    #batch_size = audio_input_data.shape[0]
    #seq_len = tf.to_int32(tf.fill([batch_size], 62))
    #test_pred = sess.run(tf.nn.ctc_beam_search_decoder(tf.nn.softmax(y_pred), model.seq_len, merge_repeated=True), feed_dict = {model.x: audio_input_data, model.y: tf.SparseTensorValue(y, np.ones_like(y[:,0]), np.array([len(timit_load_data.phones),1])), model.seq_len : np.ones([audio_input_data.shape[0]])*61})
    #test_pred = sess.run(test_pred, feed_dict = {model.x: audio_input_data, model.y: tf.SparseTensorValue(y, np.ones_like(y[:,0]), np.array([len(timit_load_data.phones),1])), model.seq_len : np.ones([audio_input_data.shape[0]])*61})
    print(test_pred.shape)
    print(test_pred)
    #for i in range(y.shape[0]):
      #print("y:", timit_load_data.ix_to_phn[np.argmax(y[i])], " y_pred:", timit_load_data.ix_to_phn[np.argmax(test_pred[i])])
    
    
n_size = 100
net = DeepSpeech(n_size,n_size,n_size,n_size,61,80)
net.init_var()
print("phonemes len", len(timit_load_data.phones))
timit_filepaths = list(sorted(set(open("timit/allfilelist.txt", 'r').read().split("\n"))))
print(len(timit_filepaths[1:-1]))
net.train(net.load_timit_ctc_data, timit_filepaths[1:-1])
