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
    ##These are DEFINITELY WRONG
    model.learning_rate = 0.01
    model.beta1 = 0.9
    model.beta2 = 0.1
    model.epsilon = 0.001
    ##

  def load_timit_data(model, filename):
##    audio, samplerate = sf.read("timit/dr1-fvmh0/sa1.wav")
##    print("sample rate = ", samplerate)
##    text = open("timit/dr1-fvmh0/sa1.phn", 'r').read()
##    print(len(audio))
##    audio = tf.convert_to_tensor(np.array(audio).reshape(1,33440), np.float32)
##    text = tf.convert_to_tensor(text)
    audio, text = timit_load_data.get_data(filename+".wav", filename+".phn")
    audio, text = tools.window_data(audio,text)
    print("text", text)
    text = tools.text_to_onehot(text,len(timit_load_data.phones))
    print("text reconstructed", tools.onehot_to_text(text))
    return audio,text 


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


    

  def BiRnn(model,x, n_steps, batch_size, previous_state=None):
    
    h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(x, model.w1), model.b1)), model.relu_clip)
    print("helllooonnnde")
    h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1, model.w2), model.b2)), model.relu_clip)
    h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2, model.w3), model.b3)), model.relu_clip)

##    init = tf.global_variables_initializer()
##    sess = tf.Session()
##    sess.run(init)
##    print(sess.run(tf.shape(h3)))

    LSTM = False
    #lstm = tf.nn.rnn_cell.LSTMCell(10)
    #state = lstm.zero_state(10, dtype=tf.float32)

    #for i in range(seq_length):
    ##hf_output, state = lstm(h3[0],None)   

    if(LSTM == True): ##unrolled biLSTM
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

    else: ##dynamic biRNN
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


  

  def train(model, input_data, y, n_steps = -1, batch_size):
    ''' trains a single data sample at a time
        input data - batch_size x frame_size
        y - onehot encoded label'''

    model.init_var()
    
    y_pred = model.BiRnn(model.x,n_steps,batch_size)
    ##loss = y - y_pred
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
                        labels=y, logits=y_pred)) 
    print("loss shape",loss.shape)
    optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate,
                                       beta1=model.beta1, beta2=model.beta2, epsilon=model.epsilon).minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    print("input data shape", sess.run(tf.shape(input_data)))
    sess.run(init)

    for epoch in range(10):
      _,l = sess.run([optimizer,loss], feed_dict = {model.x: input_data})
      #if (epoch % 10 == 0):
        #print('Step %i: Loss: %f' % (epoch, l))
      print("epoch ", epoch, " Loss ", l)
    for i in range(113):
      print("y:", timit_load_data.ix_to_phn[np.argmax(sess.run((y[i])))], " y_pred:", timit_load_data.ix_to_phn[np.argmax(sess.run((tf.nn.softmax(y_pred[i])), feed_dict={model.x: input_data}))])
    
    

net = DeepSpeech(204*2,204*2,204*2,204*2,62,480)
audio, text = net.load_data()
net.init_var()
print("phoneemes len", len(timit_load_data.phones))
#audio_batch = tf.reshape(audio, [,10])
#net.BiRnn(tf.convert_to_tensor(audio,np.float32),-1,audio.shape[0])
##net.train(tf.convert_to_tensor(audio,np.float32), tf.convert_to_tensor(text, np.float32), -1, audio.shape[0])
#print("Text", text)
net.train(audio, tf.convert_to_tensor(text, np.float32), -1, audio.shape[0])
