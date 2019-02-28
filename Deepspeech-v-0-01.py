import numpy as np
import tensorflow as tf
import soundfile as sf
import sys
import timit_load_data
import tools
import pickle
import time

class DeepSpeech(object):

  def __init__(model, h1_size, h2_size, h3_size, h4_size, h5_size, input_size):
    model.h1_size, model.h2_size, model.h3_size, model.h4_size, model.h5_size = h1_size, h2_size, h3_size, h4_size, h5_size
    model.input_size = input_size #number of features of windowed data
    model.relu_clip = 20
    model.sess = tf.Session()
    ##Hyper Parameters
    model.learning_rate = 0.0001
    model.beta1 = 0.9
    model.beta2 = 0.999
    model.epsilon = 1e-8
    model.print = False

  def load_timit_data(model, filename):
    start = time.time()
    audiopath_full = filename+".wav"
    phonemepath_full = filename+".phn"
    #print(audiopath_full)
    audio, text = timit_load_data.get_data(audiopath_full , phonemepath_full)
    audio, text = tools.window_data(audio,text)
    #print("text", text)
    text = tools.text_to_onehot(text,len(timit_load_data.phones))
    end = time.time()
    #print("load data time taken : ", end-start)
    #print("text reconstructed", tools.onehot_to_text(text))
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

    model.cell_fw = tf.nn.rnn_cell.LSTMCell(model.h3_size, activation=tf.nn.relu, cell_clip=model.relu_clip, initializer=tf.initializers.he_normal())
    model.cell_bw = tf.nn.rnn_cell.LSTMCell(model.h3_size, activation=tf.nn.relu, cell_clip=model.relu_clip, initializer=tf.initializers.he_normal())

    model.w5 = tf.Variable(tf.random_normal([model.h4_size, model.h5_size], stddev = tf.sqrt(2/model.h4_size)), dtype=tf.float32)
    model.b5 = tf.Variable(tf.zeros([model.h5_size]), dtype=tf.float32)

    model.y = tf.placeholder("float", shape=[None,model.h5_size])



    

  def BiRnn(model,x, batch_size, seq_len=None, previous_state=None):
    if (seq_len == None):
      seq_len = tf.shape(x)[0]
    
    h1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(x, model.w1), model.b1)), model.relu_clip)
    h2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h1, model.w2), model.b2)), model.relu_clip)
    h3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h2, model.w3), model.b3)), model.relu_clip)  

    ##dynamic biRNN current choice 
    h3_reshape = tf.reshape(h3, [batch_size, seq_len, model.h3_size])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      cell_fw = model.cell_fw,
      cell_bw = model.cell_bw,
      dtype=tf.float32,
      inputs = h3_reshape,
      time_major = False)

    output_fw, output_bw = outputs
    states_fw, states_bw = states
    model.output_fw = tf.minimum(output_fw, model.relu_clip)
    model.output_bw = tf.minimum(output_bw, model.relu_clip)
    if(model.print==True):
      print(model.sess.run(tf.shape(output_fw)))
      print(model.sess.run(tf.shape(output_bw)))
      
    h4 = tf.reshape(tf.add(model.output_fw,model.output_bw), [batch_size,model.h3_size])
    h5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(h4, model.w5), model.b5)), model.relu_clip)
    ##h6 = tf.nn.softmax(h5)

    ##print(sess.run(tf.shape(h5)))
    return h5

  #------------------------------------------------------------------------------------------------

  def train(model, load_data_function, filenames_list):
    ''' trains a single data sample at a time

        load_data_function must return:
        audio input data - batch_size x frame_size
        y - onehot encoded label

        filenames_list must be a list of str valued filepaths
    '''
    
    model.y_pred = model.BiRnn(model.x,seq_len=None,batch_size=-1)
  
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2( 
                        labels=model.y, logits=model.y_pred))
    
    print("loss shape",loss.shape)
    optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate,
                                       beta1=model.beta1, beta2=model.beta2, epsilon=model.epsilon).minimize(loss)
    init = tf.global_variables_initializer()
    model.sess.run(init)

##    with open(r"processed_timit_audio.pickle", "rb") as input_file:
##      audio_input_data = pickle.load(input_file)
##
##    with open(r"processed_timit_phns.pickle", "rb") as input_file:
##      text_input_data = pickle.load(input_file)

    model.print = True
    #print("output shape",model.sess.run(tf.shape(model.output_fw), feed_dict = {model.x: audio_input_data[0], model.y: text_input_data[0]}))
    num_epochs = 2
    average_loss = np.zeros((num_epochs))
    for epoch in range(num_epochs):
      for i in range(len(filenames_list)):
        audio_input_data, y = load_data_function(filenames_list[i])
      #for i in range(np.int(0.7*len(audio_input_data))):
        _,l = model.sess.run([optimizer,loss], feed_dict = {model.x: audio_input_data, model.y: y})
        average_loss[epoch] += l
      average_loss[epoch] /= len(audio_input_data)
      print("epoch ", epoch, " Loss ", average_loss[epoch])

    #batch_size = audio_input_data.shape[0]
    #seq_len = tf.to_int32(tf.fill([batch_size], 61))
    #test_pred = sess.run(tf.nn.ctc_beam_search_decoder(tf.nn.softmax(tf.reshape(y_pred,shape=[1,batch_size,62])), seq_len, merge_repeated=False), feed_dict={model.x: audio_input_data, model.y : y})
    #print(test_pred.shape)
 #-----------------------------------------------------------------------------------------------------------------------------------------------   

  def test(model, load_data_function, filenames_list):
    ''' test phoneme accuracy for timed labeled phonemes
    '''
    #y_pred = model.BiRnn(model.x,seq_len=None,batch_size=-1)
    unigram = timit_load_data.get_unigram("bg261_t0_d05.1N.LM.txt")
    #test_pred = sess.run(tf.nn.softmax(y_pred), feed_dict={model.x: audio_input_data, model.y : y})
    correct = 0
    tot_phns = 0

##    with open(r"processed_timit_audio.pickle", "rb") as input_file:
##      audio_input_data = pickle.load(input_file)
##
##    with open(r"processed_timit_phns.pickle", "rb") as input_file:
##      text_input_data = pickle.load(input_file)

   ##print(text_input_data[0].shape)
      
    for i in range(len(filenames_list)):
      audio_input_data, y = load_data_function(filenames_list[i])
    #for i in range(np.int(0.7*len(audio_input_data)), len(audio_input_data)):
      #y = text_input_data[i]
      test_pred = model.sess.run(tf.nn.softmax(model.y_pred), feed_dict={model.x: audio_input_data, model.y : y})
      for i in range(y.shape[0]):
        y_argmax = timit_load_data.ix_to_phn[np.argmax(y[i])]
        ypred_argmax = timit_load_data.ix_to_phn[np.argmax(test_pred[i]*np.exp(unigram))]
        tot_phns +=1
        #print("y_argmax",y_argmax,"ypred_argmax",ypred_argmax)
        if(y_argmax == ypred_argmax):
          correct +=1
        #print("y:", timit_load_data.ix_to_phn[np.argmax(y[i])], " y_pred:", timit_load_data.ix_to_phn[np.argmax(test_pred[i]*np.exp(unigram))])

    print("correct phonemes predicted", correct, "/", tot_phns, "=", 100*(correct/tot_phns), "%")


n_size = 1024
net = DeepSpeech(n_size,n_size,n_size,n_size,61,80)
net.init_var()
print("phonemes len", len(timit_load_data.phones))

timit_train_filepaths = list(sorted(set(open("trainfilelist.txt", 'r').read().split("\n"))))
for i in range(1, len(timit_train_filepaths)):
  timit_train_filepaths[i] = "/share/spandh.ami1/data/audvis/asr/studio/us/timit/NIST_1-1.1/timit/" + str(timit_train_filepaths[i])

timit_test_filepaths = list(sorted(set(open("testfilelist.txt", 'r').read().split("\n"))))
for i in range(1, len(timit_test_filepaths)):
  timit_test_filepaths[i] = "/share/spandh.ami1/data/audvis/asr/studio/us/timit/NIST_1-1.1/timit/" + str(timit_test_filepaths[i])

timit_filepaths = list(sorted(set(open("timit/allfilelist.txt", 'r').read().split("\n"))))
print("n_size=", n_size)
print("number of training examples used", len(timit_train_filepaths[1:]))
net.train(net.load_timit_data, timit_train_filepaths[1:])
net.test(net.load_timit_data, timit_test_filepaths[1:])
