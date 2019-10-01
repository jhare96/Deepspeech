import tensorflow as tf
import numpy as np
import time
import soundfile as sf
#import timit_load_data
import datetime
import os
import libri_load_data 
import tools
import string
import WER
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

def relu_clipped(x, clip=20):
    return tf.minimum(tf.maximum(x, 0.0), clip)


class DeepspeechTower(object):
    def __init__(self, hidden_size, num_filter_banks, num_classes, batch_size=1, bidirectional=False, rnn_layers=2, rnn_type='GRU', namescope=None, relu_clip=20):
        with tf.name_scope(namescope):
            self.num_classes = num_classes
            self.batch_size = batch_size
            self.relu_clip = relu_clip

            rnn_types = {'GRU': self.GRUCell, 'RNN':self.RNNCell, 'LSTM':self.LSTMCell}
            try:
                rnn_cell = rnn_types[rnn_type]
            except KeyError:
                raise ValueError('{} is not a valid rnn cell type, valid types are {}'.format(rnn_type,rnn_types.keys()))
                


            self.input = tf.placeholder(tf.float32, shape=[None, num_filter_banks], name='filterbanks')
            self.sparselabels = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2], name='indices'), tf.placeholder(tf.int32, shape=[None], name='values'), tf.placeholder(tf.int64, shape=[2], name='dense_shape'))
            self.dropout = tf.placeholder(tf.bool, shape=(), name='dropout_flag')
            self.seq_len = tf.placeholder(tf.int32, shape=[batch_size], name='seq_length') # unpadded sequence length
            
            with tf.variable_scope('Deepspeech', reuse=tf.AUTO_REUSE):
                self.h5, self.train_ctc_logits = self.build_Deepspeech(self.input, hidden_size=hidden_size, rnn_cell=rnn_cell, batch_size=batch_size, bidirectional=bidirectional, rnn_layers=rnn_layers, seq_len=self.seq_len)
            print('ctc logits shape', self.train_ctc_logits.get_shape().as_list())
            
            #self.loss = tf.nn.ctc_loss_v2(self.labels, self.train_ctc_logits, self.logit_len, self.label_len, logits_time_major=True)
            #self.loss = tf.keras.backend.ctc_batch_cost(self.labels, tf.nn.softmax(self.train_ctc_logits, axis=-1), self.logit_len, self.label_len)

            # ctc loss
            self.loss = tf.nn.ctc_loss(self.sparselabels, self.train_ctc_logits, self.seq_len, time_major=True)
            # ctc decoder 
            self.batch_ctc_decoded = tf.nn.ctc_beam_search_decoder(self.train_ctc_logits, self.seq_len, beam_width=100, top_paths=1, merge_repeated=True)
                
            # with tf.name_scope('Single_Inference'):
            #     with tf.variable_scope('Deepspeech', reuse=True):
            #         self.val_seq_len = tf.placeholder(tf.int32, shape=[1], name='val_seq_length') # unpadded sequence length
            #         _, self.validate_ctc_logits = self.build_Deepspeech(self.input, hidden_size=hidden_size, batch_size=1, seq_len=self.val_seq_len)
            #     #self.decoder_logit_len = tf.placeholder(tf.int32, shape=[None], name='decoder_logit_length')
                
            #     #with tf.device('CPU:0'):
            #     self.singular_ctc_decoded = tf.nn.ctc_beam_search_decoder(self.validate_ctc_logits, self.val_seq_len, beam_width=100, top_paths=1)
            #     #self.ctc_decoded = tf.keras.backend.ctc_decode(tf.nn.softmax(self.validate_ctc_logits, axis=-1), self.decoder_logit_len, greedy=False, beam_width=100, top_paths=1)


            self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.grads = tf.gradients(self.loss, self.weights)
           

        # for pretraining only
        # self.pretrain_optimiser = tf.train.AdamOptimizer(1e-4)
        # self.decoded_seq = self.decoder(self.h4, num_filter_banks)
        # self.pretrain_loss = tf.reduce_mean(tf.square(self.decoded_seq-self.input))
        # self.pretrain_op = self.pretrain_optimiser.minimize(self.pretrain_loss)

    def build_Deepspeech(self, x, rnn_cell, hidden_size=1024, batch_size=1, bidirectional=False, rnn_layers=2, activation=relu_clipped, initialiser=tf.initializers.glorot_uniform(), seq_len=None, training=True): #tf.initializers.orthogonal
        h1 = tf.layers.dropout(tf.layers.dense(x, hidden_size, activation=activation, kernel_initializer=initialiser), rate=0.05, training=training)
        h2 = tf.layers.dropout(tf.layers.dense(h1, hidden_size, activation=activation, kernel_initializer=initialiser), rate=0.05, training=training)
        h3 = tf.layers.dropout(tf.layers.dense(h2, hidden_size, activation=activation, kernel_initializer=initialiser), rate=0.05, training=training)
        h3_reshape = tf.reshape(h3, shape=[batch_size, -1, hidden_size])

        if bidirectional:
            cells_fw = [rnn_cell(hidden_size) for i in range(rnn_layers)]
            fw_stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            cells_bw = [rnn_cell(hidden_size) for i in range(rnn_layers)]
            bw_stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw = fw_stacked_cells,
                                                            cell_bw = bw_stacked_cells,
                                                            dtype=tf.float32,
                                                            inputs = h3_reshape,
                                                            sequence_length=seq_len,
                                                            time_major = False,
                                                            swap_memory=True)

            output_fw, output_bw = outputs
            states_fw, states_bw = states
            h4 = tf.reshape(tf.add(output_fw, output_bw), [-1, hidden_size])
        
        else:
            cells = [rnn_cell(hidden_size) for i in range(rnn_layers)]
            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
            outputs, states = tf.nn.dynamic_rnn(cell= stacked_cells,
                                                dtype=tf.float32,
                                                inputs = h3_reshape,
                                                sequence_length=seq_len,
                                                time_major = False,
                                                swap_memory=True)
            print('rnn outputs', outputs.get_shape().as_list())
            h4 = tf.reshape(outputs, [-1, hidden_size])

        
        h5 = tf.layers.dropout(tf.layers.dense(h4, hidden_size, activation=activation, kernel_initializer=initialiser), rate=0.05, training=training)
        ctc_logits = tf.layers.dropout(tf.reshape(tf.layers.dense(h5, self.num_classes, activation=None), shape=[batch_size, -1 , self.num_classes]), rate=0.05, training=training)
        return h5, tf.transpose(ctc_logits,perm=[1,0,2])

    def RNNCell(self, hidden_size, activation=relu_clipped):
        return tf.nn.rnn_cell.BasicRNNCell(hidden_size, activation=activation)
    
    def GRUCell(self, hidden_size):
        return tf.nn.rnn_cell.GRUCell(hidden_size)
    
    def LSTMCell(self, hidden_size):
        return tf.nn.rnn_cell.LSTMCell(hidden_size)


    def decoder(self, x, input_size, activation=tf.nn.relu, initialiser=tf.initializers.he_normal):
        d1 = tf.layers.dense(x, input_size*2, activation=activation, kernel_initializer=initialiser)
        yhat = tf.layers.dense(d1, input_size, activation=activation, kernel_initializer=initialiser)
        return yhat
    
    def set_session(self, sess):
        self.sess = sess



## -------------------------------------------------------------------------------------------------------------------------------------
## Deepspeech Model and trainer 
## -------------------------------------------------------------------------------------------------------------------------------------
class Deepspeech(object):
    #self, hidden_size, num_filter_banks, num_classes, batch_size=1, bidirectional=False, rnn_layers=2, namescope=None, relu_clip=20):
    def __init__(self, hidden_size, num_filter_banks, num_classes, batch_size=1, bidirectional=False, rnn_layers=2, rnn_type='GRU', sess=None, number_GPUs=1, save_model=True):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_towers = number_GPUs
        self.num_filters = num_filter_banks
        self.Towers = []
        for d in range(number_GPUs):
            with tf.device('gpu:%i'%d):
                with tf.variable_scope('Model', tf.AUTO_REUSE):
                    self.Towers.append(DeepspeechTower(hidden_size=hidden_size,
                                                       num_filter_banks=num_filter_banks,
                                                       num_classes=num_classes,
                                                       batch_size=batch_size,
                                                       bidirectional=bidirectional,
                                                       rnn_layers=rnn_layers,
                                                       rnn_type=rnn_type,
                                                       namescope='Tower_%i'%d))

        self.optimiser = tf.train.AdamOptimizer(1e-4)
        self.loss = tf.reduce_mean([tower.loss for tower in self.Towers])
        tower_gradients = zip(*[tower.grads for tower in self.Towers])
        
        avg_grads = []
        for grads in tower_gradients:
            avg_grads.append(tf.reduce_mean(tf.concat([tf.expand_dims(gs,0) for gs in grads], axis=0), axis=0))

        #print('avg grads', avg_grads)
        grads_vars = list(zip(avg_grads, self.Towers[0].weights))
        self.train_op = self.optimiser.apply_gradients(grads_vars)
    
        if sess is None:
            config = tf.ConfigProto(allow_soft_placement=True) # Allow automatic CPU placement if GPU kernel not availiable 
            config.gpu_options.allow_growth = True # Allow GPU growth
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess
        
        self.sess.run(tf.global_variables_initializer()) 

        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.saver = tf.train.Saver() 

        if save_model:
            self.model_dir = 'models/Deepspeech/' + current_time + '/' 
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        
        self.train_writer = tf.summary.FileWriter('logs/DeepSpeech/'+current_time+'/', graph=self.sess.graph)
        self.loss_placeholder = tf.placeholder(tf.float32, shape=())
        self.summary_loss = tf.summary.scalar('ctc_loss', self.loss_placeholder)
    
    def save(self, checkpoint):
        self.saver.save(self.sess, self.model_dir + 'model'+str(checkpoint)+'.ckpt')
    
    def load_weights(self, model_path, modelname):
        if os.path.exists(model_path + modelname+ ".ckpt"+ ".meta"):
            self.saver.restore(self.sess, model_path + modelname+".ckpt")
            print("loaded:", model_path+modelname)
            return True
        else:
            print(model_path + modelname, " does not exist")
            return False

    def pretrain(self, load_data_function, filenames_list, epochs=5):
        num_samples = len(filenames_list)
        for epoch in range(epochs):
            l,i = 0,0
            start = time.time()
            for file in filenames_list:
                filterbanks, *_ = load_data_function(file)
                feed_dict = {self.input:filterbanks}
                loss, _ = self.sess.run([self.pretrain_loss, self.pretrain_op], feed_dict=feed_dict)
                l+=loss
                i+=1
            
            time_taken = time.time() - start
            wer = self.validate(load_data_function, filenames_list)
            print('epoch {}, WER {}, average loss {}, time taken {}, samples per second {}'.format(epoch, wer, l/i, time_taken, num_samples/time_taken))

    def labels_to_sparse(self, labels):
        # labels - list of numpy 1-D arrays containing values of character onehot indices
        indices = []
        values = []
        maxlen = 0
        for b in range(len(labels)):
            label_len = len(labels[b])
            if label_len > maxlen:
                maxlen = label_len
            for t in range(label_len):
                indices.append([b,t])
                values.append(labels[b][t])
        
        dense_shape = [len(labels), maxlen]

        return(indices, values, dense_shape)

    
    def get_tower_batches(self, load_data_function, files, i, train=True):
        feed_dict = {}
        for j in range(len(self.Towers)):
            audio_waveforms, labels = zip(*[load_data_function(file) for file in files[i+(self.batch_size*j):i+(self.batch_size*(j+1))]])
            filterbanks = [tools.window_data(audio, self.num_filters) for audio in audio_waveforms]
            fb_lens = np.stack([len(fb) for fb in filterbanks])
            # pad filterbanks sequences so they all have the same shape
            filterbanks = np.stack([tools.pad_to_length(fb, np.max(fb_lens), mode='zeros') for fb in filterbanks])
            
            filterbanks = tools.fold_batch(filterbanks) # fold batch [batch,time,nfilters] --> [batch*time,nfilters]

            feed_dict[self.Towers[j].input] = filterbanks
            feed_dict[self.Towers[j].sparselabels] = self.labels_to_sparse(labels)
            feed_dict[self.Towers[j].seq_len] = fb_lens
            feed_dict[self.Towers[j].dropout] = train
            #feed_dict[self.Towers[j].label_len] = lb_lens
        
        return feed_dict

    def train(self, load_data_function, trainfiles, testfiles, epochs=5, validate_freq=5000, compute_WER=False):
        tot_batch_size = self.batch_size * self.num_towers
        # find closest multiple of batch_size * num_towers to length of training samples
        batch_multiple = (len(trainfiles) // tot_batch_size) * tot_batch_size 
        tot_samples = 0
        start = time.time()
        avgloss = 0
        count = 0
        for epoch in range(epochs):
            np.random.shuffle(trainfiles) # shuffle list to ensure all samples are seen over many epochs 
            for i in range(0, batch_multiple, tot_batch_size):
                #start3 = time.time()
                feed_dict = self.get_tower_batches(load_data_function, trainfiles, i)
                # compute ctc loss over all towers
                loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                avgloss += np.mean(loss)
                count += 1

                tot_samples += tot_batch_size
                #time_taken = time.time() - start3
                #print('time taken {}, fps {}'.format(time_taken, tot_batch_size / time_taken))
                

            print('validation starting', i, tot_samples)
            self.save(epoch)
            time_taken = time.time() - start
            start2 = time.time()
            if compute_WER:
                wer = self.validate(load_data_function, testfiles, save_to_file=True)
                validation_time = time.time()- start2
            else:
                wer = None
            avgloss = avgloss/count
            printline = 'epoch {}, total_samples {}, average loss {}, time taken {}, samples per second {}'.format(epoch,
                            tot_samples, avgloss, time_taken, validate_freq/time_taken)
            if compute_WER:
                printline = 'WER {}, '.format(wer) + printline + ', validation_time {}'.format(validation_time)
            print(printline)

            # write average ctc loss to Tensorboard log 
            summ_loss = self.sess.run(self.summary_loss, {self.loss_placeholder:avgloss})
            self.train_writer.add_summary(summ_loss, tot_samples)
            
            avgloss = 0
            count = 0
            start = time.time()
                
                

    
    def validate(self, load_data_function, files, save_to_file=False):
        wer = 0
        batch_multiple = (len(files) // self.batch_size) * self.batch_size
        writefile = open(self.model_dir + 'sample.txt', 'w')
        for i in range(0, batch_multiple, self.batch_size): # a single Tower is used for decoding for simplicity 
            start = time.time()
            audio_waveforms, labels = zip(*[load_data_function(file) for file in files[i:i+self.batch_size]])
            filterbanks = [tools.window_data(audio, self.num_filters) for audio in audio_waveforms]
            fb_lens = np.stack([len(fb) for fb in filterbanks])
            # pad filterbanks sequences so they all have the same shape
            filterbanks = np.stack([tools.pad_to_length(fb, np.max(fb_lens), mode='zeros') for fb in filterbanks])
            
            filterbanks = tools.fold_batch(filterbanks) # fold batch [batch,time,nfilters] --> [batch*time,nfilters]
            
            feed_dict = {self.Towers[0].input:filterbanks,
                        self.Towers[0].sparselabels:self.labels_to_sparse(labels),
                        self.Towers[0].seq_len:fb_lens,
                        self.Towers[0].dropout:False}

            start3 = time.time()
            outputs = self.sess.run(self.Towers[0].batch_ctc_decoded, feed_dict=feed_dict)
            best_decoded = outputs[0][0] # for output in outputs]
            #print('indices', best_decoded.indices.shape)
            #print('values', best_decoded.values.shape)
            values = [[] for batch in range(self.batch_size)]
            #print('outputs len', len(outputs), len(outputs[0]))
            #print('decoder time', time.time() - start3)
            start2 = time.time()
            for m in range(best_decoded.values.shape[0]):
                b,t = best_decoded.indices[m]
                value = best_decoded.values[m]
                values[b].append(value)
            
            
            for k in range(self.batch_size):
                predtext = ''.join(libri_load_data.ix_to_char[idx] for idx in values[k])
                strlabels = ''.join(libri_load_data.ix_to_char[idx] for idx in labels[k])
        
                wer += WER.WER(strlabels.split('<SPACE>'), predtext.split('<SPACE>')) # Levenshtein distance metric 
            
                if save_to_file:
                    writefile.write('\nactual: ' + strlabels.replace('<SPACE>', ' ') +'\n')
                    writefile.write('predicted: ' + predtext.replace('<SPACE>', ' ') +'\n')
            
            #print('WER time', time.time() - start2)
        writefile.close()
        return wer/ batch_multiple

    

    def save_ctc(self, load_data_function, file):
        wer = 0
        audio_waveform, label = load_data_function(file)
        filterbanks = tools.window_data(audio_waveform, self.num_filters)

        feed_dict = {self.Towers[0].input:filterbanks,
                    self.Towers[0].sparselabels:self.labels_to_sparse([label]),
                    self.Towers[0].seq_len:[len(filterbanks)]}

        outputs = self.sess.run(tf.nn.softmax(self.Towers[0].train_ctc_logits), feed_dict=feed_dict)
        np.save('ctc_mat', outputs)







def main():
    train_files =  open('LibriSpeech/train-clean-100-transcripts.txt').read().split('\n')[:-1] + open('LibriSpeech/train-clean-360-transcripts.txt').read().split('\n')[:-1]
    #train_files = [line for line in train_files if len(line) > 2]
    print(len(train_files))
    test_files = open('LibriSpeech/test-clean-transcripts.txt').read().split('\n')[:-1]
    test_files = [line for line in test_files if len(line) > 2]
    deepspeech = Deepspeech(2048, 40, len(libri_load_data.alphabet)+1, batch_size=16, bidirectional=True, rnn_layers=1, rnn_type='GRU', number_GPUs=2, save_model=True)
    #if not deepspeech.load_weights('models/Deepspeech/2019-09-21_19-33-39/', 'model9'):
        #exit()
    #deepspeech.pretrain(load_timit_ctc_chars, files[:], epochs=1)
    print('idx to char', libri_load_data.ix_to_char)
    
    batch_multiple = (len(train_files) // (deepspeech.batch_size * deepspeech.num_towers)) * (deepspeech.batch_size * deepspeech.num_towers)
    print('batch_multiple', batch_multiple)
    
    deepspeech.train(libri_load_data.load_libri_ctc_chars, train_files, test_files, epochs=20, validate_freq=batch_multiple, compute_WER=True)
    print('finished training')

    wer = deepspeech.validate(libri_load_data.load_libri_ctc_chars, test_files, save_to_file=True)
    print('final WER', wer)
    



if __name__ == '__main__': 
    main()
