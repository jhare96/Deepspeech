import numpy as np
import scipy.stats
import time
import matplotlib.pyplot as plt

def window_data(audio, num_filters=80, text=None):
   #pre-emphasis
   pre_emphasis = 0.97
   audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
   
   start_time = time.time()
   frame_size = 0.02 ##s
   frame_stride = 0.01 ##s
   sample_rate = 16000 #fps
   #sample period 1/16,000
   frame_length = int(round(frame_size * sample_rate)) #=320
   frame_step = int(round(frame_stride * sample_rate)) #=160
   signal_length = audio.shape[0]
   num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step))

   
   pad_signal_length = num_frames * frame_step + frame_length
   z = np.zeros((pad_signal_length - signal_length))
   pad_signal = np.append(audio, z)

   indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
   audio_windowed = pad_signal[indices.astype(np.int32, copy=False)]

   audio_windowed *= np.hamming(frame_length) #hamming window

   NFFT = 512
   magnitude_audio = np.abs(np.fft.rfft(audio_windowed, NFFT))  # Magnitude spectrum of the FFT
   power_audio = ((1.0 / NFFT) * ((magnitude_audio) ** 2))  # Power Spectrum

   nfilt = num_filters
   upper_freq_lim = (2595 * np.log10(1 + (sample_rate / 2) / 700))
   mel_points = np.linspace(0,upper_freq_lim,nfilt+2)
   hz_points = (700 * (10**(mel_points / 2595) - 1)) 
   bin = np.floor((NFFT + 1) * hz_points / sample_rate)

   fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
   for m in range(1, nfilt + 1):
       f_m_minus = int(bin[m - 1])   # left
       f_m = int(bin[m])             # center
       f_m_plus = int(bin[m + 1])    # right

       for k in range(f_m_minus, f_m):
           fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
       for k in range(f_m, f_m_plus):
           fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

   filter_banks = np.dot(power_audio, fbank.T)
   eps = 0.000000001
   #filter_banks[filter_banks < eps] = eps
   filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
   filter_banks = 20 * np.log10(filter_banks)

##   plt.imshow(filter_banks.T, cmap='jet',origin='lowest', aspect='auto', extent=(0,signal_length,0,nfilt))
##   plt.colorbar()
##   plt.show()

   end = time.time()
   if text is None: 
      return filter_banks
   else:
      # get the most frequent value for each window for time aligned text 
      batch_size = filter_banks.shape[0]
      #text_windowed = text[:batch_size * nfilt].reshape(batch_size,nfilt)
      text_windowed = text[indices.astype(np.int32, copy=False)]
      #print("txt windowed",text_windowed.shape)
      text_labeled = np.zeros((batch_size,1),dtype=np.int32)
      for i in range(text_windowed.shape[0]):
         text_labeled[i,:] = scipy.stats.mode(text_windowed[i,:])[0]
      return filter_banks, text_labeled

def num_frames(signal_length, frame_length=320.0, frame_step=160.0):
    num_frames = int(np.ceil((signal_length - frame_length) / frame_step))
    return num_frames

def ix_to_onehot(ix,length):
   label = np.zeros((length))
   label[ix] = 1
   return label

def text_to_onehot(text,length):
   onehot = np.zeros((text.shape[0],length))
   for i in range(text.shape[0]):
      ix = text[i][0]
      onehot[i,ix] = 1
      #print("ix:",ix,"\n",labels[i])
   return onehot

def onehot_to_text(onehot):
   ##converts onehot Matrix samples x classes , into indexes matrix samples x 1
   text = np.zeros((onehot.shape[0],1))
   for i in range(onehot.shape[0]):
      text[i] = np.argmax(onehot[i])
   return text


def fold_batch(x):
   batch, time = x.shape[0], x.shape[1]
   return np.reshape(x, (batch*time, *x.shape[2:]))


def pad_to_length(x, maxlen, mode='same'):
   # pad x along axis 0 to 
   xlen = x.shape[0] 
   if xlen == maxlen:
      return x
   elif xlen > maxlen:
      raise ValueError('input sequence has greater length (%i) than maxlength supplied (%i)'%(xlen,maxlen) )
   else:
      if len(x.shape) == 1:
         xpad = np.zeros((maxlen))
      else:
         xpad = np.zeros((maxlen, *x.shape[1:]))
      xpad[:xlen] = x
      if mode == 'same':
         xpad[xlen:] = x[-1]
      elif mode == 'zeros':
         pass
      elif mode == 'reflect':
         diff = maxlen - xlen
         xpad[xlen:] = x[-diff][::-1]
      
      return xpad

def pad_labels(x, maxlen, value):
   xlen = x.shape[0] 
   if xlen == maxlen:
      return x
   elif xlen > maxlen:
      raise ValueError('padded sequence has greater length than maxlength supplied')
   else:
      padded_x = np.ones((maxlen, *x.shape[1:])) * value
      padded_x[:xlen] = x
      return padded_x


