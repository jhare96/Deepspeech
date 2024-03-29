import numpy as np
import soundfile as sf
import time
import string
import tools

phones = list(sorted(set(open("all_phonemes.txt", 'r').read().split("\n"))))[1:]
#print("phones", phones)
phn_to_ix = { phn:ix for ix,phn in enumerate(phones) }
ix_to_phn = { ix:phn for ix,phn in enumerate(phones) }

alphabet = list(sorted(set(string.ascii_lowercase))) + ['<SPACE>']
char_to_ix = { char:ix for ix,char in enumerate(alphabet) }
ix_to_char = { ix:char for ix,char in enumerate(alphabet) }

def get_data(audio_filepath, text_filepath):
  starty = time.time()
  audio, samplerate = sf.read(audio_filepath)
  #print("samplerate", samplerate )
  phonemes = np.zeros_like(audio, dtype=np.int32)
  
  text = open(text_filepath, 'r').read().split("\n")
  #print(text)

  for line in text[0:-1]:
    stamp = line.split(" ")
    if (stamp[-1] == '') : break
    start = int(stamp[0])
    stop = int(stamp[1])
    phn = phn_to_ix[stamp[2]]
    phonemes[start:stop] = phn

  end = time.time()
  #print("load data time taken:", (end-starty)
  #print("phones", phonemes)
  return audio, phonemes

def get_ctc_data(audio_filepath, text_filepath):
  audio, samplerate = sf.read(audio_filepath)
  text = open(text_filepath, 'r').read().split("\n")[0:-1]
  phonemes = np.zeros((len(text),1), dtype=np.int32)
  values = np.ones_like(phonemes)
  i = 0
  for line in text:
    stamp = line.split(" ")
    if (stamp[-1] == '') : break
    #print(stamp[2])
    phn = phn_to_ix[stamp[2]]
    phonemes[i] = phn
    i+=1
  return audio, phonemes

def load_timit_ctc_chars(filepath):
    audiopath = filepath + '.wav'
    wordpath = filepath + '.wrd'
    audio, text = get_ctc_data_chars(audiopath, wordpath)
    audio = tools.window_data(audio)
    return audio, text


def get_ctc_data_chars(audio_filepath, text_filepath):
    audio, samplerate = sf.read(audio_filepath)
    text = open(text_filepath, 'r').read().split("\n")[0:-1]
    words = [stamp.split(' ')[2].replace("'", '') for stamp in text]
    characters = []
    #characters.append(char_to_ix['<\START>'])
    for word in words:
        for char in word:
            #print('char', char)
            characters.append(char_to_ix[char])
        characters.append(char_to_ix['<SPACE>'])
    
    #characters.append(char_to_ix['<\END>'])
   
    return audio, np.array(characters)

  
def get_unigram(filepath):
  text = open(filepath, 'r').read().split("\n")[3:-1]
  unigram = np.zeros((len(text)), dtype=np.float32)
  for i in range(len(text)):
    unigram[i] = text[i].split(" ")[0]
  print("unigram shape", unigram.shape)
  #print(type(unigram[0]))
  return unigram


#get_ctc_data("timit/dr1-fvmh0/sa1.wav","timit/dr6-mbma1/sa1.phn")
