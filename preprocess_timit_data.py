import numpy as np
import timit_load_data
import tools
import pickle


def load_timit_data(filename):
    audiopath_full = "/share/spandh.ami1/data/audvis/asr/studio/us/timit/NIST_1-1.1/timit/"+str(filename)+".wav"
    phonemepath_full = "/share/spandh.ami1/data/audvis/asr/studio/us/timit/NIST_1-1.1/timit/"+str(filename)+".phn"
    #print(audiopath_full)
    audio, text = timit_load_data.get_data(audiopath_full , phonemepath_full)
    audio, text = tools.window_data(audio,text)
    #print("text", text)
    text = tools.text_to_onehot(text,len(timit_load_data.phones))
    #print("text reconstructed", tools.onehot_to_text(text))
    return audio,text 

timit_train_filepaths = list(sorted(set(open("trainfilelist.txt", 'r').read().split("\n"))))[1:]
print("number of filepaths", len(timit_train_filepaths))
windowed_audio_data_list = []
windowed_text_list = []
for i in range(len(timit_train_filepaths)):
      audio_data, text = load_timit_data(timit_train_filepaths[i])
      windowed_audio_data_list.append(audio_data)
      windowed_text_list.append(text)


with open(r"processed_timit_train_audio.pickle", "wb") as output_file:
   pickle.dump(windowed_audio_data_list, output_file)

with open(r"processed_timit_train_phns.pickle", "wb") as output_file:
   pickle.dump(windowed_text_list, output_file)

with open(r"processed_timit_train_audio.pickle", "rb") as input_file:
   e = pickle.load(input_file)

with open(r"processed_timit_train_phns.pickle", "rb") as input_file:
   f = pickle.load(input_file)

print("len e", len(e))
print("len f", len(f))
print("number of audio files processed", len(windowed_audio_data_list))
