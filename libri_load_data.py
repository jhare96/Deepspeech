import numpy as np
import tools
import string
import soundfile as sf

alphabet = list(sorted(set(string.ascii_lowercase))) + ['<SPACE>']
char_to_ix = { char:ix for ix,char in enumerate(alphabet) }
ix_to_char = { ix:char for ix,char in enumerate(alphabet) }

LIBRI_DIR = 'LibriSpeech/'

def load_libri_ctc_chars(line):
    line = line.split(' ')
    audiopath = LIBRI_DIR + line[0] + '.flac'
    words = line[1:]
    audio, chars = get_ctc_data_chars(audiopath, words)
    #audio = tools.window_data(audio)
    return audio, chars


def get_ctc_data_chars(audio_filepath, words):
    audio, samplerate = sf.read(audio_filepath)
    words = [word.replace("'", '').lower() for word in words]
    characters = []
    #characters.append(char_to_ix['<\START>'])
    for word in words:
        for char in word:
            #print('char', char)
            characters.append(char_to_ix[char])
        characters.append(char_to_ix['<SPACE>'])
    
    del characters[-1] # remove last space
    #characters.append(char_to_ix['<\END>'])
   
    return audio, np.array(characters)
