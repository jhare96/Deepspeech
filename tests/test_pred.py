import tools 
import numpy as np
import soundfile as sf




if __name__ == '__main__':
    
    test_files = open('LibriSpeech/test-clean-transcripts.txt').read().split('\n')
    test_files = [line for line in test_files if len(line) > 2]

    incorrect = 0
    for i in range(len(test_files)):
        line = test_files[i]
        path = 'LibriSpeech/' + line.split(' ')[0] + '.flac'
        audio, samplerate = sf.read(path)
        pred_size = tools.num_frames(audio.shape[0])
        filterbanks = tools.window_data(audio)
        actual_shape = int(len(filterbanks))
        if pred_size != actual_shape:
            incorrect+=1
            print('pred_size', pred_size, 'actual size', actual_shape)
    
    print('%i/%i incorrect'%(incorrect,i))
