# Deepspeech

This repository contains a Tensorflow implementation of Baidu's [Deepspeech](https://arxiv.org/abs/1412.5567) architecture, a neural speech-to-text model. This repository was inspired by the [Mozilla's implementation](https://github.com/mozilla/DeepSpeech).

# Architecture 
The model architecture is a sequence-to-sequence Artificial Neural Network (ANN) with Connectionist Temporal Classification (CTC), 
which maps a given audio file to a series of graphemes (characters) e.g. the english alphabet **a-z**. 

## Input 
```tools.py``` contains a function which converts a raw audio file of samplerate 16kHz to a series of mel-spaced log-filterbanks features, using a stride of 10ms and window size 20ms.
![waveform](images/waveform3.png)


![mel filterbanks](images/filterbanks1.png)

## Model
The model consists of 3 linear layers followed by a recurrent network with options for bidirectionality as well as cell type e.g. GRU, LSTM or RNN, following this is another linear layer and finally a linear projection to form the unnormalised log probabilities (logits) for the CTC input. 
