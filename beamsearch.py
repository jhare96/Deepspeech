import numpy as np
import string
from collections import defaultdict, Counter
import itertools
import pandas as pd
from language_model import NgramLanguageModel, word_Ngrams, char_Ngrams, NgramTree, NgramNode
import time
import re

alphabet = list(sorted(set(string.ascii_lowercase))) + [' ', '-']
char_to_ix = { char:ix for ix,char in enumerate(alphabet) }
ix_to_char = { ix:char for ix,char in enumerate(alphabet) }


class BeamCandidate:
    def __init__(self, beam='', pb=1, pnb=0):
        self.pb = pb
        self.pnb = pnb
        self.beam = beam
    
    def __len__(self):
        return len(self.beam)

def word_beam_search(x, LM, k=20, alpha=2, beta=0.8):
    Aprev = []
    Aprev.append(BeamCandidate(beam=''))
    W = lambda l: [w for w in l.beam.replace('-','').split(' ') if w != '']
    sorter = lambda l: (l.pb + l.pnb)*(len(W(l))-1)**beta
    T = len(x)
    for t in range(T):
        Anext = []
        for l in Aprev:
            for c in range(len(alphabet)):
                if c == len(alphabet) -1 : #blank
                    l.pb = x[t,-1] * (l.pb + l.pnb)#pb[l][t] += x[t,-1] * (pb[l][t-1] + pnb[l][t-1])
                    Anext.append(l)
                else:
                    lplus = BeamCandidate(beam=l.beam + ix_to_char[c])
                    #print('lplus', lplus.beam)
                    ## lplus = l + char
                    if len(l.beam) > 0 and c == l.beam[-1]:
                        ##pnb[lplus][t] += x[t,c] * pb[l][t-1]
                        lplus.pnb = x[t,c] * l.pb
                        ## pnb[l][t] += x[t,c] * pb[l][t-1]
                        l.pnb = x[t,c] * l.pb
                    elif c == len(alphabet) - 2: #space
                        ## pnb[lplus][t] = p(W(lplus)|W(l)) ** alpha * x[t,c] * (pb[l][t-1] + pnb[l][t-1])
                        #print('space')
                        ngram = W(lplus)[-LM.N-1:]
                        if len(ngram) == 0 or LM.p(ngram[-1]) == 0:
                            print('ngram', ngram)
                            LMprob = 0
                        else:
                            ngram_str = ''.join(word + ' ' for word in ngram)[:-1]
                            print('real word', ngram_str)
                            LMprob = LM.p(ngram_str)**alpha
                        # else:
                        #     len(ngram) == 0
                        #     LMprob = 1
                        lplus.pnb = LMprob * x[t,c] * (l.pb + l.pnb)
                    else:
                        ## pnb[lplus][t] += x[t,c] * (pb[l][t-1] + pnb[l][t-1])
                        lplus.pnb = x[t,c] * (l.pb + l.pnb)
                    
                    if lplus.beam not in [prev.beam for prev in Aprev]:
                        ## pb[lplus][t] += x[t,-1] * (pb[lplus][t-1] + pnb[lplus][t-1])
                        lplusprev = [prev.beam for prev in Aprev if prev.beam == lplus.beam][0]
                        lplus.pb = x[t,-1] * (lplusprev.pb + lplusprev.pnb)
                        ## pnb[lplus][t] += x[t,c] * pnb[lplus][t-1]
                        lplus.pnb = x[t,-1] * lplusprev.pnb
                    
                    Anext.append(lplus)
        
            Aprev = sorted(Anext, key=sorter)[-k:]
    
    bestbeam = Aprev[-1].beam.replace('-', '')
    return bestbeam


# class CharBeamCandidate:
#     def __init__(self, beam='', pb=1, pnb=0, ptxt=1):
#         self.pb = pb
#         self.pnb = pnb
#         self.ptxt = ptxt
#         self.beam = beam
    
#     def __len__(self):
#         return len(self.beam)

# def char_beam_search(x, charLM, k=100):
#     Aprev = []
#     Aprev.append(BeamCandidate(beam=alphabet[-1]))
#     sorter = lambda l: (l.pb + l.pnb)*l.ptxt
#     T = len(x)
#     for t in range(T):
#         Anext = []
#         for l in Aprev:
#             for c in range(len(alphabet)):
#                 if c == len(alphabet): #blank
#                     l.pb = x[t,-1] * (l.pb + l.pnb)#pb[l][t] += x[t,-1] * (pb[l][t-1] + pnb[l][t-1])
#                     Anext.append(l)
#                 else:
#                     lplus = BeamCandidate(beam=l.beam + ix_to_char[c])
#                     #print('lplus', lplus.beam)
#                     ## lplus = l + char
#                     if len(l.beam) > 0 and c == l.beam[-1]:
#                         ##pnb[lplus][t] += x[t,c] * pb[l][t-1]
#                         lplus.pnb = x[t,c] * l.pb
#                         ## pnb[l][t] += x[t,c] * pb[l][t-1]
#                         l.pnb = x[t,c] * l.pb
#                     elif c == len(alphabet) - 2: #space
#                         ## pnb[lplus][t] = p(W(lplus)|W(l)) ** alpha * x[t,c] * (pb[l][t-1] + pnb[l][t-1])
#                         #print('space')
#                         ngram = W(lplus)[-LM.N:]
#                         if len(ngram) == 0 or LM.p(ngram[-1]) == 0:
#                             print('ngram', ngram)
#                             LMprob = 0
#                         elif len(ngram) > 0 and LM.p(ngram[-1]) > 0:
#                             ngram_str = ''.join(word + ' ' for word in ngram)[:-1]
#                             print('real word', ngram_str)
#                             LMprob = LM.p(ngram_str)**alpha
#                         # else:
#                         #     len(ngram) == 0
#                         #     LMprob = 1
#                         lplus.pnb = LMprob * x[t,c] * (l.pb + l.pnb)
#                     else:
#                         ## pnb[lplus][t] += x[t,c] * (pb[l][t-1] + pnb[l][t-1])
#                         lplus.pnb = x[t,c] * (l.pb + l.pnb)
                    
#                     # if lplus.beam not in [prev.beam for prev in Aprev]:
#                     #     ## pb[lplus][t] += x[t,-1] * (pb[lplus][t-1] + pnb[lplus][t-1])
#                     #     lplus.pb = x[t,-1] * (lplus.pb + lplus.pnb)
#                     #     ## pnb[lplus][t] += x[t,c] * pnb[lplus][t-1]
#                     #     lplus.pnb = x[t,-1] * lplus.pnb
                    
#                     Anext.append(lplus)
        
#             Aprev = sorted(Anext, key=sorter)[-k:]
    
#     bestbeam = Aprev[-1].beam.replace('-', '')
#     return bestbeam


# def wordchar_beam_search(x, charLM, wordLM, k=200, alpha=0.30, beta=2):
#     Aprev = []
#     Aprev.append(BeamCandidate(beam=alphabet[-1]))
#     W = lambda l: [w for w in l.beam.replace('-','').split(' ') if w != '']
#     sorter = lambda l: (l.pb + l.pnb)#*(len(W(l))-1)**beta
#     T = len(x)
#     for t in range(T):
#         Anext = []
#         for l in Aprev:
#             for c in range(len(alphabet)):
#                 if c == len(alphabet): #blank
#                     l.pb = x[t,-1] * (l.pb + l.pnb)#pb[l][t] += x[t,-1] * (pb[l][t-1] + pnb[l][t-1])
#                     Anext.append(l)
#                 else:
#                     lplus = BeamCandidate(beam=l.beam + ix_to_char[c])
#                     #print('lplus', lplus.beam)
#                     ## lplus = l + char
#                     if len(l.beam) > 0 and c == l.beam[-1]:
#                         ##pnb[lplus][t] += x[t,c] * pb[l][t-1]
#                         lplus.pnb = charLM.p(lplus.beam.strip('-')[-charLM.N:]) * x[t,c] * l.pb
#                         ## pnb[l][t] += x[t,c] * pb[l][t-1]
#                         l.pnb = charLM.p(lplus.beam.strip('-')[-charLM.N:]) * x[t,c] * l.pb
#                     elif c == len(alphabet) - 2: #space
#                         ## pnb[lplus][t] = p(W(lplus)|W(l)) ** alpha * x[t,c] * (pb[l][t-1] + pnb[l][t-1])
#                         #print('space')
#                         ngram = W(lplus)[-wordLM.N:]
#                         if len(ngram) == 0 or wordLM.p(ngram[-1]) == 0:
#                             #print('ngram', ngram)
#                             LMprob = 0
#                         elif len(ngram) > 0 and wordLM.p(ngram[-1]) > 0:
#                             ngram_str = ''.join(word + ' ' for word in ngram)[:-1]
#                             #print('real word', ngram_str)
#                             LMprob = wordLM.p(ngram_str)#**alpha
#                         # else:
#                         #     len(ngram) == 0
#                         #     LMprob = 1
#                         lplus.pnb = LMprob * x[t,c] * (l.pb + l.pnb)
#                     else:
#                         ## pnb[lplus][t] += x[t,c] * (pb[l][t-1] + pnb[l][t-1])
#                         lplus.pnb = charLM.p(lplus.beam.strip('-')[-charLM.N:]) * x[t,c] * (l.pb + l.pnb)
                    
#                     # if lplus.beam not in [prev.beam for prev in Aprev]:
#                     #     ## pb[lplus][t] += x[t,-1] * (pb[lplus][t-1] + pnb[lplus][t-1])
#                     #     lplus.pb = x[t,-1] * (lplus.pb + lplus.pnb)
#                     #     ## pnb[lplus][t] += x[t,c] * pnb[lplus][t-1]
#                     #     lplus.pnb = x[t,-1] * lplus.pnb
                    
#                     Anext.append(lplus)
        
#             Aprev = sorted(Anext, key=sorter)[-k:]
    
#     bestbeam = Aprev[-1].beam.replace('-', '')
#     return bestbeam


def prefix_beam_search(ctc, lm=None, k=50, alpha=0.30, beta=5, prune=0.001):
    """
    Performs prefix beam search on the output of a CTC network.
    Args:
        ctc (np.ndarray): The CTC output. Should be a 2D array (timesteps x alphabet_size)
        lm (func): Language model function. Should take as input a string and output a probability.
        k (int): The beam width. Will keep the 'k' most likely candidates at each timestep.
        alpha (float): The language model weight. Should usually be between 0 and 1.
        beta (float): The language model compensation term. The higher the 'alpha', the higher the 'beta'.
        prune (float): Only extend prefixes with chars with an emission probability higher than 'prune'.
    Retruns:
        string: The decoded CTC output.
    """

    lm = (lambda l: 1) if lm is None else lm # if no LM is provided, just set to function returning 1
    W = lambda l: re.findall(r'\w+[\s|>]', l)
    Q = lambda l: [w for w in l.replace('-','').split(' ') if w != '']
    F = ctc.shape[1]
    ctc = np.vstack((np.zeros(F), ctc)) # just add an imaginative zero'th step (will make indexing more intuitive)
    T = ctc.shape[0]

    # STEP 1: Initiliazation
    O = ''
    Pb, Pnb = defaultdict(Counter), defaultdict(Counter)
    Pb[0][O] = 1
    Pnb[0][O] = 0
    A_prev = [O]
    # END: STEP 1

    # STEP 2: Iterations and pruning
    for t in range(1, T):
        pruned_alphabet = [alphabet[i] for i in np.where(ctc[t] > prune)[0]]
        for l in A_prev:
            
            if len(l) > 0 and l[-1] == '>':
                Pb[t][l] = Pb[t - 1][l]
                Pnb[t][l] = Pnb[t - 1][l]
                continue  

            for c in pruned_alphabet:
                c_ix = alphabet.index(c)
                # END: STEP 2
                
                # STEP 3: “Extending” with a blank
                if c == '-':
                    Pb[t][l] += ctc[t][-1] * (Pb[t - 1][l] + Pnb[t - 1][l])
                # END: STEP 3
                
                # STEP 4: Extending with the end character
                else:
                    l_plus = l + c
                    if len(l) > 0 and c == l[-1]:
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pb[t - 1][l]
                        Pnb[t][l] += ctc[t][c_ix] * Pnb[t - 1][l]
                # END: STEP 4

                    # STEP 5: Extending with any other non-blank character and LM constraints
                    elif len(l.replace(' ', '')) > 0 and c in (' '):
                        ngram = Q(l_plus.replace('-', ''))[-lm.N:]
                        ngram_str = ''.join(word + ' ' for word in ngram)[:-1]
                        lm_prob = lm.p(ngram_str) ** alpha
                        Pnb[t][l_plus] += lm_prob * ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    else:
                        Pnb[t][l_plus] += ctc[t][c_ix] * (Pb[t - 1][l] + Pnb[t - 1][l])
                    # END: STEP 5

                    # STEP 6: Make use of discarded prefixes
                    if l_plus not in A_prev:
                        Pb[t][l_plus] += ctc[t][-1] * (Pb[t - 1][l_plus] + Pnb[t - 1][l_plus])
                        Pnb[t][l_plus] += ctc[t][c_ix] * Pnb[t - 1][l_plus]
                    # END: STEP 6

        # STEP 7: Select most probable prefixes
        A_next = Pb[t] + Pnb[t]
        sorter = lambda l: A_next[l] * (len(W(l)) + 1) ** beta
        A_prev = sorted(A_next, key=sorter, reverse=True)[:k]
        # END: STEP 7

    return A_prev[0].strip('>').strip('-')




def test_beamsearch():
    test_file = open('LibriSpeech/test-clean-transcripts.txt').read().split('\n')[0]
    len_filename = len(test_file.split(' ')[0]) +1 
    true_output = test_file[len_filename:]
    ctc_mat = np.load('ctc_mat.npy')[:,0,:]
    df = pd.DataFrame(ctc_mat, columns=[ix_to_char[ix] for ix in range(len(alphabet))])
    df.to_html('temp.html')
    corpus = 'LibriSpeech/all_sentences.txt'
    #LM = NgramLanguageModel(corpus, 3, 'words', False, 'Kneser Ney')
    #LM.save('testLM.lm') 
    wordLM = NgramLanguageModel.load(None, 'LibriClean_trigram.lm')
    #charLM = NgramLanguageModel.load(None, 'LibriClean_char_trigram.lm')
    #pred_output = word_beam_search(ctc_mat, wordLM, k=200)
    pred_output = prefix_beam_search(ctc_mat, wordLM, k=200)
    #pred_output = wordchar_beam_search(ctc_mat, charLM, wordLM)
    print('actual:', true_output, '\npredicted:', pred_output)


if __name__ == "__main__":
    test_beamsearch()