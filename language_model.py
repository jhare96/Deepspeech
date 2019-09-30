from collections import defaultdict
import re
import pickle
import time
import numpy as np


class NgramTree:
    def __init__(self, N, ngram_type='words'):
        self.N = N
        types = ['words', 'characters']
        if ngram_type in types:
            self.ngram_type = ngram_type
            if ngram_type == 'words':
                self.splitter = self.word_splitter
                self.joiner = self.word_joiner
            elif ngram_type == 'characters':
                self.splitter = self.char_splitter
                self.joiner = self.char_joiner
        else:
            raise ValueError('ngram_type "'"{}"'" is not a valid type, valid types are: {}'.format(ngram_type, types))
        
        self._root = NgramNode(parent=None, name='-rootNode-', count=0, splitter=self.splitter, joiner=self.joiner, level=0)
        self._unq_counts = {}
    
    # use functions instead of lambdas in order to pickle
    def word_splitter(self, line):
        return line.split(' ')
    
    def word_joiner(self, line):
        # line type list 
        return ''.join(word + ' ' for word in line)[:-1] #exclude last space
    
    def char_splitter(self, line):
        return list(line)
    
    def char_joiner(self, line):
        # line type list 
        return ''.join(char for char in line)
    
    def update(self, gram, count=0):
        units = self.splitter(gram)
        if len(units) > self.N:
            raise ValueError('length of grams > {} for {}-gram model'.format(self.N,self.N))
        node = self._root
        for unit in units:
            try: 
                node[unit]
            except KeyError:
                node._children[unit] = NgramNode(node, unit, 0, self.splitter, self.joiner, node.level+1)
            node = node[unit]
            #print('node', node.name)
            
        node.count += count
    
    def remove_node(self, gram):
        node = self.node_from_str(gram)
        parent = node.parent
        del parent._children[self.splitter(gram)[-1]]
    
    def node_from_str(self, gram):
        units = self.splitter(gram)
        node = self._root
        for unit in units:
            node = node[unit] # will raise KeyError if node does not exist 
        return node
    
    def compute_unique_counts(self,): 
        # calculates unique number for all n-grams in tree,
        # e.g. n=1 unique unigrams, n=2 unique bigrams, ... n=N unique N-grams
        for n in range(1,self.N+1):
            self._unq_counts[n] = self._root._unique_count(n)
    
    def compute_recursive_counts(self):
        def _compute_recursive_counts(node):
            node.recursive_count = node.get_count()
            for child in node._children.values():
                _compute_recursive_counts(child)
        
        _compute_recursive_counts(self._root)
                
    
    def kneser_ney_smoothing(self, delta=0.7):
        self.compute_unique_counts()
        self.compute_recursive_counts()
        def _kneser_ney_smoothing(node):
            for child in node._children.values():
                child._p = child._recursive_KneserNey(self._unq_counts, delta)
                _kneser_ney_smoothing(child)
                
        return _kneser_ney_smoothing(self._root)
            
       
    
    def unsmoothed_probs(self):
        def _unsmoothed(node):
            for child in node._children.values():
                child._p = child.get_count()/child.parent.get_count()
                _unsmoothed(child) 
        return _unsmoothed(self._root)
    
        
    def sum_prob(self):
        p=0
        for child in self._root._children:
            p+= child._p
        return p
    
    def bottom_nodes(self, node, list_):
        for child in node._children.values():
            if child.level == self.N:
                list_.append(child)
            else:
                self.bottom_nodes(child, list_)
        return list_
                
            
        
    def __str__(self):
        return self._root.__str__()
    
    def __getitem__(self, gram):
        return self._root.__getitem__(gram)
    
    def __setitem__(self, gram):
        return self._root.__setitem__(gram) 
    
    
        
    
class NgramNode:
    def __init__(self, parent, name='', count=0, splitter=None, joiner=None, level=0):
        self.parent = parent
        self._children = {}
        self.name = name
        self.count = count
        self.level = level
        self.recursive_count = 0
        self._p = 0
    
    def full_name(self, names):
        node = self
        while node.parent is not None:
            names.append(node.name)
            node = node.parent
        return names[:0:-1]
        
    
    def __str__(self, level=0):
        out = "----"*level+"'"+self.name+"' c="+str(self._recursive_count()) + ' p=' + str(format(self._p, '.5f'))+"\n"
        for child in self._children.values():
            out += child.__str__(level+1)
        return out
    
    def get_count(self):
        return self._recursive_count()
    
    def _recursive_KneserNey(self, unq_counts, delta=0):
        c = self.recursive_count
        if self.level == 0:
            return 0
        else:
            if self.parent.recursive_count > 0:
                discounted_prob = max(c-delta, 0)/self.parent.recursive_count
            else:
                discounted_prob = 0
            #print('discount', discounted_prob)
            norm_coeff = delta*(self.parent.unique_following()/unq_counts[self.level])
            #print('norm_coeff', norm_coeff)
            #print('parent {}, level {}, name {}'.format(self.parent.name,self.level, self.name))
            return discounted_prob + norm_coeff * self.parent._recursive_KneserNey(unq_counts, delta)
            #print('p', self._p, self.name)
        
    def _recursive_count(self):
        count = self.count
        for child in self._children.values():
            count += child._recursive_count()
        return count
    
    def _unique_count(self, n):
        if self.level == n-1:
            ucount = len(self._children)
        else:
            ucount = 0
        for child in self._children.values():
            ucount += child._unique_count(n)
        return ucount
    
    def unique_following(self):
        # compute the number of unique words or chars following this sequence
        return len(self._children)
        
    
    def __getitem__(self, gram):
        if len(self._children) == 0:
            raise KeyError
        else:
            try:
                node = self._children[gram]
                return node
            except:
                raise KeyError ('node {} as no key {}'.format(self.full_name(['']), gram))


def word_Ngrams(line, N):
    ngrams = []
    line = line.split(' ')
    for i in range(len(line)):
        ngrams.append(''.join(word + ' ' for word in line[i:i+N])[:-1])
    return ngrams

def char_Ngrams(line, N):
    ngrams = []
    line = list(line)
    for i in range(len(line)):
        ngrams.append(line[i:i+N])
    return ngrams
    


class NgramLanguageModel:
    def __init__(self, corpus, N, ngram_type='words', grammar=False, smoothing='Kneser Ney'):
        self.N = N
        types = ['words', 'characters']
        if ngram_type in types:
            self.ngram_type = ngram_type
            if ngram_type == 'words':
                self.ngrams = word_Ngrams
            elif ngram_type == 'characters':
                self.ngrams = char_Ngrams
        else:
            raise ValueError('ngram_type "'"{}"'" is not a valid type, valid types are: {}'.format(ngram_type, types))
        
        smoothing_types = ['None', 'Kneser Ney']
        self.smoothing = smoothing
        if smoothing not in smoothing_types:
            raise ValueError('smoothing "'"{}"'" is not supported,supported smoothing are {}'.format(smoothing, smoothing_types))
        
        self.tree = NgramTree(N=self.N, ngram_type=ngram_type)
        start = time.time()
        text = open(corpus, 'r').read().split('\n')[:-1]
        print('load text time', time.time() - start)
        
        start = time.time()
        for line in text:
            if not grammar:
                line = re.sub(r'[^a-z ]', '', line.lower())
            for gram in self.ngrams(line, self.N):
                self.tree.update(gram, 1)
        print('tree creation time', time.time() - start)
        
        start = time.time()
        if smoothing == 'None':
            self.tree.unsmoothed_probs()
        elif smoothing == 'Kneser Ney':
            self.tree.kneser_ney_smoothing()
        print('tree prob smoothing time', time.time() - start)



    def p(self, x):
        ''' probability of word given N-1 previous words p(w_k|w_{k-1}, ..., w_{k-N-1})
            for characters Ngrams spaces will be counted as characters
            
            args: 
                str x - string of words or chars with spaces which prob is to be calculated     
        '''
        x = self.tree.splitter(x)
        if len(x) > self.N:
            raise ValueError('cannot compute the probability sequence of length {} for a {}-gram model'.format(len(x), self.N))
        
        try: #if sequence is seen in corpus 
            node = self.tree.node_from_str(self.tree.joiner(x))
            p = node._p
        except KeyError: # backoff prob
            #print('x', x)
            if self.smoothing == 'Kneser Ney' and len(x) > 1:
                xjoined = self.tree.joiner(x)
                self.tree.update(xjoined, 0) #extend tree to include this new sequence with count 0
                node = self.tree.node_from_str(xjoined)
                p = node._recursive_KneserNey(self.tree._unq_counts, .7)
                self.tree.remove_node(xjoined)
                #print(self.tree)
            else:
                p = 0 
        return p
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            LM = pickle.load(f)
        return LM



        

                

        





if __name__ == '__main__':
    corpus = 'LibriSpeech/all_sentences.txt'
    LM = NgramLanguageModel(corpus, 7, 'characters', False, 'Kneser Ney')
    LM.save('LibriClean_char_trigram.lm')
    