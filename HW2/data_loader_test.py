import torch
import torch.utils.data as data
import numpy as np
import random
import os
from random import choice
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence

class Title(data.Dataset):
    """Custom Dataset of Part A data compatible with torch.utils.data.DataLoader."""

    def __init__(self):
        
        self.model = api.load('glove-wiki-gigaword-300')#word2vec-google-news-300 
        #model['word']
        self.data_dict = self.make_data_dict()
        #self.data_dict_exp = self.make_data_exp_dict()
        print('Finished init the dataset...')
    
   
    def make_data_dict(self):
        i = 0
        with open('test.csv', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                if i == 0:
                    i = 1
                else:
                    #label = float(line.split(',')[-1].replace('\n',''))#/5.0
                    sentence = u','.join(line.split(',')[1:-2]).encode('utf-8')
                    id = int(line.split(',')[0])
                    data[id]=[sentence,id]
                
        return data
        
    '''def make_data_exp_dict(self):
        with open('dict.txt', 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            data = {}
            for line in lines:
                oriw = line.split(' ')[0]
                tanw = line.replace('\n','').split(' ')[1:]
                data[oriw]=tanw
                
        return data'''                 
           
    def __getitem__(self, index):
        data = self.data_dict[index+1]
        word = word_tokenize(data[0].decode('utf-8'))
        first = True
        for w in word:
            try:
                if first:
                    new_emb = self.model[w.lower()]
                    first = False
                else:
                    emb = self.model[w.lower()]
                    new_emb = np.vstack((new_emb, emb))
                    #print(emb.shape)
            except:
                try:
                    if first:
                        if w[0] == '\'':
                            new_emb = self.model[w[0]]
                            emb = self.model[w[1:].lower()]
                            new_emb = np.vstack((new_emb, emb))
                            #print(emb.shape)
                        elif w.find('-'):
                            neww = w.split('-')
                            new_emb = self.model[neww[0].lower()]
                            f = True
                            for nw in neww:
                                emb = self.model['-']
                                #print(emb.shape)
                                new_emb = np.vstack((new_emb, emb))
                                if f:
                                    f = False
                                    continue
                                else:
                                    emb = self.model[nw.lower()]
                                    #print(emb.shape)
                                    new_emb = np.vstack((new_emb, emb))

                        first = False
                    else:
                        if w[0] == '\'':
                            emb = self.model[w[0]]
                            #print(emb.shape)
                            new_emb = np.vstack((new_emb, emb))
                            emb = self.model[w[1:].lower()]
                            #print(emb.shape)
                            new_emb = np.vstack((new_emb, emb))
                        elif w.find('-'):
                            neww = w.split('-')
                            emb = self.model[neww[0].lower()]
                            #print(emb.shape)
                            new_emb = np.vstack((new_emb, emb))
                            f = True
                            for nw in neww:
                                emb = self.model['-']
                                new_emb = np.vstack((new_emb, emb))
                                #print(emb.shape)
                                if f:
                                    f = False
                                    continue
                                else:
                                    emb = self.model[nw.lower()]
                                    #print(emb.shape)
                                    new_emb = np.vstack((new_emb, emb))
                except:
                    #print(w)
                    try:
                        with open('test_exception.txt','a') as f:
                            f.write(w+'\n')
                    except:
                        n = 0
                    '''try:
                        if w in list(self.data_dict_exp.keys()):
                            for e in self.data_dict_exp[w]:
                                emb = self.model[e.lower()]
                                new_emb = np.vstack((new_emb, emb))
                    except:
                        try:
                            with open('dict_exception.txt','a') as f:
                                f.write(w+'\n')
                        except:
                            n = 0'''

        #max len = 50 in this dataset
        if len(new_emb) < 50:
            length = 50 - len(new_emb)
            emb = np.zeros((length,300), dtype=np.float32)
            new_emb = np.vstack((new_emb, emb))
        #print(new_emb.shape)
        return new_emb, data[1]

    def __len__(self):
        return len(self.data_dict)
        
def get_loader(batch_size=16, num_workers=0):
    """Build and return a data loader."""
    
    dataset = Title()
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader
