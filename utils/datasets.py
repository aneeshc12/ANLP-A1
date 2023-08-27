import torch
from torch import nn 
from torch.utils.data import Dataset, DataLoader
import re
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
import random

class Q1Dataset(Dataset):
    def __init__(self, embs, fivegrams, labels):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.embs = embs
        self.fivegrams = fivegrams
        self.labels = labels

    def __len__(self):
        return len(self.fivegrams)
    
    def __getitem__(self, i):
        emb = np.concatenate(self.embs[self.fivegrams[i]])
        emb = torch.from_numpy(emb).to(self.device)
        return emb, self.labels[i]

class Q1Preprocessor():
    def __init__(self, corpus_path, emb_path='/home/aneesh/UbuntuStorage/Homework/ANLP/ANLP-A1/data/glove.6B.50d.txt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load emb, and vocab
        f = open(emb_path, 'r')
        lines = f.readlines()
        f.close()

        self.vocab = []
        self.word2idx = {}
        self.idx2word = {}
        embs = []
        for i, line in enumerate(lines):
            a, b = line.split(' ', 1)
            self.vocab.append(a)

            self.word2idx[a] = i
            self.idx2word[i] = a

            embs.append(np.fromstring(b, dtype=np.float32, sep=' '))
        self.embs = np.stack(embs)       # pad eos sos pre included in the default embeddings, as 0, rand and rand

        # load vocab
        self.corpus_path = corpus_path
        f = open(corpus_path, 'r')
        text = f.read()
        f.close()

        text = text.lower()
        text = re.sub('\n+', ' ', text)
        text = re.sub('([.!?])', r'\1\n', text)
        self.lines = re.split('\n', text)
        self.lines = [word_tokenize(k) for k in self.lines]

        # convert lines to indexes
        print("Getting index lines...")
        self.index_lines = []
        for line in tqdm(self.lines):
            if len(line) == 0:
                pass

            index_line = []
            for word in line:
                try:
                    index = self.word2idx[word]
                except KeyError:
                    index = self.word2idx["<unk>"]
                index_line.append(index)
            self.index_lines.append(index_line)
        random.shuffle(self.index_lines)

        # generate 5-grams from each sentence
        print("Generating 5-grams and labels...")
        self.fivegrams = []
        self.labels = []
        for index_line in tqdm(self.index_lines):
            padded = [self.word2idx["<pad>"]] * 4 + [self.word2idx["<sos>"]] + index_line + [self.word2idx["<eos>"]]
            for i in range(6, len(padded)):
                self.fivegrams.append(padded[i-6:i-1])
                self.labels.append(padded[i])

    def get_dicts(self):
        return self.word2idx, self.idx2word
    
    def get_splits(self, val_percent=0.2, test_percent=0.1):
        numgrams = len(self.fivegrams)
        train_idx = [0, int(numgrams * (1-val_percent-test_percent))]
        val_idx = [train_idx[1], train_idx[1] + int(numgrams * (val_percent))]
        test_idx = [val_idx[1], numgrams]

        trainset = Q1Dataset(self.embs, self.fivegrams[train_idx[0] : train_idx[1]], self.labels[train_idx[0] : train_idx[1]])
        valset = Q1Dataset(self.embs, self.fivegrams[val_idx[0] : val_idx[1]], self.labels[val_idx[0] : val_idx[1]])
        testset = Q1Dataset(self.embs, self.fivegrams[test_idx[0] : test_idx[1]], self.labels[test_idx[0] : test_idx[1]])

        return trainset, valset, testset
    


# q = Q1Preprocessor('/home/aneesh/UbuntuStorage/Homework/ANLP/ANLP-A1/data/Auguste_Maquet.txt')
# print(len(q.labels))