import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Q1Model(nn.Module):
    def __init__(self, word_emb_dim, vocab_size, hidden_size=300):
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size
        self.ngram = 5
        self.hidden_size = 500

        self.fc1 = nn.Linear(self.word_emb_dim * self.ngram, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.vocab_size)
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        x = self.softmax(x)

        return x
    