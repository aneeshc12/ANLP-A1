from ..src.q1_model import *
from ..utils.datasets import *

import torch
import numpy as np
from torch.utils.data import DataLoader

# init dataset
preproc = Q1Preprocessor('/home/aneesh/UbuntuStorage/Homework/ANLP/ANLP-A1/data/Auguste_Maquet.txt')
trainset, valset, testset = preproc.get_splits()
word2idx, idx2word = preproc.get_dicts()

trainloader = DataLoader(trainset, batch_size=8, shuffle=True)
valloader = DataLoader(valset, batch_size=8, shuffle=True)
testloader = DataLoader(testset, batch_size=8, shuffle=True)

# init model, optim and training params
model = Q1Model(50, len(word2idx))
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# num_epochs = 5
# for epoch in num_epochs:
#     for 

for embs, labels in trainloader:
    print(embs.shape, labels.shape)
    exit(0)