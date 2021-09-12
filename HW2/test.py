import os
import pickle
import torch
import numpy as np
from math import ceil
from model import Classifier
from data_loader_test import get_loader
import torch.nn.functional as F
import csv

lstm_input_size = 300
hidden_state_size = 128
num_sequence_layers = 2
output_dim = 1
rnn_type = 'LSTM'
batch_size = 1
model_id = 'bilstm_10000'
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

data_loader = get_loader(1)
C = Classifier(lstm_input_size, hidden_state_size, batch_size, output_dim=output_dim, num_layers=num_sequence_layers, rnn_type=rnn_type).eval().to(device)

c_checkpoint = torch.load('checkpoint/' + model_id + '/C.ckpt')
C.load_state_dict(c_checkpoint)
first = True
with open('309505001.csv','w') as f:
    for i in range(len(data_loader)):
        if first:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Label'])
            first = False
        try: 
            x_real,id = next(data_iter)
        except:
            data_iter = iter(data_loader)
            x_real,id = next(data_iter)
                
            
        x_real = x_real.to(device)
        y_pred = C(x_real)
        writer.writerow([id.item(), y_pred.item()])
        

