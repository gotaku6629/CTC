
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np


class BasicModule(nn.Module):
    def __init__(self, inDim, outDim, dp_rate = 0.3):
        super(BasicModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inDim, outDim), # (1024, 33)
            nn.BatchNorm1d(outDim),
            nn.ReLU(),
            nn.Dropout(p = dp_rate)
        )
    def forward(self, x):
        return self.layers(x)


class LSTM_cell(nn.Module):
    def __init__(self, args, _BaseModule=BasicModule):
        super(LSTM_cell, self).__init__()
        self.args = args

        ## Store required sizes
        self.input_size = args.extract_feat_dim # 1024        
        self.v_class = args.v_class # 33
        self.batch_size = args.batch_size # 10 
        self.temporal = args.temporal # 10

        ## extract node features from I3D features
        self.v = _BaseModule(self.input_size, self.v_class) # 1024→33

        ## The LSTM cell
        self.v_cell = nn.LSTMCell(self.v_class, self.v_class)

    
    def forward(self, feat, v_hsn, v_csn): 
        # feat = [temporal,batch,1024]
        # hidden state = [batch,1024]
        # cell state   = [batch,1024]
            
        v_series = Variable(torch.zeros(self.temporal, self.batch_size, self.v_class)).cuda()

        for time in range(self.temporal):

            v = self.v(feat[time]) # 1024→33
            v_hsn, v_csn = self.v_cell(v, (v_hsn, v_csn))
            v_series[time] = v_hsn

        return v_series