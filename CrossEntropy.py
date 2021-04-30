"""
    ラベルスムージングを行うために, 自分でCrossEntropyを自作していじります!!
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.small_value = 1e-4
        self.lamda = 0.9
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        # output = [batch, n_label]
        # target = [batch, n_label]に変更
        batch_size = target.shape[0]
        n_label = target.shape[1]

        output = self.softmax(output)
        loss = torch.zeros(batch_size).cuda()
        for b in range(batch_size):
            loss[b] = torch.log(torch.sum(torch.exp(output[b])))
            for n in range(n_label): # 正解の数だけlossがマイナスされていく!
                if target[b][n] == 1:
                    #print('b={}, n={}, output={}'.format(b, n, output[b][n]))
                    loss[b] = loss[b] - output[b][n] 
        e = torch.mean(loss) # batchで平均
        return e

        """     このプログラムだとnonが出力されてしまう...
        #one_hot = F.one_hot(target, output.shape[1]).float()
        one_hot = torch.zeros(output.shape).cuda()
        for b in range(output.shape[0]):
            for n in range(output.shape[1]):
                if target[b] == n:
                    one_hot[b][n] = self.lamda        # ラベルスムージングします!!  
                else:
                    one_hot[b][n] = (1-self.lamda)/output.shape[1]
        input_flattened = output.view(-1)
        log_input = torch.log(input_flattened)
        target_flattened = one_hot.view(-1)
        e = torch.matmul(log_input, target_flattened)
        return -e
        """