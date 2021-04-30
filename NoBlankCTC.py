"""
    pytorch ctcLoss()だと, blankを考える必要があるが, 今回はblank無しでいきます!!
    /Gated_Temporal-Energy-Graph_ctcloss_ver2/
"""
import torch
import torch.nn as nn
import numpy as np


def _softmax(x):
  val = torch.exp(x - torch.max(x, axis=2, keepdims=True)[0])
  val = val / torch.sum(val, axis=2, keepdim=True)
  return val


def _logsumexp(a, axis=0):
  vmax = torch.max(a, axis=axis, keepdim=True)[0]
  vmax = vmax + torch.log(torch.sum(torch.exp(a - vmax), dim=axis, keepdim=True, dtype=a.dtype))
  return torch.squeeze(vmax, axis=axis)


class NoBlankCTC(nn.Module):
    def __init__(self):
        super(NoBlankCTC, self).__init__()
        self.zero_padding = -10000000000000.0 # 4.91044616699219
        #self.logsoftmax = nn.LogSoftmax()
    

    def flip_path(self, path, path_length):
        """Flips label sequence.
        This function rotates a label sequence and flips it.
        ``path[b, t]`` stores a label at time ``t`` in ``b``-th batch.
        The rotated matrix ``r`` is defined as
        ``r[b, t] = path[b, t + path_length[b]]``
        .. ::
        a b c d .     . a b c d    d c b a .
        e f . . .  -> . . . e f -> f e . . .
        g h i j k     g h i j k    k j i h g
        """
        rotate = (torch.arange(self.max_path_length).cuda() + path_length[:, None]) % self.max_path_length
        new_path = torch.zeros((self.n_batch, self.max_path_length), dtype=torch.int32).cuda()
        for b in range(self.n_batch):
            for l in range(self.max_path_length):
                new_path[b][l] = path[b][rotate[b]][self.max_path_length-l-1]
        return new_path


    def flip_label_probability(self, y, input_length):
        rotate = (torch.arange(self.max_input_length, dtype=torch.int32)[:, None].cuda() + input_length) % self.max_input_length
        new_prob = torch.zeros(self.max_input_length, self.n_batch, self.n_unit).cuda()
        for s in range(self.max_input_length):
            for b in range(self.n_batch):
                for v in range(self.n_unit):
                    new_prob[s][b][v] = y[rotate[self.max_input_length-s-1][b]][b][v]
        return new_prob


    def flip_path_probability(self, prob, input_length, path_length):
        #self.max_input_length, self.n_batch, self.max_path_length
        #seq, n_batch, n_label = prob.shape
        rotate_input = ((torch.arange(self.max_input_length, dtype=torch.int32)[:, None].cuda() + input_length) % self.max_input_length)
        rotate_label = ((torch.arange(self.max_path_length, dtype=torch.int32).cuda() + path_length[:, None]) % self.max_path_length)
        new_prob = torch.zeros(self.max_input_length, self.n_batch, self.max_path_length).cuda()
        for s in range(self.max_input_length):
            for b in range(self.n_batch):
                for n in range(self.max_path_length):
                    new_prob[s][b][n] = prob[rotate_input[self.max_input_length-s-1][b]][b][rotate_label[b][self.max_path_length-n-1]]
        return new_prob


    def computes_transition(self, t, prev_prob, path, path_length, cum_prob, y):
        # グラフの遷移:a(t-1)(s),a(t-1)(s-1)の初期化
        mat = torch.full((2, self.n_batch, self.max_path_length), self.zero_padding, dtype=y.dtype).cuda()
        mat[0, :, :] = prev_prob  # a(t-1)(s)
        if t > 0:
            mat[1, :, 1:] = prev_prob[:, :-1] # a(t-1)(s-1)

        prob = _logsumexp(mat, axis=0) # a(t-1)(s)+a(t-1)(s-1)
        outside = torch.arange(self.max_path_length).cuda() >= path_length[:, None]
        prob[outside] = self.zero_padding
        cum_prob += prob
        
        for b in range(self.n_batch):
            for l in range(self.max_path_length):
                prob[b][l] = prob[b][l] + y[b][path[b][l]]
        #print('prob={}'.format(prob))
        return prob        


    def calc_trans(self, yseq, input_length, path, path_length):
        # 動的計画法のノード:aの初期化
        forward_prob = torch.full((self.n_batch, self.max_path_length), self.zero_padding, dtype=yseq.dtype).cuda()
        forward_prob[:, 0] = 0

        # 各時刻に対応するラベルの確率
        prob = torch.zeros(self.max_input_length, self.n_batch, self.max_path_length).cuda()
        for t in range(self.max_input_length):
            for b in range(self.n_batch):
                for l in range(self.max_path_length):
                    #prob[t][b][l] = yseq[t][b][path[b][l]] # 普通だったら, 正解のみ1なので, その記号だけ考えるが...
                    # ラベルスムージングでは, 正解はx0.9, そのほかは,x0じゃなくて, x(1-λ)x1/kをかける!!
                    prob[t][b][l] = yseq[t][b][path[b][l]]#*self.lamda
                    #print('true={}'.format(path[b][l]))
                    #for n in range(self.n_unit):
                    #    if n != path[b][l]:
                            #print('false={}'.format(n))
                            #prob[t][b][l] += yseq[t][b][n]*(1-self.lamda)/self.n_unit
        
        # 前向き計算(forward_computation)
        for t, y in enumerate(yseq):
            forward_prob = self.computes_transition(t, forward_prob, path, path_length, prob[t], y)

        """
        # flip matrix
        r_path = self.flip_path(path, path_length)
        #print('r_path={}'.format(r_path))
        yseq_inv = self.flip_label_probability(yseq, input_length)
        #print('yseq_inv={}'.format(yseq_inv))
        prob = self.flip_path_probability(prob, input_length, path_length)
        #print('prob={}'.format(prob))

        # 後ろ向き計算(bachward_computation)
        for t, y_inv, in enumerate(yseq_inv):
            backward_prob = self.computes_transition(t, backward_prob, r_path, path_length, prob[t], y_inv)
        """
        return self.flip_path_probability(prob, input_length, path_length)


    def forward(self, yseq, label, input_length, target_length):
        self.max_input_length, self.n_batch, self.n_unit = yseq.shape
        self.max_path_length = label.shape[1]
        #self.lamda = 0.9

        #yseq_softmax = _softmax(yseq)
        #yseq_log = log_matrix(yseq_softmax)
        yseq = torch.nn.LogSoftmax(dim=2)(yseq)
        prob_trans = self.calc_trans(yseq, input_length, label, target_length)
        #loss = -_logsumexp(prob_trans[0], axis=1)
        loss = -prob_trans[0,:,0]
        loss_mean = torch.mean(loss)
        return loss_mean



