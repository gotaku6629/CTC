"""
program test
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
#from NoBlankBinaryCTC import NoBlankCTC
import math


"""  Accuracy default :  Computes the precision@k for the specified values of k """
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) # get index of until topk    
    pred = pred.t()

    correct = torch.zeros(*pred.shape)
    #print('pred=',pred)
    for i in range(correct.shape[0]):
        for j in range(correct.shape[1]):            
            #print('pred=', pred[i,j])
            #print('target=', target[j, pred[i, j]])
            correct[i, j] = target[j, pred[i, j]] > 0.5
            #print('correct=', correct[i,j])

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], correct[:1].view(-1).float()


def accuracy_time(output, target, time, topk=(1,)):
    """ バッチごとに入力時刻(temporal=10)分の精度を測定
        outputの長さとtargetの長さの違いを考慮!!       """

    # output = [temporal, class]  # temporal = 4
    # target = [temporal, class]      # time = 3
    maxk = max(topk)
    temporal = output.size(0)
    _, pred = output.topk(maxk, 1, True, True) # [temporal, topk]  
    pred = pred.t() # [topk, temporal]
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    current_id = torch.zeros(pred.shape[0]) # 現在が正解ラベルにおけるどこにいるかのindex
    correct = torch.zeros(*pred.shape)

    for i in range(correct.shape[0]): # topk
      for j in range(correct.shape[1]): # temporal
          for t in range(time): # time(実際のラベルの長さ)
              if t < current_id[i]: continue
              if target[t, pred[i, j]] > 0.5:
                  correct[i, j] = 1
                  current_id[i] = t
                  break
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / temporal))
    return res[0], res[1], correct[:1].view(-1).float()


def accuracy_s(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    temporal = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / temporal))
    return res[0], res[1], correct[:1].view(-1).float()



"""  main :  set input, label and compute Accuracy """
sigmoid = nn.Sigmoid()
yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2], [-0.1, 1.2, 0.4, 2.5, 3.2]],
        [[1.2, 2.3, 1.4, -0.5, 2.2], [-0.1, 1.2, 0.4, 2.5, 3.2]],
         [[0.5, 1.3, 2.2, 0.1, 2.4], [1.1, 2.2, 0.7, 1.4, 2.1]],
         [[0.8, -1.5, 2.3, 1.2, 2.4], [0.9, 1.4, 0.6, 2.3, 1.0]],
         [[0.2, 2.7, 1.3, 2.2, 2.4], [0.2, 1.0, 1.6, 1.3, 1.2]]]
#yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2]],
#         [[0.5, 1.3, 2.2, 0.1, 2.4]],
#         [[0.8, -1.5, 2.3, 1.2, 2.1]],
#         [[0.2, -1.0, 1.3, 2.2, 0.1]]]
yseqs = torch.tensor(yseq).cuda()
yseqs_sigmoid = sigmoid(yseqs)
print('yseqs_sigmoid=',yseqs_sigmoid.shape) # [time, batch, class]

#label = [[1, 4, 3, 3], [4, 1, 0, 3]]
#label = [[1, 4, 3, 3]]
label = [[[0,1,0,0,0], [0,0,0,1,0]],
         [[0,0,0,0,1], [0,1,0,0,0]],
         [[0,0,0,1,0], [1,0,0,0,0]],
         [[0,0,0,1,0], [0,0,0,1,0]]]
#label = [[[0,1,0,0,0]], [[0,0,0,0,1]], [[0,0,0,1,0]], [[0,0,0,1,0]]]
labels = torch.tensor(label).cuda()
print('labels=',labels.shape) # [time, batch, class]
#print(accuracy(yseqs_sigmoid[0], labels[0], topk=(1,3))) # 1時刻を取り出したAccuracy


# ここから動的時間伸縮法(DTW)の単体テスト
output = yseqs_sigmoid.transpose(0,1)
print('output={}'.format(output.shape))  # [batch, temporal, class]
dtw_label = [[[0,1,0,0,0], [0,0,0,1,0]],
            # [[0,1,0,0,0], [0,1,0,0,0]],
             [[0,0,0,0,1], [1,0,0,0,0]],
            # [[0,0,0,0,1], [1,0,0,0,0]],
             [[-1,-1,-1,-1,-1], [1,0,0,0,0]]] 
dtw_labels = torch.tensor(dtw_label).cuda()
dtw_target = dtw_labels.transpose(0,1)
print('dtw_target={}'.format(dtw_target.shape)) # [batch, time, class]

print('output:\n', output[0]) # [temporal, class]
print('dtw_target:\n', dtw_target[0]) # [time, class]
print(accuracy_time(output[0], dtw_target[0], 2, topk=(1,3))) # 1 batchを取り出したAccuracy

"""
batch_size = output.shape[0]
temporal = output.shape[1]

o_prec1_var, o_prec5_var, o_prec1_output_var = 0, 0, 0
for b in range(batch_size):
  o_prec1, o_prec5, o_prec1_output = accuracy_time(output[b], dtw_target[b], topk=(1,3))
  o_prec1_var += o_prec1
  o_prec5_var += o_prec5
  o_prec1_output_var += o_prec1_output
o_prec1_var = o_prec1_var / batch_size
o_prec5_var = o_prec5_var / batch_size
o_prec1_output_var = o_prec1_output_var / batch_size
print('o_prec1={}'.format(o_prec1_var))
print('o_prec5={}'.format(o_prec5_var))
print('o_prec1_output={}'.format(o_prec1_output_var))  # OKでしょう!!


## 予測の精度だけを見るのなら... input_length(=temporal)の中に1つでも正解があれば... いいんじゃね??
o_prec1_var, o_prec5_var, o_prec1_output_var = 0, 0, 0
future_label =  [[0,0,0,1,0], [0,0,0,1,0]]  # [batch, class]
future_labels = torch.tensor(future_label).cuda()
for t in range(temporal):
  o_prec1, o_prec5, o_prec1_output = accuracy(yseqs_sigmoid[t], future_labels, topk=(1,3))
  if o_prec1_var <= o_prec1.data:  o_prec1_var = o_prec1
  if o_prec5_var <= o_prec5.data:  o_prec5_var = o_prec5
  #if o_prec1_output_var <= o_prec1_output.data:  o_prec1_output_var = o_prec1_output
print('予測精度のみなら....')
print('o_prec1={}'.format(o_prec1_var))
print('o_prec5={}'.format(o_prec5_var))
#print('o_prec1_output={}'.format(o_prec1_output_var))  # OKでしょう!!



# s_labelに対して
dtw_s_label = [1, 2]
dtw_s_labels = torch.tensor(dtw_s_label).cuda()
output = yseqs_sigmoid # [temporal, batch, class]

s_prec1_var, s_prec5_var, s_prec1_output_var = 0, 0, 0
for t in range(temporal):
  s_prec1, s_prec5, s_prec1_output = accuracy_s(output[t], dtw_s_labels, topk=(1,3))
  s_prec1_var += s_prec1
  s_prec5_var += s_prec5
  s_prec1_output_var += s_prec1_output
s_prec1_var = s_prec1_var / temporal
s_prec5_var = s_prec5_var / temporal
s_prec1_output_var = s_prec1_output_var / temporal
print('s_prec1={}'.format(s_prec1_var))
print('s_prec5={}'.format(s_prec5_var))
print('s_prec1_output={}'.format(s_prec1_output_var))  # OKでしょう!!

"""


"""  テスト transposeやviewの行列計算の違い 
feat = [[[1,2,3,4], [11,12,13,14]],
          [[5,6,7,8], [15,16,17,18]],
          [[0,2,4,6], [10,12,14,16]]]
feats = torch.tensor(feat)  # (temporal, batch, n_class)
print('feats.shape={}'.format(feats.shape)) # (3, 2, 4)

for t in range(feats.shape[0]):
  oo = feats[t]
  print('oo={}'.format(oo))
  oo_a = oo.view(-1, 4, 1)
  print('oo_a={}'.format(oo_a))
  oo_b = oo.view(-1, 1, 4)
  print('oo_b={}'.format(oo_b))
  oo_ab = torch.bmm(oo_a, oo_b)
  print('oo_ab={}'.format(oo_ab))
  oo_view = oo_ab.view(-1, 4*4)
  print('oo_view={}'.format(oo_view))
"""
"""   テスト Accuracyの測定方法 
def accuracy(output, target, topk=(1,)):
    #Computes the precision@k for the specified values of k
    # output=(temporal,batch,n_label)
    # target=(temporal,batch)

    maxk = max(topk)
    # 各バッチ, 各時刻ごとのtop1のインデントを生成
    _, pred = output.topk(maxk, dim=2, largest=True, sorted=True) #(temporal,batch,1)
    #pred = pred.view(-1, batch_size) # (temporal,batch)
    pred = pred.transpose(0,1).view(-1, output.shape[0]) # (batch,temporal,1)
    # correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct = torch.zeros(*pred.shape)
    for i in range(correct.shape[0]): # batch_size
        for j in range(correct.shape[1]): # temporal
            if target[i, j] == pred[i, j]:
                correct[i, j] = 1
            #correct[i, j] = target[i, pred[i, j]] > 0.5
    #print('pred={}'.format(pred)) # 正解インデント # (batch,temporal)
    #print('correct={}'.format(correct)) # 正解なら1, 間違っていれば0
    res = []
    correct_k = correct.view(-1).float().sum(0, keepdim=True)
    print('correct_k={}'.format(correct_k))
    res.append(correct_k.mul_(1 / (output.shape[1]*output.shape[0])))
    print('res={}'.format(res))
    return res[0]


m = nn.Softmax(dim=2)
yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2], [-0.1, 1.2, 0.4, 2.5, 3.2]],
         [[0.5, 1.3, 2.2, 0.1, 2.4], [1.1, 2.2, 0.7, 1.4, 2.1]],
         [[0.8, -1.5, 2.3, 1.2, 2.1], [0.9, 1.4, 0.6, 2.3, 1.0]],
         [[0.2, -1.0, 1.3, 2.2, 0.1], [0.2, 1.0, 1.6, 1.3, 1.2]]]
yseqs = torch.tensor(yseq).cuda()
yseqs_softmax = m(yseqs)
print(yseqs_softmax)
label = [[1, 4, 3, 3], [4, 1, 0, 3]]
labels = torch.tensor(label).cuda()
#print('yseqs.shape={}'.format(yseqs.shape))
#print('labels.shape={}'.format(labels.shape))

print(accuracy(yseqs_softmax, labels, topk=(1,)))
"""

# テスト ver2 : CTC Lossの前向き推論(動的計画法)
""" CTC Loss
    self.CTCLoss(Log_probs, Targets, Input_lengths, Target_lengths)
    1. Log_probs      = [T(input_sequence_length), N(batch_size), C(num_class)]
    2. Targets        = [N(batch_size), S(max_target_length)]
    3. Input_lengths  = [N(batch_size)] # ここは10時刻
    4. Target_lengths = [N(batch_size)] # 実際のラベルの長さ

# input_length=4, batch_size=1, num_class=5
#yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2], [-0.1, 1.2, 0.4, 2.5, 3.2]],
#         [[0.5, 1.3, 2.2, 0.1, 2.4], [1.1, 2.2, 0.7, 1.4, 2.2]],
#         [[0.8, -1.5, 2.3, 1.2, 2.1], [0.9, 1.4, 0.6, 2.3, 1.0]],
#         [[0.2, -1.0, 1.3, 2.2, 0.1], [0.2, 1.0, 1.6, 1.3, 1.2]]]
yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2]],
         [[0.5, 1.3, 2.2, 0.1, 2.4]],
         [[0.8, -1.5, 2.3, 1.2, 2.1]],
         [[0.2, -1.0, 1.3, 2.2, 0.1]]]
#label = [[2, 3, 4],　# [batch, max_time_label]
#         [1, 2, 0]]
#label = [[2, 3, 4]]
label = [[[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]]]  # [batch, max_target_length, class]
#input_length = [4, 4]
input_length = [4]
#label_length = [3, 2]
label_length = [3]
yseqs = torch.tensor(yseq).cuda()
labels = torch.tensor(label).float().cuda()
input_lengths = torch.tensor(input_length).cuda()
label_lengths = torch.tensor(label_length).cuda()

ctc_loss = NoBlankCTC()
loss = ctc_loss(yseqs, labels, input_lengths, label_lengths)
print(loss)
"""

"""  テスト : CTC Lossの前向き推論(動的計画法)
def flip_path_probability(prob, input_length, path_length):
  seq, n_batch, n_label = prob.shape
  rotate_input = ((torch.arange(seq, dtype=torch.int32)[:, None] + input_length) % seq)
  print('rotate_input.shape={}'.format(rotate_input.shape))
  rotate_label = ((torch.arange(n_label, dtype=torch.int32) + path_length[:, None]) % n_label)
  print('rotate_label.shape={}'.format(rotate_label.shape))
  new_prob = torch.zeros(seq, n_batch, n_label)
  for s in range(seq):
    for b in range(n_batch):
      for n in range(n_label):
        new_prob[s][b][n] = prob[rotate_input[seq-s-1][b]][b][rotate_label[b][n_label-n-1]]
  return new_prob

def flip_label_probability(y, input_length):
  seq, n_batch, n_vocab = y.shape
  rotate = (torch.arange(seq, dtype=torch.int32)[:, None] + input_length) % seq
  new_prob = torch.zeros(seq, n_batch, n_vocab)
  for s in range(seq):
    for b in range(n_batch):
      for v in range(n_vocab):
        new_prob[s][b][v] = y[rotate[seq-s-1][b]][b][v]
  return new_prob
  

def flip_path(path, path_length):
  n_batch, n_label = path.shape
  rotate = (torch.arange(n_label) + path_length[:, None]) % n_label
  new_path = torch.zeros((n_batch, n_label), dtype=torch.int32)
  for b in range(n_batch):
    for l in range(n_label):
      new_path[b][l] = path[b][rotate[b][n_label-l-1]]
  return new_path



def _logsumexp(a, axis=0):
  vmax = torch.max(a, axis=axis, keepdim=True)[0]
  vmax = vmax + torch.log(torch.sum(torch.exp(a - vmax), dim=axis, keepdim=True, dtype=a.dtype))
  return torch.squeeze(vmax, axis=axis)



def computes_transition(t, prev_prob, path, path_length, cum_prob, y):
  n_batch, max_path_length = path.shape
  zero_padding = -10000.0

  mat = torch.full((2, n_batch, max_path_length), zero_padding, dtype=y.dtype)
  mat[0, :, :] = prev_prob
  if t > 0:
    mat[1, :, 1:] = prev_prob[:, :-1]

  prob = _logsumexp(mat, axis=0) # (n_batch, max_path_length)
  outside = torch.arange(max_path_length) >= path_length[:, None]
  prob[outside] = zero_padding
  cum_prob += prob

  for b in range(n_batch):
    for l in range(max_path_length):
      prob[b][l] = prob[b][l] + y[b][path[b][l]]
  print('prob={}'.format(prob))
  return prob


def calc_trans(yseq, input_length, path, path_length):
  max_input_length, n_batch, n_unit = yseq.shape
  max_path_length = path.shape[1]
  zero_padding = -10000.0

  forward_prob = torch.full((n_batch, max_path_length), zero_padding, dtype=yseq.dtype)
  forward_prob[:, 0] = 0
  backward_prob = forward_prob

  prob = torch.zeros(max_input_length, n_batch, max_path_length)
  for t in range(max_input_length):
    for b in range(n_batch):
      for l in range(max_path_length):
        prob[t][b][l] = yseq[t][b][path[b][l]]
  
  # forward computation
  for t, y in enumerate(yseq):
    forward_prob = computes_transition(t, forward_prob, path, path_length, prob[t], y)

  # flip matrix
  r_path = flip_path(path, path_length)
  print('r_path={}'.format(r_path))
  yseq_inv = flip_label_probability(yseq, input_length)
  print('yseq_inv={}'.format(yseq_inv))
  prob = flip_path_probability(prob, input_length, path_length)
  print('prob={}'.format(prob))

  # backward computation
  for t, y_inv, in enumerate(yseq_inv):
    backward_prob = computes_transition(t, backward_prob, r_path, path_length, prob[t], y_inv)

  return flip_path_probability(prob, input_length, path_length)

def _softmax(x):
  val = torch.exp(x - torch.max(x, axis=2, keepdims=True)[0])
  val = val / torch.sum(val, axis=2, keepdim=True)
  return val


# input_length=4, batch_size=1, num_class=5
yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2], [-0.1, 1.2, 0.4, 2.5, 3.2]],
         [[0.5, 1.3, 2.2, 0.1, 2.4], [1.1, 2.2, 0.7, 1.4, 2.2]],
         [[0.8, -1.5, 2.3, 1.2, 2.1], [0.9, 1.4, 0.6, 2.3, 1.0]],
         [[0.2, -1.0, 1.3, 2.2, 0.1], [0.2, 1.0, 1.6, 1.3, 1.2]]]
#yseq = [[[1.2, 2.3, 1.4, -0.5, 2.2]],
#         [[0.5, 1.3, 2.2, 0.1, 2.4]],
#         [[0.8, -1.5, 2.3, 1.2, 2.1]],
#         [[0.2, -1.0, 1.3, 2.2, 0.1]]]
label = [[2, 3, 4],
         [1, 2, 0]]
#label = [[2, 3, 4]]
input_length = [4, 4]
#input_length = [4]
label_length = [3, 2]
#label_length = [3]
yseqs = torch.tensor(yseq)
labels = torch.tensor(label)
input_lengths = torch.tensor(input_length)
label_lengths = torch.tensor(label_length)

yseq_softmax = _softmax(yseqs)
prob_trans = calc_trans(yseq_softmax, input_lengths, labels, label_lengths)

loss = -_logsumexp(prob_trans[0], axis=1)
print(loss)
"""

"""
### テスト1
a = torch.zeros([6, 6])
b = torch.ones([6, 6])
print('a.shape={}'.format(a.shape))
print('b.shape={}'.format(b.shape))

# (2,2)のテンソルを4つ加えて, (4,2,2)にするぞ!!
# unsqueezeが次元を上げてくれる!!
c = torch.cat([torch.randn(2,2).unsqueeze(0) for _ in range(4)], dim =0)
print('c.shape={}'.format(c.shape)) # torch.Size([4, 2, 2]) 



### テスト2
rgb_feat = torch.randn(3,2,3)
print('rgb_feat={}'.format(rgb_feat))

for i in range(rgb_feat.shape[0]):
    print('rgb_feat[', i, ']={}'.format(rgb_feat[i]))
    #tensor([[ 0.0924, -1.3739,  1.4417],  # (2,3)
    #        [ 0.7881,  1.3122, -0.9455]])
    print('rgb_feat[', i, '].unsqueeze(0)={}'.format(rgb_feat[i].unsqueeze(0)))
    #tensor([[[ 0.0924, -1.3739,  1.4417], # (1, 2, 3)
    #         [ 0.7881,  1.3122, -0.9455]]])

s = torch.cat([rgb_feat[i].unsqueeze(0) for i in range(3)])
print('s={}'.format(s))



### テスト3
rgb_feat = torch.randn(3,2,3)
print('rgb_feat={}'.format(rgb_feat))
ss = torch.cat([rgb_feat[i].unsqueeze(0) for i in range(2)])
print('ss={}'.format(ss))
ss_2 = rgb_feat[2].unsqueeze(0)
print('ss_2={}'.format(ss_2))
ss = torch.cat([ss, ss_2], dim=0)
print('ss={}'.format(ss))  # OK



### テスト4
hwc_img = torch.rand(25,3,3,10,224,224)
print('hwc_img.shape={}'.format(hwc_img.shape))
whc_img = hwc_img.transpose(0,1)
print('whc_img.shape={}'.format(whc_img.shape))



### テスト5
x = 8589934624
print(bin(x))
print(int('001000000000000000000000000000100000', 2))

y = [0,0,0,1,1,0]
y_index = np.argmax(y)
print('y = ', y)
print('y_index = ', y_index)



### テスト6
List = []
for i in range(10):
  List.append(i)
print('List = {}'.format(List))

List.pop(-1)
print('List = {}'.format(List))



### テスト7 : CTC Loss Test (Target are to be un-padded)
## cf) 「PytorchでCNNsを徹底解説」 https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
## cf) 「Pytorch optim SGD徹底解説」https://qiita.com/mathlive/items/2c67efa2d451ea1da1b1
  # 注意点 : if use CTCLoss, set log_softmax before!!
    #input = F.log_softmax(input, dim=2)
    # softmax = nn.Softmax(dim=2)  
    # input = softmax(input)

T = 21   # Input sequence length 
  # time = {0, 1, 2, 3, 4}
  # input = { 0[s]='walk', 1[s]='walk', 2[s]='walk', 3[s]='hold', 4[s]='sit'}
C = 16   # Number of classes (include blank)
  # {0:'walk', 1:'put', 2:'open', 3:'hold', 4:'play', 5:'check', 6:'sit', 7:'close', 8:'run', 9:'_(none)'}
N = 2   # Batch size

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1,1,C)
    #self.relu = nn.ReLU()

  def forward(self, x):
    x = self.conv1(x)
    #x = self.relu(x)
    return x

device = torch.device("cuda:0")
net = Net()
net = net.to(device)
ctc_loss = nn.CTCLoss()
input = torch.randn(T, N, C).log_softmax(2).to(device)
optimizer = optim.SGD([input], lr=0.001, momentum=0.9)

for epoch in range(10):
  print('epoch={}'.format(epoch))
  #input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
  input = torch.randn(T, N, C).log_softmax(2).to(device)
  if epoch == 0:
    print('input = {}'.format(input))
  #input = torch.tensor([[[0.321, 0.022, 0.012, 0.098, 0.012, 0.028, 0.009, 0.012, 0.002, 0.009]],
  #                      [[0.321, 0.022, 0.012, 0.098, 0.012, 0.028, 0.009, 0.012, 0.002, 0.009]],
  #                      [[0.321, 0.022, 0.012, 0.098, 0.012, 0.028, 0.009, 0.012, 0.002, 0.009]], 
  #                      [[0.098, 0.022, 0.012, 0.498, 0.012, 0.028, 0.009, 0.012, 0.002, 0.009]],
  #                      [[0.021, 0.022, 0.012, 0.098, 0.012, 0.028, 0.753, 0.012, 0.002, 0.009]]
  #                     ])

  input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(device)
  if epoch == 0:
    print('input_lengths = {}'.format(input_lengths))

  target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long).to(device)
  if epoch == 0:
    print('target_lengths = {}'.format(target_lengths))
  #target_lengths = torch.tensor([3], dtype=torch.long)

  target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long).to(device)
  if epoch == 0:
    print('target = {}'.format(target))
  #target = torch.tensor([0, 3, 6], dtype=torch.long)

  optimizer.zero_grad()
  input_var = torch.autograd.Variable(input, requires_grad=True)

  #ctc_loss = nn.CTCLoss(blank=9)
  loss = ctc_loss(input_var, target, input_lengths, target_lengths)
  print('loss = {}'.format(loss))
  loss.backward()
  optimizer.step()
"""