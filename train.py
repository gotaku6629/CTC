""" Defines the Trainer class which handles train/validation/validation_video
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
#from utils import map
from utils import get_predictions, get_ov_predictions, eval_visual_relation
import gc
import csv
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(startlr, decay_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = startlr * (0.1 ** (epoch // decay_rate))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_s(output, target, topk=(1,)):
    """ 1時刻の精度Accuracy測定 (Computes the accuracy over the k top predictions for the specified values of k)"""
    # output = [batch, class]
    # target = [batch, class]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], correct[:1].view(-1).float()


def accuracy(output, target, topk=(1,)):
    """ 1時刻の精度Accuracy測定 (Computes the precision@k for the specified values of k)"""
    # output = [batch, class]
    # target = [batch, class]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True) # [batch, topk]  
    pred = pred.t() # [topk, batch]

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = torch.zeros(*pred.shape)
    for i in range(correct.shape[0]): # topk
        for j in range(correct.shape[1]): # batch
            correct[i, j] = target[j, pred[i, j]] > 0.5

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0], res[1], correct[:1].view(-1).float()


# 遷移精度調査(Recall)
def recall_time(output, target, trans, topk=(1,)):
    """ バッチごとに入力時刻(temporal=10)分のAccuracyを測定
        outputの長さとtargetの長さの違いを考慮!!       """
    # output = [temporal, class]  # temporal = 4
    # target = [adjust_time, class]  # trans = 3
    maxk = max(topk)
    #temporal = output.size(0)
    _, pred = output.topk(maxk, 1, True, True) # [temporal, topk]  
    pred = pred.t() # [topk, temporal]
    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    current_id = torch.zeros(pred.shape[0]) # 現在が正解ラベルにおけるどこにいるかのindex
    correct = torch.zeros(5, trans) # [topk, trans]

    for i in range(correct.shape[0]): # topk
      for j in range(correct.shape[1]): # temporal
          for t in range(trans): # trans(実際のラベルの長さ)
              if t < current_id[i]: continue
              if target[t, pred[i, j]] > 0.5:
                  correct[i, t] = 1
                  current_id[i] = t
                  break
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / trans))
    return res[0], res[1], correct[:1].view(-1).float()


# 遷移精度調査(Accuracy)
def accuracy_time(output, target, time, topk=(1,)):
    """ バッチごとに入力時刻(temporal=10)分の精度を測定
        outputの長さとtargetの長さの違いを考慮!!       """
    # output = [temporal, class]  # temporal = 4
    # target = [adjust_time, class]   # time = 3
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


def accuracy_s_time(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # temporalの平均を取る!!
    # output = [temporal, class]
    # target = [class]
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



def accuracy_future(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k using time """
    # 動的伸縮法(DTW)的な感じで{o, v}ラベルについてAccuracyを測定
    # output = [temporal, class]  # temporal = 4
    # target = [class]      # time = 3  (ホントは,target=[temporal,class]だけど,timeもらって無理やり減らす!!)
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True) # [temporal, topk]  
    pred = pred.t() # [topk, temporal]

    # correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = torch.zeros(*pred.shape)
    count = torch.zeros(pred.shape[0])  # topkごとに正解数をカウントしていきます!!
    for i in range(correct.shape[0]): # topk
        for j in range(correct.shape[1]): # temporal
            #correct[i, j] = target[pred[i, j]] > 0.5
            if target[pred[i, j]] > 0.5:
                correct[i, j] = 1
                count[i] += 1  # top iのときに正解した個数
        if count[i] == 0: count[i] = 1  # 1個も正解しないと0で割ることになるので, 1にして置きます
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / count[:k+1].view(-1).float().sum(0, keepdim=True))) # 1個でも正解すればいいので, countで割ります!!
    return res[0], res[1], correct[:1].view(-1).float()


def submission_file(ids, outputs, filename):
    """ write list of ids and outputs to filename"""
    with open(filename, 'w') as f:
        for vid, output in zip(ids, outputs):
            scores = ['{:g}'.format(x)
                      for x in output]
            f.write('{} {}\n'.format(vid, ' '.join(scores)))

def gtmat(sizes, target):
    # convert target to a matrix of zeros and ones
    out = torch.zeros(*sizes)
    for i, t in enumerate(target):
        t = t.data[0] if type(t) is torch.Tensor else t
        if len(sizes) == 3:
            out[i, t, :] = 1
        else:
            out[i, t] = 1
    return out.cuda()


def write_csv(s_labels, o_labels, v_labels, writer, image_path=None, topk=(1,)):
    maxk = max(topk) # 点数の良い5番目まで
    _, s_pred = s_labels.topk(maxk, 1, True, True) # 25(=batch_size)枚のそれぞれにおいて, 上位5番目のラベル
    _, o_pred = o_labels.topk(maxk, 1, True, True)
    _, v_pred = v_labels.topk(maxk, 1, True, True)
    length = len(s_pred)
    s_pred = s_pred.t()
    o_pred = o_pred.t()
    v_pred = v_pred.t()
    
    for i in range(length): # 25枚分, 行う
        #print('write_csv:i=', i)
        print('i=', i)

        print('s_label')
        for j in range(5):
            s_index = int(s_pred[j,i].to('cpu').detach().clone().numpy())
            print('(',j+1, ')=', s_lable_name(s_index))
            print(s_labels[i][s_index])            
        print('o_label')
        for j in range(5):
            o_index = int(o_pred[j,i].to('cpu').detach().clone().numpy())
            print('(',j+1, ')=', o_lable_name(o_index))
            print(o_labels[i][o_index])
        print('v_label')
        for j in range(5):
            v_index = int(v_pred[j,i].to('cpu').detach().clone().numpy())
            print('(',j+1, ')=', v_lable_name(v_index))
            print(v_labels[i][v_index])

        writer.writerow([s_pred[0,i].to('cpu').detach().clone().numpy(), #0番目:つまり1番スコアの良いラベルを選択
                         o_pred[0,i].to('cpu').detach().clone().numpy(),
                         v_pred[0,i].to('cpu').detach().clone().numpy(),
                         image_path[i]])


def s_lable_name(label_index):
    scenes = {  0: 'Basement',
               1: 'Bathroom',
               2: 'Bedroom',
               3: 'Closet',
               4: 'Dining room',
               5: 'Entryway',
               6: 'Garage',
               7: 'Hallway',
               8: 'Office',
               9: 'Kitchen',
              10: 'Laundry room',
              11: 'Living room',
              12: 'Other',
              13: 'Pantry',
              14: 'Recreation room',
              15: 'Stairs'}
    return scenes[label_index]

def o_lable_name(label_index):
    objects = {  0: 'None',
               1: 'bag',
               2: 'bed',
               3: 'blanket',
               4: 'book',
               5: 'box',
               6: 'broom',
               7: 'chair',
               8: 'closet',
               9: 'clothes',
              10: 'cup',
              11: 'dish',
              12: 'door',
              13: 'doorknob',
              14: 'doorway',
              15: 'floor',
              16: 'food',
              17: 'groceries',
              18: 'hair',
              19: 'hands',
              20: 'laptop',
              21: 'light',
              22: 'medicine',
              23: 'mirror',
              24: 'paper',
              25: 'phone',
              26: 'picture',
              27: 'pillow',
              28: 'refrigerator',
              29: 'sandwich',
              30: 'shelf',
              31: 'shoe',
              32: 'sofa',
              33: 'table',
              34: 'television',
              35: 'towel',
              36: 'vacuum',
              37: 'window'}
    return objects[label_index]

def v_lable_name(label_index):
    verbs = {  0: 'awaken',
               1: 'close',
               2: 'cook',
               3: 'dress',
               4: 'drink',
               5: 'eat',
               6: 'fix',
               7: 'grasp',
               8: 'hold',
               9: 'laugh',
              10: 'lie',
              11: 'make',
              12: 'open',
              13: 'photograph',
              14: 'play',
              15: 'pour',
              16: 'put',
              17: 'run',
              18: 'sit',
              19: 'smile',
              20: 'sneeze',
              21: 'snuggle',
              22: 'stand',
              23: 'take',
              24: 'talk',
              25: 'throw',
              26: 'tidy',
              27: 'turn',
              28: 'undress',
              29: 'walk',
              30: 'wash',
              31: 'watch',
              32: 'work'}
    return verbs[label_index]


class Trainer():
    def train(self, loader, base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, base_optimizer, logits_optimizer, epoch, args, writer):
        adjust_learning_rate(args.lr, args.lr_decay_rate, base_optimizer, epoch)
        adjust_learning_rate(args.lr, args.lr_decay_rate, logits_optimizer, epoch)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()  # 全体のLoss
        ctc_losses = AverageMeter()  # CTC Loss
        ce_losses = AverageMeter()  # CE Loss 
        v_losses = AverageMeter()
        v_f_top1 = AverageMeter()
        v_f_top5 = AverageMeter()
        v_class = args.v_class
        batch_size = args.batch_size
        temporal = args.temporal
        alpha = args.alpha      # Scale Parametor of CrossEntropy Against CTC

        # switch to train mode
        base_model.train()   # I3D Networks
        logits_model.train() # Structural-RNN
        base_optimizer.zero_grad()
        logits_optimizer.zero_grad()


        def part(x):
            return itertools.islice(x, int(len(x)*args.train_size)) # default=0.2, but now is 1.0
        end = time.time()

        for i, (input, o_target, v_target, s_target, o_f_target, v_f_target, s_f_target, meta) in enumerate(part(loader)):
            """
            ## image → time=10時刻分
            # input = [batch, temporal, 3, 10, 224, 224]
            ## CTC label -- onehot形式(multi-label) → adjust_time=10時刻分 (最大長の長さ)
            # o_target = [batch, adjust_time, o_class]  # ここは, {O,V}で独立したラベル遷移になっている
            # v_target = [batch, adjust_time, v_class]
            # s_target = [batch, 1]
            # o_f_target = [batch, o_class] 
            # v_f_target = [batch, v_class]
            # s_f_target = [batch]
            # meta = {id:[batch], time:[batch]} # 実際の長さ {O,V,S}のそれぞれ独立した長さ!!
            """
            #torch.autograd.set_detect_anomaly(True)
            gc.collect()
            data_time.update(time.time() - end)
            meta['epoch'] = epoch

            # define input
            input_var = torch.autograd.Variable(input.cuda()) # 計算グラフ構築(require_grad=True)
            # define label
            o_target_var = torch.autograd.Variable(o_target.float().cuda()) # CTC計算用 
            v_target_var = torch.autograd.Variable(v_target.cuda()) # adjust_time = temporal
            s_target_var = torch.autograd.Variable(s_target.cuda())
            o_f_target_var = torch.autograd.Variable(o_f_target.float().cuda()) # CE計算用                                                                     
            v_f_target_var = torch.autograd.Variable(v_f_target.long().cuda()) # adjust_time = temporal                                                       
            s_f_target_var = torch.autograd.Variable(s_f_target.long().cuda())
            s_f_target = torch.tensor(s_f_target).cuda()
            v_f_target = torch.tensor(v_f_target).cuda()

            # define ctc_label
            input_length = torch.full(size=(batch_size,), fill_value=temporal, dtype=torch.long).cuda()
            o_target_length = meta['o_time'].cuda()  # 各ラベルそれぞれ独立した長さ
            v_target_length = meta['v_time'].cuda()
            s_target_length = meta['s_time'].cuda()

            # define Hidden States
            v_hidden_states_node = torch.zeros(batch_size, v_class).cuda()
            v_hidden_states_node_RNNs = torch.autograd.Variable(v_hidden_states_node)
            v_cell_states_node = torch.zeros(batch_size, v_class).cuda()
            v_cell_states_node_RNNs = torch.autograd.Variable(v_cell_states_node)

            # 1. I3D Networks
            #print('3.1 base_model') #[batch,time,3,10,224,224] → [batch,time,1024]
            feat = base_model(input_var)
            #print('feat.shape={}'.format(feat.shape)) # [batch, time, 1024]
            feat = feat.transpose(0,1)
            #print('feat.shape={}'.format(feat.shape)) # [time, batch, 1024]


            # 2. LSTM  
            v_output= logits_model(feat, v_hidden_states_node_RNNs, v_cell_states_node_RNNs)

            # 3. Loss 
            #print('3.3 CTC Loss')
            """ NoBlankBinaryCTC Loss
                self.CTCLoss(Log_probs, Targets, Input_lengths, Target_lengths)
                1. Log_probs      = [T(input_sequence_length), N(batch_size), C(num_class)]
                2. Targets        = [N(batch_size), S(max_target_length), C(num_class)]
                3. Input_lengths  = [N(batch_size)] # ここは10時刻
                4. Target_lengths = [N(batch_size)] # 実際のラベルの長さ                """
            v_loss = ctc_loss(v_output, v_target_var, input_length, v_target_length)  # バイナリじゃなくて, OneHotにさせます！
            CTC_Loss = v_loss
            Loss = CTC_Loss

            # 4. Accuracy
            #print('3.4 evaluation')  
            # 4.1 予測のAccuracy
            v_f_prec1, v_f_prec5, v_f_prec1_output = accuracy_s(v_output[temporal-1].data, v_f_target, topk=(1, 5)) # OneHotの評価
            # 4.2 ラベルの遷移精度 (O, V単体で測定)
            v_f_top1.update(v_f_prec1[0], input.size(0))
            v_f_top5.update(v_f_prec5[0], input.size(0))

            losses.update(Loss.data, input.size(0))             # 全体のLoss
            ctc_losses.update(CTC_Loss.data, input.size(0))     # CTC Loss
            v_losses.update(v_loss.data, input.size(0))

            # 5. back propagation
            Loss.backward()

            # 6. updating parameters
            if i % args.accum_grad == args.accum_grad-1: # accum_grad=1
                if False:
                    #print('i3d parameters updating')
                    base_optimizer.step()  # I3D Networks, not learning
                    base_optimizer.zero_grad()
                logits_optimizer.step()    # Linear Networks
                logits_optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            # 7. debug print
            if i % args.print_train_freq == 0:
                print('Epoch: [{0}][{1}/{2}({3})]\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'          # 全体のLoss
                      'CTC_Loss {ctc_loss.val:.3f} ({ctc_loss.avg:.3f})\t'         # CTC Loss
                      'V_Loss {v_loss.val:.3f} ({v_loss.avg:.3f})\t'        # V_CTC_Loss
                      'V_F_Prec@1 {v_f_top1.val:.3f} ({v_f_top1.avg:.3f})\t'
                      'V_F_Prec@5 {v_f_top5.val:.3f} ({v_f_top5.avg:.3f})'.format(
                        epoch, i, int(len(loader)*args.train_size), len(loader),
                        loss       = losses,
                        ctc_loss   = ctc_losses,
                        v_loss     = v_losses,
                        v_f_top1   = v_f_top1,
                        v_f_top5   = v_f_top5
                        ))


                writer.writerow([i, losses.val.to('cpu').detach().clone().numpy(),
                                    ctc_losses.val.to('cpu').detach().clone().numpy(),
                                    v_losses.val.to('cpu').detach().clone().numpy(),
                                    v_f_top1.val.to('cpu').detach().clone().numpy(),
                                    v_f_top5.val.to('cpu').detach().clone().numpy()
                                    ])

        return v_f_top1.avg, v_f_top5.avg


    def validate(self, loader, base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, epoch, args, writer):
        with torch.no_grad():
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()    # 全体のLoss
            ctc_losses = AverageMeter()   # CTC Loss
            ce_losses = AverageMeter()    # CE Loss
            v_losses = AverageMeter()
            v_f_top1 = AverageMeter()
            v_f_top5 = AverageMeter()
            v_class = args.v_class
            node_rnn_size = args.node_rnn_size # 1024
            edge_rnn_size = args.edge_rnn_size # 1024
            batch_size = args.batch_size
            temporal = args.temporal
            alpha = args.alpha      # Scale Parametor of CrossEntropy Against CTC

            # switch to evaluate mode
            base_model.eval()   # I3D Networks
            logits_model.eval() # Structural-RNN


            def part(x):
                return itertools.islice(x, int(len(x)*args.val_size)) # default=0.2, but now is 1.0
            end = time.time()

            for i, (input, o_target, v_target, s_target, o_f_target, v_f_target, s_f_target, meta) in enumerate(part(loader)):
                """
                ## image → time=10時刻分
                # input = [batch, temporal, 3, 10, 224, 224]
                ## CTC label -- onehot形式(multi-label) → adjust_time=10時刻分 (最大長の長さ)
                # o_target = [batch, adjust_time, o_class] 
                # v_target = [batch, adjust_time, v_class]
                # s_target = [batch, 1]
                # o_f_target = [batch, o_class] 
                # v_f_target = [batch, v_class]
                # s_f_target = [batch]
                # meta = {id:[batch], time:[batch]} # 実際の長さ {O,V,S}のそれぞれ独立した長さ!!
                """
                #torch.autograd.set_detect_anomaly(True)
                gc.collect()
                data_time.update(time.time() - end)
                meta['epoch'] = epoch

                # define input
                input_var = torch.autograd.Variable(input.cuda()) # 計算グラフ構築(require_grad=True)
                v_target_var = torch.autograd.Variable(v_target.cuda()) # adjust_time = temporal
                v_f_target_var = torch.autograd.Variable(v_f_target.long().cuda()) # adjust_time = temporal                                                       
                s_f_target = torch.tensor(s_f_target).cuda()
                v_f_target = torch.tensor(v_f_target).cuda()
                # define ctc_label
                input_length = torch.full(size=(batch_size,), fill_value=temporal, dtype=torch.long).cuda()
                v_target_length = meta['v_time'].cuda()

                # define Hidden States
                v_hidden_states_node = torch.zeros(batch_size, v_class).cuda()
                v_hidden_states_node_RNNs = torch.autograd.Variable(v_hidden_states_node)
                # define Cell States
                v_cell_states_node = torch.zeros(batch_size, v_class).cuda()
                v_cell_states_node_RNNs = torch.autograd.Variable(v_cell_states_node)


                # 1. I3D Networks
                #print('3.1 base_model') #[batch,time,3,10,224,224] → [batch,time,1024]
                feat = base_model(input_var)
                #print('feat.shape={}'.format(feat.shape)) # [batch, time, 1024]
                feat = feat.transpose(0,1)
                #print('feat.shape={}'.format(feat.shape)) # [time, batch, 1024]

                # 2. Structural-RNN
                #print('3.2 structural_RNN')
                """
                o_output, v_output, s_output = logits_model(feat, 
                            ov_hidden_states_edge_RNNs, vs_hidden_states_edge_RNNs, so_hidden_states_edge_RNNs,
                            #oo_hidden_states_edge_RNNs, vv_hidden_states_edge_RNNs, ss_hidden_states_edge_RNNs,
                            o_hidden_states_node_RNNs, v_hidden_states_node_RNNs,  s_hidden_states_node_RNNs,
                            ov_cell_states_edge_RNNs, vs_cell_states_edge_RNNs, so_cell_states_edge_RNNs,
                            #oo_cell_states_edge_RNNs, vv_cell_states_edge_RNNs, ss_cell_states_edge_RNNs,
                            o_cell_states_node_RNNs, v_cell_states_node_RNNs, s_cell_states_node_RNNs)
                """
                v_output= logits_model(feat, v_hidden_states_node_RNNs, v_cell_states_node_RNNs)

                # 3. Loss 
                #print('3.3 Cross Entoropy Loss')
                """ NoBlankBinaryCTC Loss
                    self.CTCLoss(Log_probs, Targets, Input_lengths, Target_lengths)
                    1. Log_probs      = [T(input_sequence_length), N(batch_size), C(num_class)]
                    2. Targets        = [N(batch_size), S(max_target_length)]
                    3. Input_lengths  = [N(batch_size)] # ここは10時刻
                    4. Target_lengths = [N(batch_size)] # 実際のラベルの長さ
                """
                v_loss = ctc_loss(v_output, v_target_var, input_length, v_target_length)
                #CTC_Loss = o_loss + v_loss + s_loss
                CTC_Loss = v_loss
                Loss = CTC_Loss

                # 4. Accuracy
                #print('3.4 evaluation')  
                # 4.1 予測のAccuracy
                v_f_prec1, v_f_prec5, v_f_prec1_output = accuracy_s(v_output[temporal-1].data, v_f_target, topk=(1, 5))
                # 4.2 ラベル遷移のAccuracy
                v_f_top1.update(v_f_prec1[0], input.size(0))
                v_f_top5.update(v_f_prec5[0], input.size(0))
                losses.update(Loss.data, input.size(0))         # 全体のLoss
                ctc_losses.update(CTC_Loss.data, input.size(0))     # CTC Loss
                ce_losses.update(CE_Loss.data, input.size(0))       # CE Loss
                v_losses.update(v_loss.data, input.size(0))
                v_ce_losses.update(v_ce_loss.data, input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                # 7. debug print
                if i % args.print_test_freq == 0:
                    print('Epoch: [{0}][{1}/{2}({3})]\t'
                        'Loss {loss.val:.3f} ({loss.avg:.3f})\t'              # 全体のLoss
                        'CTC_Loss {ctc_loss.val:.3f} ({ctc_loss.avg:.3f})\t'         # CTC Loss
                        'V_Loss {v_loss.val:.3f} ({v_loss.avg:.3f})\t'
                        'V_F_Prec@1 {v_f_top1.val:.3f} ({v_f_top1.avg:.3f})\t'
                        'V_F_Prec@5 {v_f_top5.val:.3f} ({v_f_top5.avg:.3f})'.format(
                            epoch, i, int(len(loader)*args.train_size), len(loader),
                            loss       = losses,
                            ctc_loss   = ctc_losses,
                            v_loss     = v_losses,
                            v_f_top1   = v_f_top1,
                            v_f_top5   = v_f_top5
                            ))


                    writer.writerow([i, losses.val.to('cpu').detach().clone().numpy(),
                                        ctc_losses.val.to('cpu').detach().clone().numpy(),
                                        v_losses.val.to('cpu').detach().clone().numpy(),
                                        v_f_top1.val.to('cpu').detach().clone().numpy(),
                                        v_f_top5.val.to('cpu').detach().clone().numpy()
                                        ])

            return v_f_top1.avg, v_f_top5.avg