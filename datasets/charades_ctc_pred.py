""" Dataset loader for the Charades dataset """
import torch
import torchvision.transforms as transforms
from datasets import transforms as arraytransforms
import torch.utils.data as data
from PIL import Image
import numpy as np
from glob import glob
import csv
import pickle as pickle
import os
import math


def parse_charades_csv(filename, s_lab2int):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            scene = row['scene']
            #print('vid={}'.format(vid))
            #print('actions={}'.format(actions))
            #print('scene={}'.format(scene))
            if actions == '':
                actions = []
            else:
                actions = [a.split(' ') for a in actions.split(';')]
                actions = [{'scene': s_lab2int[scene], 'class': x, 'start': float(y), 'end': float(z)} 
                for x, y, z in actions]
            labels[vid] = actions
    
    #print('labels') # {'46GP8': [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2}, ...}
    #print(labels)
    return labels


def cls2int(x, c2ov = None):
    o, v = c2ov[int(x[1:])]
    return o, v

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path, 'RGB')
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def cache(cachefile):
    """ Creates a decorator that caches the result to cachefile """
    def cachedecorator(fn):
        def newf(*args, **kwargs):
            print('cachefile {}'.format(cachefile))
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as f:
                    print("Loading cached result from '%s'" % cachefile)
                    return pickle.load(f)
            res = fn(*args, **kwargs)
            with open(cachefile, 'wb') as f:
                print("Saving result to cache '%s'" % cachefile)
                pickle.dump(res, f)
            return res
        return newf
    return cachedecorator


class Charades(data.Dataset):
    def __init__(self, rgb_root, split, labelpath, cachedir, temporal, gap, num_trans, rgb_transform=None, target_transform=None):
    # rgb_root       = './gsteg/Charades_v1_rgb/' 
    # split          = 'train'
    # labelpath      = './gsteg/Charades/Charades_v1_train.csv'
    # cache_dir      = './gsteg/cr_caches/'
    # temporal       = '10'
    # gap            = '0'
        self.s_classes = 16 
        self.o_classes = 38 
        self.v_classes = 33 
        self.rgb_transform = rgb_transform
        self.target_transform = target_transform
        self.rgb_root = rgb_root
        self.testGAP = 10
        self.temporal = temporal
        self.gap = gap
        self.num_trans = num_trans  # ラベル遷移数 
        self.c2ov = {0: (9, 8), # (object, verb)
                    1: (9, 16),
                    2: (9, 23),
                    3: (9, 25),
                    4: (9, 26),
                    5: (9, 30),
                    6: (12, 1),
                    7: (12, 6),
                    8: (12, 12),
                    9: (33, 16),
                    10: (33, 18),
                    11: (33, 18),
                    12: (33, 26),
                    13: (33, 30),
                    14: (33, 32),
                    15: (25, 8),
                    16: (25, 14),
                    17: (25, 16),
                    18: (25, 23),
                    19: (25, 24),
                    20: (1, 8),
                    21: (1, 12),
                    22: (1, 16),
                    23: (1, 23),
                    24: (1, 25),
                    25: (4, 1),
                    26: (4, 8),
                    27: (4, 12),
                    28: (4, 16),
                    29: (4, 19),
                    30: (4, 23),
                    31: (4, 25),
                    32: (4, 31),
                    33: (35, 8),
                    34: (35, 16),
                    35: (35, 23),
                    36: (35, 25),
                    37: (35, 26),
                    38: (35, 30),
                    39: (5, 1),
                    40: (5, 8),
                    41: (5, 12),
                    42: (5, 16),
                    43: (5, 23),
                    44: (5, 23),
                    45: (5, 25),
                    46: (20, 1),
                    47: (20, 8),
                    48: (20, 12),
                    49: (20, 16),
                    50: (20, 23),
                    51: (20, 31),
                    52: (20, 14),
                    53: (31, 8),
                    54: (31, 16),
                    55: (31, 3),
                    56: (31, 23),
                    57: (31, 28),
                    58: (31, 25),
                    59: (7, 18),
                    60: (7, 22),
                    61: (16, 8),
                    62: (16, 16),
                    63: (16, 23),
                    64: (16, 25),
                    65: (29, 5),
                    66: (29, 11),
                    67: (29, 8),
                    68: (29, 16),
                    69: (29, 23),
                    70: (3, 8),
                    71: (3, 16),
                    72: (3, 21),
                    73: (3, 23),
                    74: (3, 25),
                    75: (3, 26),
                    76: (27, 8),
                    77: (27, 16),
                    78: (27, 21),
                    79: (27, 23),
                    80: (27, 25),
                    81: (30, 16),
                    82: (30, 26),
                    83: (26, 23),
                    84: (26, 8),
                    85: (26, 9),
                    86: (26, 16),
                    87: (25, 13),
                    88: (26, 31),
                    89: (37, 1),
                    90: (37, 12),
                    91: (37, 30),
                    92: (37, 31), # (watch, window)
                    93: (23, 8),
                    94: (23, 19),
                    95: (23, 30),
                    96: (23, 31),
                    97: (14, 29), # (doorway, walk)
                    98: (6, 8),
                    99: (6, 16),
                    100: (6, 23),
                    101: (6, 25),
                    102: (6, 26),
                    103: (21, 6),
                    104: (21, 27),
                    105: (21, 27),
                    106: (10, 4),
                    107: (10, 8),
                    108: (10, 15),
                    109: (10, 16),
                    110: (10, 23),
                    111: (10, 30),
                    112: (8, 1),
                    113: (8, 12),
                    114: (8, 26),
                    115: (24, 8),
                    116: (24, 16),
                    117: (24, 23),
                    118: (11, 8),
                    119: (11, 16),
                    120: (11, 23),
                    121: (11, 30),
                    122: (32, 10),
                    123: (32, 18),
                    124: (15, 10),
                    125: (15, 18),
                    126: (15, 25),
                    127: (15, 26),
                    128: (22, 8),
                    129: (22, 5),
                    130: (17, 16),
                    131: (34, 9),
                    132: (34, 31),
                    133: (2, 0),
                    134: (2, 10),
                    135: (2, 18),
                    136: (36, 6),
                    137: (36, 8),
                    138: (36, 23),
                    139: (19, 30),
                    140: (13, 6),
                    141: (13, 7),
                    142: (28, 1),
                    143: (28, 12),
                    144: (18, 6),
                    145: (24, 32),
                    146: (0, 0),
                    147: (16, 2),
                    148: (9, 3),
                    149: (0, 9),
                    150: (0, 17),
                    151: (0, 18),
                    152: (0, 19),
                    153: (0, 20),
                    154: (0, 22), # (none, stand)
                    155: (9, 28),
                    156: (16, 5)}
        
        self.o_name = { 0: 'None',
                        1: 'bag',
                        2: 'bed',
                        3: 'blanket',
                        4: 'book',
                        5: 'box',
                        6: 'broom',
                        7:'chair',
                        8: 'closet/cabinet',
                        9: 'clothes',
                        10: 'cup/glass/bottle',
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
                        24: 'paper/notebook',
                        25: 'phone/camera',
                        26: 'picture',
                        27: 'pillow',
                        28: 'refrigerator',
                        29: 'sandwich',
                        30: 'shelf',
                        31: 'shoe',
                        32: 'sofa/couch',
                        33: 'table',
                        34: 'television',
                        35: 'towel',
                        36: 'vacuum',
                        37: 'window' }

        self.v_name = { 0: 'awaken',
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

        self.s_name = { 0: 'Basement', #  (A room below the ground floor)
                        1: 'Bathroom',
                        2: 'Bedroom',
                        3: 'Closet',
                        4: 'Dining room',
                        5: 'Entryway', # (A hall that is generally located at the entrance of a house)
                        6: 'Garage',
                        7: 'Hallway',
                        8: 'Home Office', #  Study (A room in a house used for work)
                        9: 'Kitchen',
                        10: 'Laundry room',
                        11: 'Living room',
                        12: 'Other',
                        13: 'Pantry',
                        14: 'Recreation room',
                        15: 'Stairs'}

        self.s_lab2int = {'Basement (A room below the ground floor)': 0,
                         'Bathroom': 1,
                         'Bedroom': 2,
                         'Closet / Walk-in closet / Spear closet': 3,
                         'Dining room': 4,
                         'Entryway (A hall that is generally located at the entrance of a house)': 5,
                         'Garage': 6,
                         'Hallway': 7,
                         'Home Office / Study (A room in a house used for work)': 8,
                         'Kitchen': 9,
                         'Laundry room': 10,
                         'Living room': 11,
                         'Other': 12,
                         'Pantry': 13,
                         'Recreation room / Man cave': 14,
                         'Stairs': 15}
        
        print('2.1.1 label from csv')
        self.labels = parse_charades_csv(labelpath, self.s_lab2int)
        # label = {'46GP8': [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2}, ...}
        
        print('2.1.2 video and label cache')
        cachename = '{}{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(rgb_root, self.labels, split, temporal, gap)



    def prepare(self, rgb_path, labels, split, temporal, gap):

        FPS, GAP, testGAP = 24, gap, self.testGAP  # defalut
        rgb_datadir = rgb_path
        #STACK= 10 + 4 * (temporal -1)  # Need to change
        STACK = 10
        #fidx = future + STACK -1  # future index from top index
        adjust_time = temporal # 最終的に入力時刻(予測ラベルも入れたいから!!)と同じ長さまでは大丈夫ということにします!!
        
        rgb_image_paths, o_targets, v_targets, s_targets, ids, o_times, v_times, s_times = [], [], [], [], [], [], [], []


        with open('./cr_caches/label_'+split+'_log.csv', 'w') as f:
            writer = csv.writer(f)
            for i, (vid, label) in enumerate(labels.items()): # num_(kind of video)
                if i % 100 == 0:
                    print('{}/{}'.format(i, len(labels)))
                # vid = ['46GP8', N11GT', ...]
                # label = [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2},
                #          {'scene': 9, 'class': 'c147', 'start':  0.0, 'end': 12.6}],
                #         [{'scene': 15, 'class': 'c098', 'start': 8.6, 'end': 14.2},
                #          {'scene': 15, 'class': 'c075', 'start': 0.0, 'end': 11.7}], ...
                                
                rgb_iddir = rgb_datadir + '/' + vid  # 1 video
                rgb_lines = glob(rgb_iddir+'/*.jpg') # all frame of 1 video
                rgb_n = len(rgb_lines) # 1 video length
                n = int(rgb_n)
                n_time = n / FPS  # この動画の(rgb画像として存在する)実際の最後の時刻

                ## 1. time label series on Whole Videos 
                time_series = []
                for x in label:
                    if x['start'] < n_time and x['start'] not in time_series: # check the contents                         
                        time_series.append(x['start'])
                    if x['end'] < n_time and x['end'] not in time_series:
                        time_series.append(x['end'])
                time_series.sort()
                time_length = len(time_series)
                if time_length < 3: continue  # ラベル遷移が3つ以上ない動画はSkip!
                start_time = time_series[0]           # 動画の始まり
                start_n = math.ceil(start_time * FPS) # 小数点は切り上げ 
                end_time = time_series[time_length-2] # 動画の終わり (予測をするので1つ前)
                end_n = int(end_time * FPS)           # 小数点は切り下げ
                if end_n - start_n < temporal*(GAP+1)*STACK: continue  # 入力時刻より短い動画はSkip!
                if end_n-1-temporal*(GAP+1)*STACK-1 < 10: continue

                ## 2. get image_path & label_series
                if split == 'val_video':  # 2.1 val_video
                    o_target = torch.IntTensor(self.o_classes).zero_()
                    v_target = torch.IntTensor(self.v_classes).zero_()
                    s_target = torch.IntTensor(1).zero_()            
                
                    spacing = np.linspace(start_n, end_n-1-temporal*(GAP+1)*STACK-1, testGAP) # 1つのビデオを10分割
                    
                    for loc in spacing:
                        now = loc/FPS
                        now_end = loc + temporal*(GAP+1)*STACK

                        #  2.1.1. get image path series
                        rgb_temporal = []
                        for t in range(temporal):
                            rgb_impath = '{}/{}-{:06d}.jpg'.format(rgb_iddir, vid, int(np.floor(loc))+1 + t*(GAP+1)*STACK)
                            rgb_temporal.append(rgb_impath)
                        
                        
                        # 2.1.2. get label_series (involved in future_label)
                        time_in_series = []
                        future_time = 0
                        for t in range(time_length):
                            if now <= time_series[t] < now_end:
                                time_in_series.append(time_series[t])
                            if time_series[t] > end_time:
                                future_time = time_series[t]
                                time_in_series.append(future_time)  # CTCだけで予測するので入れておく
                                break
                        time_in_length = len(time_in_series)
                        if future_time == 0: continue
                        if time_in_length -1 < self.
                        :  continue  # 入力ラベル遷移(予測を含まない)が少なかったら
                        if time_in_length > adjust_time:  continue        # CTCの計算的に... 

                        for t in range(time_in_length): # [0.0, 11.9, 12.6, 21.2] # adjust_timeより少ないものも...
                            for x in label: # label of 1 video
                                if x['start'] <= time_in_series[t] <= x['end']:
                                    o, v = cls2int(x['class'], self.c2ov)
                                    o_target[o] = 1
                                    v_target[v] = 1
                        s_target[0] = label[0]['scene']
                        scene = label[0]['scene']  # csvファイルデバック用
                        rgb_image_paths.append(rgb_temporal)
                        o_targets.append(o_target)
                        v_targets.append(v_target)
                        s_targets.append(s_target)
                        ids.append(vid)
                        o_times.append(time_in_length)  # ちょっと適当... アセアセ''
                        v_times.append(time_in_length)
                        s_times.append(1)

                        # 2.1.3 出力デバック
                        writer.writerow([vid])
                        writer.writerow(time_in_series)
                        o_writer = []
                        v_writer = []
                        for o in range(self.o_classes):
                            if o_target[o] == 1:
                                o_writer.append(self.o_name[o])
                        for v in range(self.v_classes):
                            if v_target[v] == 1:
                                v_writer.append(self.v_name[v])
                        writer.writerow(o_writer)
                        writer.writerow(v_writer)
                        writer.writerow(self.s_name[scene])


                else:  # 2.2 train, test(val)
                    for ii in range(start_n, end_n-1-temporal*(GAP+1)*STACK-1, 10):  # なんとなく10のGAPを持たせてみる!
                        now = ii/FPS
                        now_end = ii + temporal*(GAP+1)*STACK

                        # 2.2.1. get image path series
                        rgb_temporal = []
                        for t in range(temporal):
                            rgb_impath = '{}/{}-{:06d}.jpg'.format(rgb_iddir, vid, ii+1+ t*(GAP+1)*STACK)
                            rgb_temporal.append(rgb_impath)

                        ## 2.2.2. get adjust_time labels (using CTC loss)
                        time_in_series = []
                        future_time = 0
                        for t in range(time_length):
                            if now <= time_series[t] < now_end:
                                time_in_series.append(time_series[t])
                            if time_series[t] > end_time:
                                future_time = time_series[t]
                                time_in_series.append(future_time)  # CTCだけで予測するので入れておく
                                break
                        time_in_length = len(time_in_series)
                        if future_time == 0: continue
                        if time_in_length -1 < self.num_trans:  continue  # 入力ラベル遷移(予測を含まない)が少なかったら
                        if time_in_length > adjust_time:  continue        # CTCの計算的に... 

                        o_target = torch.IntTensor(adjust_time, self.o_classes).zero_()
                        v_target = torch.IntTensor(adjust_time, self.v_classes).zero_()
                        o_target_10 = torch.IntTensor(adjust_time).zero_()
                        v_target_10 = torch.IntTensor(adjust_time).zero_()
                        s_target = torch.IntTensor(1).zero_()
                        #print('time_series={}'.format(time_series))
                        for t in range(time_in_length): # [0.0, 11.9, 12.6, 21.2] # adjust_timeより少ないものも...
                            for x in label: # label of 1 video
                                if x['start'] <= time_in_series[t] <= x['end']:
                                    o, v = cls2int(x['class'], self.c2ov)
                                    o_target[t, o] = 1
                                    v_target[t, v] = 1
                            for o in range(self.o_classes):
                                o_target_10[t] += o_target[t, o]*2**o
                            for v in range(self.v_classes):
                                v_target_10[t] += v_target[t, v]*2**v
                        s_target[0] = label[0]['scene']
                        scene = label[0]['scene'] # csvファイルデバック用
                        """ {Object, Verb}で独立した系列を作る """
                        o_only_target = torch.IntTensor(adjust_time, self.o_classes).zero_()
                        v_only_target = torch.IntTensor(adjust_time, self.v_classes).zero_()
                        o_only_target_10 = torch.IntTensor(adjust_time).zero_()
                        v_only_target_10 = torch.IntTensor(adjust_time).zero_()
                        o_target_length = 0
                        v_target_length = 0
                        for t in range(adjust_time):
                            if o_target_10[t] not in o_only_target_10:
                                o_only_target_10[t] = o_target_10[t]
                                o_only_target[o_target_length] = o_target[t]
                                o_target_length += 1
                            if v_target_10[t] not in v_only_target_10:
                                v_only_target_10[t] = v_target_10[t]
                                v_only_target[v_target_length] = v_target[t]
                                v_target_length += 1
                        # add padding # 短いやつは, paddingを加えます!!
                        if o_target_length < adjust_time:  # 現在の動画の時系列が, adjust_timeよりも小さかったら...
                            for pad in range(adjust_time - o_target_length):
                                o_only_target[o_target_length+pad] = -1
                        if v_target_length < adjust_time:  # 現在の動画の時系列が, adjust_timeよりも小さかったら...
                            for pad in range(adjust_time - v_target_length):
                                v_only_target[v_target_length+pad] = -1

                        rgb_image_paths.append(rgb_temporal)
                        o_targets.append(o_only_target)
                        v_targets.append(v_only_target)
                        s_targets.append(s_target)
                        ids.append(vid)
                        o_times.append(o_target_length)
                        v_times.append(v_target_length)
                        s_times.append(1)

                        # 2.2.3 出力デバック
                        writer.writerow([vid])
                        writer.writerow(time_in_series)
                        o_writer = []
                        v_writer = []
                        for t in range(temporal):
                            o_t_writer = []
                            for o in range(self.o_classes):
                                if o_only_target[t, o] == 1:
                                    o_t_writer.append(self.o_name[o])
                            o_writer.append(o_t_writer)
                            v_t_writer = []
                            for v in range(self.v_classes):
                                if v_only_target[t, v] == 1:
                                    v_t_writer.append(self.v_name[v])
                            v_writer.append(v_t_writer)
                        writer.writerow(['Object', o_target_length])    
                        writer.writerow(o_writer)
                        writer.writerow(['Verb', v_target_length])
                        writer.writerow(v_writer)
                        writer.writerow(['Scene', self.s_name[scene]])


        print('len(rgb_image_paths)={}'.format(len(rgb_image_paths)))
        print('len(o_targets)={}'.format(len(o_targets)))
        return {'rgb_image_paths': rgb_image_paths, 'o_targets': o_targets, 'v_targets': v_targets, 's_targets': s_targets, 'ids': ids, 'o_times': o_times, 'v_times': v_times, 's_times': s_times}



    def __getitem__(self, index):
        """
        Args:
            index (int): Index # Indexから10枚の画像を連続してとっていく (default)
        Returns:
            tuple: (image, target) where target is class_index of the target class.

        For Predict
            print('index=',   1392003)
            print('rgb_path=',   Z5THW-000001.jpg)
            print('index+1=', 1392004)
            print('rgb_path+1=', Z5THW-000005.jpg)
        """
        rgb_STACK=10
        rgb_temporal = []
        for t in range(self.temporal):
            #print('index={}, t={}'.format(index, t))
            rgb_path = self.data['rgb_image_paths'][index][t] # 1 flame
            #print(rgb_path) # ./gsteg/Charades_v1_rgb//0GH5O/0GH5O-000297.jpg
            rgb_base = rgb_path[:-5-5]   # ./gsteg/Charades_v1_rgb//0GH5O/0GH5O-
            rgb_framer = int(rgb_path[-5-5:-4])  # 000297
            rgb_img = []

            # 3.1 get 10-frame, but gap!!!
            for i in range(rgb_STACK):
                _img = '{}{:06d}.jpg'.format(rgb_base, rgb_framer+(self.gap+1)*i)
                img = default_loader(_img)
                rgb_img.append(img)
            # now the img is  <PIL.Image.Image image mode=RGB size=240x320 at 0x7FBDC4EF53D0>]

            # 3.2 frame_transform
            if self.rgb_transform is not None:
                _rgb_img = []
                for _per_img in rgb_img:
                    tmp = self.rgb_transform(_per_img)
                    _rgb_img.append(tmp)
                rgb_img = torch.stack(_rgb_img, dim=1)
                # now the img is 25x3x10x224x224
            rgb_temporal.append(rgb_img)
        rgb_temporal_image = torch.cat([rgb_temporal[i].unsqueeze(0) for i in range(len(rgb_temporal))], dim=0)

        # 3.3 select anotation by len(time_series)
        o_target = self.data['o_targets'][index]
        v_target = self.data['v_targets'][index]
        s_target = self.data['s_targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        meta['o_time'] = self.data['o_times'][index]
        meta['v_time'] = self.data['v_times'][index]
        meta['s_time'] = self.data['s_times'][index]

        if self.target_transform is not None:
            o_target = self.target_transform(o_target)
            v_target = self.target_transform(v_target)
            
        return rgb_temporal_image, o_target, v_target, s_target, meta
        
    def __len__(self):
        #print('Charades __len__')
        return len(self.data['rgb_image_paths']) 

    def __repr__(self):
        #print('Charades __repr__')
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    RGB_Root Location: {}\n'.format(self.rgb_root)
        tmp = '    RGB_Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.rgb_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get(args):
    """ Entry point. Call this function to get all Charades dataloaders """
    # rgb_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                       std=[0.229, 0.224, 0.225])
    print('charades_ctc_pred ready ...')
    rgb_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    train_file = args.train_file # csv.file
    val_file = args.val_file # csv.file
    temporal = args.temporal
    gap = args.gap
    num_trans = args.num_trans

    print('temporal = {}'.format(temporal))
    print('gap = {}'.format(gap))
    print(('num_trans = {}'.format(num_trans)))

    train_dataset = Charades(
        args.rgb_data, 'train', train_file, args.cache_dir, temporal, gap, num_trans,
        rgb_transform=transforms.Compose([ # Tensor & normalization
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            #transforms.RandomResizedCrop(args.inputsize),
            #transforms.ColorJitter(
            #    brightness=0.4, contrast=0.4, saturation=0.4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # missing PCA lighting jitter
            rgb_normalize,
        ])
        )
    val_dataset = Charades(
        args.rgb_data, 'val', val_file, args.cache_dir, temporal, gap, num_trans,
        rgb_transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            rgb_normalize,
        ]))
    valvideo_dataset = Charades(
        args.rgb_data, 'val_video', val_file, args.cache_dir, temporal, gap, num_trans,
        rgb_transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            rgb_normalize,
        ])
    )

    return train_dataset, val_dataset, valvideo_dataset # , groundtruth.gt_table
