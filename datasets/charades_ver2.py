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
        self.s_classes = 16 # 
        self.o_classes = 38 # 
        self.v_classes = 33 # 
        self.rgb_transform = rgb_transform
        self.target_transform = target_transform
        self.rgb_root = rgb_root
        self.testGAP = 25 # not used
        self.temporal = temporal
        self.gap = gap
        self.num_trans = num_trans  # ?????????????????? 
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
        
        if split == 'groundtruth':
            print('2.1.3 make ground truth lookup table')
            self.gt_table = self.prepare_groundtruth(self.labels, split, temporal, gap)

        elif split == 'future_groundtruth':
            self.gt_future_table = self.prepare_future_groundtruth(self.labels, split, temporal, gap)
        else:
            print('2.1.2 video and label cache')
            cachename = '{}{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
            self.data = cache(cachename)(self.prepare)(rgb_root, self.labels, split, temporal, gap)


    def prepare_future_groundtruth(self, labels, split, temporal, gap):
        FPS, GAP, testGAP = 24, gap, self.testGAP  # defalut
        STACK = 10
        gt_table = {} # ground_truth_lookup_tabel

        with open(split+'_log.csv', 'w') as f:
            writer = csv.writer(f)
            for i, (vid, label) in enumerate(labels.items()):
                #if i % 100 == 0:
                #    print('{}/{}'.format(i, len(labels)))
                # vid = ['46GP8', N11GT', ...]
                # label = [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2},
                #          {'scene': 9, 'class': 'c147', 'start':  0.0, 'end': 12.6}],
                #         [{'scene': 15, 'class': 'c098', 'start': 8.6, 'end': 14.2},
                #          {'scene': 15, 'class': 'c075', 'start': 0.0, 'end': 11.7}], ...

                # 1. time series
                time_series = [] # ??????????????????????????????
                for x in label:
                    if x['start'] not in time_series: # check the contents                            
                        time_series.append(x['start'])
                    if x['end'] not in time_series:
                        time_series.append(x['end'])
                time_series.sort()
                time_length = len(time_series)
                if time_length == 0: continue # print('sorry this video is not labels') # A lot!!
                #print('len={}/series={}'.format(time_length, time_series))

                # 2. adjust time series  # temporal*STACK*(GAP+1)???????????????????????????!! 
                start_time = time_series[0]
                adjust_series = []
                future_time = 0
                for t in range(time_length):
                    if time_series[t] > start_time+temporal*STACK*(GAP+1)/FPS: 
                        future_time = time_series[t]
                        break
                    adjust_series.append(time_series[t])
                adjust_length = len(adjust_series)
                if future_time == 0:    continue
                #print('adj={}/adjust={}'.format(adjust_length, adjust_series))

                # 3. get label
                gt_label = []
                s = label[0]['scene']
                for t in range(adjust_length):
                    for x in label:
                        if x['start'] == future_time:
                            o, v = cls2int(x['class'], self.c2ov)
                            gt_label.append([s,o,v])
                gt_table[vid] = gt_label
                #print('gt_tabel={}'.format(gt_table[vid]))  
                writer.writerow(vid)
                writer.writerow(gt_table[vid])
        return gt_table      



    def prepare_groundtruth(self, labels, split, temporal, gap):
        FPS, GAP, testGAP = 24, gap, self.testGAP  # defalut
        STACK = 10
        adjust_time = temporal
        gt_table = {} # ground_truth_lookup_tabel

        with open(split+'_log.csv', 'w') as f:
            writer = csv.writer(f)
            for i, (vid, label) in enumerate(labels.items()):
                #if i % 100 == 0:
                #    print('{}/{}'.format(i, len(labels)))
                # vid = ['46GP8', N11GT', ...]
                # label = [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2},
                #          {'scene': 9, 'class': 'c147', 'start':  0.0, 'end': 12.6}],
                #         [{'scene': 15, 'class': 'c098', 'start': 8.6, 'end': 14.2},
                #          {'scene': 15, 'class': 'c075', 'start': 0.0, 'end': 11.7}], ...

                ## 1. time series 
                time_series = []
                for x in label:
                    if x['start'] not in time_series: # check the contents                         
                        time_series.append(x['start'])
                    if x['end'] not in time_series:
                        time_series.append(x['end'])
                time_series.sort()
                time_length = len(time_series)
                if time_length == 0: continue
                start_time = time_series[0]
                end_time = start_time + temporal*(GAP+1)*STACK/FPS

                # 2.  ????????????????????????????????????1????????????????????????????????????
                # ??? end_time?????????time_series?????????????????????????????????????????????!!
                future_time = 0
                time_in_series = []
                for t in range(time_length):
                    if time_series[t] <= end_time:
                        time_in_series.append(time_series[t])
                    if time_series[t] > end_time:
                        future_time = time_series[t]
                        time_in_series.append(future_time)
                        break
                time_in_length = len(time_in_series)  # ????????????????????????????????? + ???????????????
                if future_time == 0:  continue
                if time_in_length -1 < self.num_trans: continue  # ??????????????????????????????????????????????????????...
                if time_in_length > adjust_time: continue  # ???????????????10???????????????CTC??????????????????????????????...

                # 3. get label
                gt_label = []
                s = label[0]['scene']
                for t in range(time_in_length):
                    for x in label:
                        if x['start'] <= time_in_series[t] and time_in_series[t] <= x['end']:
                            o, v = cls2int(x['class'], self.c2ov)
                            if [s,o,v] not in gt_label:
                                gt_label.append([s,o,v])
                gt_table[vid] = gt_label
                #print('gt_tabel={}'.format(gt_table[vid]))  
                writer.writerow(vid)
                writer.writerow(gt_table[vid])
        return gt_table      



    def prepare(self, rgb_path, labels, split, temporal, gap):
        print('Charades prepare_ver2')
        """????????????
            1. Padding??????????????????????????????
                1.1 ???????????????????????????????????????????????????, ????????????????????????????????????
                1.2 ???????????????????????????????????????????????? ??? meta?????????'time'???????????????

            2. ????????????1???????????????????????????
                2.1. time_start?????????????????????, ?????????
                2.2. ?????????????????????time_start????????????, ?????????????????????
        """
        FPS, GAP, testGAP = 24, gap, self.testGAP  # defalut
        rgb_datadir = rgb_path
        #STACK= 10 + 4 * (temporal -1)  # Need to change
        STACK = 10
        #fidx = future + STACK -1  # future index from top index
        adjust_time = temporal # ????????????????????????(????????????????????????????????????!!)????????????????????????????????????????????????????????????!!
        
        rgb_image_paths, o_targets, v_targets, s_targets, ids, times = [], [], [], [], [], []


        with open('label_'+split+'_log.csv', 'w') as f:
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

                ## 1. time series 
                time_series = []
                for x in label:
                    if x['start'] not in time_series: # check the contents                         
                        time_series.append(x['start'])
                    if x['end'] not in time_series:
                        time_series.append(x['end'])
                time_series.sort()
                time_length = len(time_series)
                if time_length == 0: continue
                start_time = time_series[0]
                end_time = start_time + temporal*(GAP+1)*STACK/FPS

                # 1.1 (1?????????????????????) > (STACK + GAP) x temporal = 100 ???????????????????????????????????????!!
                if n < start_time*FPS+temporal*(GAP+1)*STACK+1:  continue  #print('skip the video : {}, len = {}'.format(vid, n))

                # 1.2 ????????????????????????????????????1????????????????????????????????????
                # ??? end_time?????????time_series?????????????????????????????????????????????!!
                future_time = 0
                time_in_series = []
                for t in range(time_length):
                    if time_series[t] <= end_time:
                        time_in_series.append(time_series[t])
                    if time_series[t] > end_time:
                        future_time = time_series[t]
                        time_in_series.append(future_time)
                        break
                time_in_length = len(time_in_series)  # ????????????????????????????????? + ???????????????
                if future_time == 0:  continue
                if time_in_length -1 < self.num_trans: continue  # ??????????????????????????????????????????????????????...
                if time_in_length > adjust_time: continue  # ???????????????10???????????????CTC??????????????????????????????...

                # 1.3 csv??????debug
                writer.writerow(vid)
                writer.writerow(time_in_series)

                ## 2. get image x 10[time]
                rgb_temporal = []
                for t in range(temporal): # get 10 temporal
                    rgb_impath = '{}/{}-{:06d}.jpg'.format(rgb_iddir, vid, math.floor(start_time*FPS) + t*(GAP+1)*STACK +1)
                    rgb_temporal.append(rgb_impath)                    
                rgb_image_paths.append(rgb_temporal)


                ## 3. get adjust_time labels (using CTC loss)
                o_target = torch.IntTensor(adjust_time, self.o_classes).zero_()
                v_target = torch.IntTensor(adjust_time, self.v_classes).zero_()
                s_target = torch.IntTensor(adjust_time).zero_()
                #print('time_series={}'.format(time_series))
                for t in range(time_in_length): # [0.0, 11.9, 12.6, 21.2] # adjust_time????????????????????????...
                    for x in label: # label of 1 video
                        if x['start'] <= time_in_series[t] <= x['end']:
                            o, v = cls2int(x['class'], self.c2ov)
                            o_target[t, o] = 1
                            v_target[t, v] = 1
                        s_target[t] = label[0]['scene']
                # add padding # ???????????????, padding???????????????!!
                if time_in_length < adjust_time:  # ??????????????????????????????, adjust_time???????????????????????????...
                    for pad in range(adjust_time - time_in_length):
                        o_target[time_in_length+pad] = -1
                        v_target[time_in_length+pad] = -1
                # csv??????debug
                writer.writerow(o_target)
                writer.writerow(v_target)
                writer.writerow(s_target)

                if i % 1000 == 0:
                    print('start_time={}'.format(start_time))
                    print('end_time={}'.format(end_time))
                    print('future_time={}'.format(future_time))
                    print('time_series={}'.format(time_series))
                    print('time_in_series={}'.format(time_in_series))
                    print('o_target={}'.format(o_target))
                    print('v_target={}'.format(v_target))
                    print('s_target={}'.format(s_target))

                o_targets.append(o_target)
                v_targets.append(v_target)
                s_targets.append(s_target)
                ids.append(vid)
                times.append(time_in_length)

        print('len(rgb_image_paths)={}'.format(len(rgb_image_paths)))
        print('len(o_targets)={}'.format(len(o_targets)))
        return {'rgb_image_paths': rgb_image_paths, 'o_targets': o_targets, 'v_targets': v_targets, 's_targets': s_targets, 'ids': ids, 'times': times}


    def __getitem__(self, index):
        """
        Args:
            index (int): Index # Index??????10?????????????????????????????????????????? (default)
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
        meta['time'] = self.data['times'][index]

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
    
    rgb_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    train_file = args.train_file # csv.file
    val_file = args.val_file # csv.file
    temporal = args.temporal
    gap = args.gap
    num_trans = args.num_trans

    print('temporal = {}'.format(temporal))
    print('gap = {}'.format(gap))
    print(('num_trans in input_time_length ={}'.format(num_trans)))

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
    groundtruth = Charades(
        args.rgb_data, 'groundtruth', val_file, args.cache_dir, temporal, gap, num_trans,
        rgb_transform=transforms.Compose([
            transforms.Resize(int(256./224*args.inputsize)),
            transforms.CenterCrop(args.inputsize),
            transforms.ToTensor(),
            rgb_normalize,
        ]))

    return train_dataset, val_dataset, valvideo_dataset, groundtruth.gt_table
