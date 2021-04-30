""" Dataset loader for the My dataset """
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


def parse_charades_csv(filename, s_lab2int):
    labels = {}
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row['id']
            actions = row['actions']
            scene = row['scene']
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

#class Charades(data.Dataset):
class Myvideo(data.Dataset):
    def __init__(self, rgb_root, split, labelpath, cachedir, temporal, gap, rgb_transform=None, target_transform=None):
    # rgb_root       = './gsteg/Charades_v1_rgb/' 
    # split          = 'my'
    # labelpath      = './gsteg/Charades/Charades_v1_train.csv'
    # args.cache_dir = './gsteg/cr_caches/'
    # temporal       = '10'
    # gap       = '0'
        self.s_classes = 16
        self.o_classes = 38
        self.v_classes = 33
        self.rgb_transform = rgb_transform
        self.target_transform = target_transform
        self.rgb_root = rgb_root
        self.testGAP = 25
        self.temporal = temporal        
        self.gap = gap
        self.c2ov = {0: (9, 8),  # (object, verb)
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
                    15: (25, 8), # phone, hold
                    16: (25, 14),
                    17: (25, 16),
                    18: (25, 23),
                    19: (25, 24),
                    20: (1, 8),  # bag, hold 
                    21: (1, 12),
                    22: (1, 16),
                    23: (1, 23),
                    24: (1, 25),
                    25: (4, 1),
                    26: (4, 8),  # book, hold
                    27: (4, 12),
                    28: (4, 16),
                    29: (4, 19),
                    30: (4, 23),
                    31: (4, 25),
                    32: (4, 31),
                    33: (35, 8),  # towl, hold
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
                    47: (20, 8),  # laptop, hold
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
                    97: (14, 29),
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
                    154: (0, 22),
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
        
        print('3.1.1 my label (not use csv file)') # テキトーなラベル
        self.labels = {'YUME0': [{'scene': 11, 'class': 'c097', 'start': 0, 'end': 3.0},
                                  # v='walk', o='doorway'
                                 {'scene': 11, 'class': 'c060', 'start': 3.0, 'end': 4.0},
                                  # v='stand', o='chair'
                                 {'scene': 11, 'class': 'c059', 'start': 4.0, 'end': 8.0},
                                  # v='sit', o='chair'
                                 {'scene': 11, 'class': 'c060', 'start': 8.0, 'end': 12.0},
                                  # v='stand', o='chair'
                                 {'scene': 11, 'class': 'c097', 'start': 12.0, 'end': 15.0}]
                                  # v='walk', o='doorway'
                      }
        #self.labels = parse_charades_csv(labelpath, self.s_lab2int)
        # label = {'46GP8': [{'scene': 9, 'class': 'c092', 'start': 11.9, 'end': 21.2}, ...}

        print('3.1.2 video and label cache')
        cachename = '{}{}_{}.pkl'.format(cachedir, self.__class__.__name__, split)
        self.data = cache(cachename)(self.prepare)(rgb_root, self.labels, split, temporal, gap)


    def prepare(self, rgb_path, labels, split, temporal, gap):
        print('My video prepare_ver3')
        """やること
            1. Paddingによる系列の長さ調整
                1.1 各ビデオのラベルの長さをゲットして, もっとも長い長さに全調整
                1.2 同時に各ビデオにおける長さを記録 → meta情報の'time'に記録する

            2. ラベルは1ラベルに限定させる
                2.1. time_startが同じラベルは, 上書き
                2.2. 新しいラベルのtime_startが来たら, ラベルを上書き
        """
        FPS, GAP,= 29.94, gap   # defalut
        rgb_datadir = rgb_path
        #STACK= 10 + 4 * (temporal -1)  # Need to change
        STACK = 10
        #fidx = future + STACK -1  # future index from top index
        rgb_image_paths, s_targets, o_targets, v_targets, ids, times = [], [], [], [], [], []

        adjust_time = temporal # 今は, 入力画像に対応するラベル(現時刻)の推定を行います!!

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

                # (1つの動画の長さ) > t*STACK*(GAP+1) = 200 でないものはスキップされる!!
                if n < temporal*STACK*(GAP+1):
                    print('skip the video : {}, len = {}'.format(vid, n))
                    continue

                ## 1. time series
                time_series = [] 
                """        
                for x in label:
                    if x['start'] not in time_series: # check the contents                            
                        time_series.append(x['start'])
                    #if x['end'] not in time_series:
                    #    time_series.append(x['end'])
                time_series.sort()
                """
                for t in range(temporal):
                    time_series.append(t*STACK*(GAP+1)/FPS)
                time_length = len(time_series)

                if n == 0: continue           # print('sorry this video is not length') # None
                if time_length == 0: continue # print('sorry this video is not labels') # A lot!!

                ## 2. get image x 10[time]       
                rgb_temporal = []
                for t in range(temporal): # get 10 temporal
                    rgb_impath = '{}/{}-{:06d}.jpg'.format(rgb_iddir, vid, t*(gap+1)*STACK+1)
                    rgb_temporal.append(rgb_impath)                    
                rgb_image_paths.append(rgb_temporal)

                ## 3. get labels {s,v,o,ids,time}
                o_target = torch.IntTensor(temporal).zero_()
                v_target = torch.IntTensor(temporal).zero_()

                for t in range(time_length): # [0.0, 20.0, 40.0, ...]
                    for x in label: # label of 1 video
                        if x['start'] <= time_series[t] and time_series[t] <= x['end']:
                            o, v = cls2int(x['class'], self.c2ov)
                            o_target[t] = o   # 同じstart_timeのラベルは上書きされる
                            v_target[t] = v                  
                            #time_end = x['end']
            
                ## s is single label
                #s_target[0] = label[0]['scene']  # blankラベル分

                if i % 100 == 0:
                    print('temporal={}'.format(time_series))
                    #print('s_target={}'.format(s_target))
                    print('o_target={}'.format(o_target))
                    print('v_target={}'.format(v_target))

                writer.writerow([vid])  # csv出力
                #s_targets.append(s_target)
                o_targets.append(o_target)
                v_targets.append(v_target)
                ids.append(vid)
                times.append(time_length)
                #if time_length > 8:
                #    times.append(8)   # 今, adjustしてるから!!
                #    writer.writerow(['time', 8])  # csv出力
                #else:
                #    times.append(time_length)
                #    writer.writerow(['time', time_length])  # csv出力
                #writer.writerow(['temporal', time_series[0], time_series[1], time_series[2], time_series[3], time_series[4], time_series[5], time_series[6], time_series[7], time_series[8], time_series[9]])
                #writer.writerow(['o_label', o_target[0].to('cpu').detach().clone().numpy(), o_target[1].to('cpu').detach().clone().numpy(), o_target[2].to('cpu').detach().clone().numpy(), o_target[3].to('cpu').detach().clone().numpy(), o_target[4].to('cpu').detach().clone().numpy(), o_target[5].to('cpu').detach().clone().numpy(), o_target[6].to('cpu').detach().clone().numpy(), o_target[7].to('cpu').detach().clone().numpy(), o_target[8].to('cpu').detach().clone().numpy(), o_target[9].to('cpu').detach().clone().numpy()])
                #writer.writerow(['v_label', v_target[0].to('cpu').detach().clone().numpy(), v_target[1].to('cpu').detach().clone().numpy(), v_target[2].to('cpu').detach().clone().numpy(), v_target[3].to('cpu').detach().clone().numpy(), v_target[4].to('cpu').detach().clone().numpy(), v_target[5].to('cpu').detach().clone().numpy(), v_target[6].to('cpu').detach().clone().numpy(), v_target[7].to('cpu').detach().clone().numpy(), v_target[8].to('cpu').detach().clone().numpy(), v_target[9].to('cpu').detach().clone().numpy()])
        
        print('len(rgb_image_paths)={}'.format(len(rgb_image_paths)))
        print('len(o_targets)={}'.format(len(o_targets)))
        
        return {'rgb_image_paths': rgb_image_paths, 'o_targets': o_targets, 'v_targets': v_targets, 'ids': ids, 'times': times}


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
        #s_target = self.data['s_targets'][index]
        o_target = self.data['o_targets'][index]
        v_target = self.data['v_targets'][index]
        meta = {}
        meta['id'] = self.data['ids'][index]
        meta['time'] = self.data['times'][index]
        
        if self.target_transform is not None:
            o_target = self.target_transform(o_target)
            v_target = self.target_transform(v_target)
            
        return rgb_temporal_image, o_target, v_target, meta
        

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
    
    temporal = args.temporal
    gap = args.gap
    print('temporal = {}'.format(temporal))
    print('gap = {}'.format(gap))

    rgb_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    #train_file = args.train_file # csv.file
    #val_file = args.val_file # csv.file
    my_file = args.val_file # make my anotation csv.file, but don't use this.
    

    my_val_dataset = Myvideo(
        args.rgb_my_data, 'my', my_file, args.cache_dir, temporal, gap,  
        rgb_transform=transforms.Compose([ # Tensor & normalization
            transforms.Resize(int(256./224*args.inputsize)), # change image to 256x256
            transforms.CenterCrop(args.inputsize), # crop into 224x224
            transforms.ToTensor(),
            rgb_normalize,
        ]))

    return my_val_dataset