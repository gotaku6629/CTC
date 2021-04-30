#!/usr/bin/env python
import sys
import pdb
import traceback
#sys.path.insert(0, '..')
sys.path.insert(0, '.')
from main import main
#from main_valvideo import main
from bdb import BdbQuit
import subprocess
subprocess.Popen('find ./exp/.. -iname "*.pyc" -delete'.split())

args = [
    '--temporal',  '10', # time of using ST-Graph
    '--gap',  '2', # gap on input image sequences
    '--num-trans', '2', # number of label transitions in input_time_length(=temporal*(gap+1)*STACK)
    '--name', 'Triplet_Single_CTC_predict',
    '--cache-dir', './cr_caches/',
    '--rgb-data', './charades/Charades_v1_rgb/', # videos
    '--rgb-my-data', './charades/Mydata_rgb', # my video
    '--rgb-pretrained-weights', './charades/rgb_i3d_pretrained.pt', # I3D pretrained file
    '--resume', './cr_caches/Triplet_Single_CTC_predict/model.pth.tar', # result write & save file
    '--train-file', './charades/Charades/Charades_v1_train.csv',
    '--val-file', './charades/Charades/Charades_v1_test.csv',
    '--groundtruth-lookup', './utils/groundtruth.p'
]
sys.argv.extend(args)
try:
    main()
except BdbQuit:
    sys.exit(1)
except Exception:
    traceback.print_exc()
    print('')
    pdb.post_mortem()
    sys.exit(1)


