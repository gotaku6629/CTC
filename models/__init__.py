"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
#from models.layers.AsyncTFCriterion import AsyncTFCriterion
#from models.layers.AsyncTFBase import AsyncTFBase
#from deepSRNN import SRNN
#from SRNN_ver3 import SRNN  # TemporalEdgeあり
#from SRNN_ver2 import SRNN   # TemporalEdgeなし
from LSTM import LSTM_cell # Verbだけ
from CrossEntropy import CrossEntropy
from NoBlankCTC import NoBlankCTC
from NoBlankBinaryCTC import NoBlankBinaryCTC
#from NoBlankBinaryCTC_ver2 import NoBlankBinaryCTC

def sub_create_model(args):    
    """ The I3D network is pre-trained from Kinetics dataset """
    pretrained_weights = args.rgb_pretrained_weights
    from models.i3d import InceptionI3d  # i3d.py

    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(157)
    
    if not pretrained_weights == '':
        print('loading pretrained-weights from {}'.format(pretrained_weights))
        model.load_state_dict(torch.load(pretrained_weights))

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #print('I3D parameters')
    #for param in model.parameters():
    #    print(param.shape)
    #print(model.state_dict().keys())

    return model, optimizer


def create_model(args):

    print('1.1 create sRNN model')
    model = LSTM_cell(args).cuda()
    #model = AsyncTFBase(args.extract_feat_dim, args.s_class, args.o_class, args.v_class, args.temporal, args.batch_size).cuda() 
    #print('AsyncTFBase parameters')
    #for param in model.parameters():
    #    print(param.shape)
    #print(model.state_dict().keys())

    print('1.2 define optimizer')
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                #momentum=args.momentum,
                                weight_decay=args.weight_decay) # using srnn optimizer
    #optimizer = torch.optim.SGD(model.parameters(), args.lr,      # gsteg optimizer
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    
    print('1.3 create i3d model')
    rgb_model, rgb_optimizer = sub_create_model(args)
    
    print('1.4 define loss function')
    #criterion = AsyncTFCriterion(args).cuda()
    #ctc_loss = nn.CTCLoss()
    ctc_loss = NoBlankCTC().cuda()    
    bctc_loss = NoBlankBinaryCTC().cuda()
    #ce_criterion = nn.CrossEntropyLoss()
    ce_loss = nn.CrossEntropyLoss().cuda()
    bce_loss = nn.BCEWithLogitsLoss().cuda()
    
    cudnn.benchmark = True
    
    return rgb_model, model, ctc_loss, bctc_loss, ce_loss, bce_loss, rgb_optimizer, optimizer
