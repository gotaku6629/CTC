
""" Initilize the datasets module
    New datasets can be added with python scripts under datasets/
"""
import torch
import torch.utils.data
import torch.utils.data.distributed
import importlib


def get_dataset(args):

    print('2.1 dataset.get') #224x224
    dataset = importlib.import_module('.'+args.dataset, package='datasets')
    #print('dataset=', dataset) # datasets/charades_ver2.py
    train_dataset, val_dataset, valvideo_dataset, gt_table = dataset.get(args)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    my_dataset = importlib.import_module('.'+args.my_dataset, package='datasets')
    myvideo_dataset = my_dataset.get(args)
    
    

    print('2.2 DataLoader') # 25x224x224
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, # shuffle=(train_sampler is None)
        num_workers=args.workers, pin_memory=True, drop_last=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    valvideo_loader = torch.utils.data.DataLoader(  # ここも!! しっかりいきたい!!
        valvideo_dataset, batch_size=valvideo_dataset.testGAP, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    myvideo_loader = torch.utils.data.DataLoader(  # だが, 今はとりあえずここ!!
        myvideo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader, valvideo_loader, myvideo_loader, gt_table