#!/usr/bin/env python

"""Charades activity recognition baseline code
   Can be run directly or throught config scripts under exp/
   Gunnar Sigurdsson, 2018

    Connectionist Temporal Classification Loss
""" 
import torch
import numpy as np
import random
import train
from models import create_model
from datasets import get_dataset
import checkpoints
from opts import parse
from utils import tee
import csv


def seed(manualseed):
    random.seed(manualseed)
    np.random.seed(manualseed)
    torch.manual_seed(manualseed)
    torch.cuda.manual_seed(manualseed)


best_mAP = 0
def main():
    global opt, best_mAP
    opt = parse()
    tee.Tee(opt.cache+'/log_model_suggest.txt')
    #print(vars(opt))
    seed(opt.manual_seed)

    print('1. create_model')
    base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, base_optimizer, logits_optimizer = create_model(opt)
    if opt.resume:
        print('checkpoints load')
        best_mAP = checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
        #checkpoints.load(opt, base_model, logits_model, base_optimizer, logits_optimizer)
    print('Scale Parameter Of CrossEntropy Against CTC = ', opt.alpha)
    trainer = train.Trainer()

    print('2. get_dataset')
    train_loader, val_loader, valvideo_loader, myvideo_loader, gt_table = get_dataset(opt)

    if opt.evaluate:                                             
        trainer.validate(val_loader, base_model, logits_model, -1, opt)
        trainer.validate_video(valvideo_loader, base_model, logits_model, -1, opt)
        return


    #print('6. Test (My video)')
    #epoch = 20
    #trainer.validate_myvideo(myvideo_loader, base_model, logits_model, epoch, opt)
    #print('Finish')
    with open('./cr_caches/train_log.csv', 'w') as csvfile:
        train_writer = csv.writer(csvfile)
        train_writer.writerow(['i', 'Loss', 'V_F_Prec@1', 'V_F_Prec@5','V_Prec@1', 'V_Prec@5', 'V_Recall@1', 'V_Recall@5'])
        with open('./cr_caches/test_log.csv', 'w') as f:
            test_writer = csv.writer(f)
            test_writer.writerow(['i', 'Loss', 'V_F_Prec@1', 'V_F_Prec@5','V_Prec@1', 'V_Prec@5', 'V_Recall@1', 'V_Recall@5'])
            with open('./cr_caches/score.csv', 'w') as score_file:
                score_writer = csv.writer(score_file)
                score_writer.writerow(['v_f_top1', 'v_f_top5', 'v_top1', 'v_top5', 'v_rec_top1', 'v_rec_top5', 'v_f_top1val', 'v_f_top5val', 'v_top1val', 'v_top5val', 'v_rec_top1val', 'v_rec_top5val'])
                for epoch in range(opt.start_epoch, opt.epochs): # 0~20
                    if opt.distributed:
                        trainer.train_sampler.set_epoch(epoch)
                    
                    #print('3 Train')
                    #o_f_top1, o_f_top5, v_f_top1, v_f_top5, s_f_top1, s_f_top5, sov_f_top1, o_top1, o_top5, v_top1, v_top5, s_top1, s_top5, sov_top1, o_rec_top1, o_rec_top5, v_rec_top1, v_rec_top5 = trainer.train(train_loader, base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, base_optimizer, logits_optimizer, epoch, opt, train_writer)
                    v_f_top1, v_f_top5 = trainer.train(train_loader, base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, base_optimizer, logits_optimizer, epoch, opt, train_writer)
                    
                    #print('4. Test (Validation)')
                    v_f_top1val, v_f_top5val = trainer.validate(val_loader, base_model, logits_model, ctc_loss, bctc_loss, ce_loss, bce_loss, epoch, opt, test_writer)

                    score_writer.writerow([v_f_top1.to('cpu').detach().clone().numpy(), v_f_top5.to('cpu').detach().clone().numpy(), v_f_top1val.to('cpu').detach().clone().numpy(), v_f_top5val.to('cpu').detach().clone().numpy()])


if __name__ == '__main__':
    main()
