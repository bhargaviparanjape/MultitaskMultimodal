import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary,RefExpFeatureDataset
import base_model
from train import train
import utils
import pdb
import sys,os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('data_root', type=str, default=None)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='refex_baseline')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = args.data_root

    dictionary = Dictionary.load_from_file(os.path.join(data_root, 'dictionary.pkl'))
    train_dset = RefExpFeatureDataset('train', dictionary, data_root)
    eval_dset = RefExpFeatureDataset('val', dictionary, data_root)
    batch_size = args.batch_size
    constructor = 'build_%s' % args.model
    #model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model.w_emb.init_embedding(os.path.join(data_root, 'glove6b_init_300d.npy'))

    #model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output)
