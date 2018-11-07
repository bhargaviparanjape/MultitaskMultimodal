import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset
import base_model
from train import train, evaluate
import json
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--analysis_file', type=str, default="saved_models/analysis.json")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    train(model, train_loader, eval_loader, args.epochs, args.output)
    if args.mode == "train":
        train(model, train_loader, eval_loader, args.epochs, args.output)
    else:
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint)
        score, analysis_log = evaluate(model, eval_loader)

        # obtain answers dictionary
        with open(args.analysis_file, "w+") as fout:
            for item in analysis_log:
                answer = item[-1]
                # q_tokens = [dictionary.idx2word[id] for id in item[-2]]
                dict_ = {
                    "question_id" : item[0],
                    "answer_id" : answer
                }
                fout.write(json.dumps(dict_) + "\n")