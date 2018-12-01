import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, FeatureDataset
import base_model
from train import train, evaluate
from train_multiplex import multitask_train
import utils
import pdb
import sys,os
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='multitask')
    parser.add_argument('--output', type=str, default='saved_models/exp2')
    parser.add_argument('--mode', type=str, default="train", help='train, eval_heldout, vqa2ref, ref2vqa')
    parser.add_argument('--analysis_file', type=str, default="saved_models/analysis_exp1.json")
    parser.add_argument('--model_file', type=str, default="saved_models/model_multi_best.pth")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--task', choices=["vqa", "ref", "ref_vqa"], type=str)
    parser.add_argument('--mt_mode', choices=['multiplex', 'sequential'], default='sequential')
    parser.add_argument("--dictionary", type=str)
    parser.add_argument("--run_as", type=str, default='normal', choices=['normal', 'debug'])
    parser.add_argument('--learning_rate', type=int, default=2e-3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = args.data_root
    batch_size = args.batch_size
    constructor = 'build_%s' % args.model

    dictionary = Dictionary.load_from_file(args.dictionary)

    if args.task == "ref_vqa":
        train_dset_vqa = FeatureDataset("vqa",'train', dictionary,data_root, args.run_as)
        eval_dset_vqa = FeatureDataset("vqa",'val', dictionary,data_root, args.run_as)
        train_dset_ref = FeatureDataset("ref",'train', dictionary, data_root, args.run_as)
        #eval_dset_ref = FeatureDataset("ref",'val_heldout' if args.mode == "eval_heldout" else 'val', dictionary, data_root)
	eval_dset_ref = FeatureDataset("ref", 'train', dictionary, data_root, args.run_as)
        model = getattr(base_model, constructor)(args.task, train_dset_vqa, args.num_hid)

    elif args.task == "vqa":
        train_dset = FeatureDataset(args.task,'train', dictionary,data_root, args.run_as)
        eval_dset =  FeatureDataset(args.task,'val', dictionary,data_root, args.run_as)
        model = getattr(base_model, constructor)(args.task, train_dset, args.num_hid)

    elif args.task == "ref":
        train_dset =  FeatureDataset(args.task,'train', dictionary, data_root, args.run_as)
        eval_dset = FeatureDataset(args.task,'val_heldout' if args.mode == "eval_heldout" else 'val', dictionary, data_root, args.run_as)
	#eval_dset = FeatureDataset(args.task,'train', dictionary, data_root)
        model = getattr(base_model, constructor)(args.task, train_dset, args.num_hid)

    else:
        print("ERROR: Give valid combination!")


    if torch.cuda.is_available():
        print('running model on CUDA')
        model = model.cuda()
    #model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model.w_emb.init_embedding(os.path.join(data_root, 'glove_6b_common_300d.npy'))
    
    if args.run_as == 'normal' and torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()

    # task-specific data loaders
    train_loaders = {'vqa': None, 'ref': None}
    eval_loaders = {'vqa': None, 'ref': None}

    if args.task == 'vqa':
        train_loaders['vqa'] = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
        eval_loaders['vqa'] =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    elif args.task == 'ref':
	train_loaders['ref'] = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4)
	eval_loaders['ref'] =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=4)

    else:
        train_loaders['vqa'] = DataLoader(train_dset_vqa, batch_size, shuffle=True, num_workers=4)
        eval_loaders['vqa'] = DataLoader(eval_dset_vqa, batch_size, shuffle=False, num_workers=4)
        train_loaders['ref'] = DataLoader(train_dset_ref, batch_size, shuffle=True, num_workers=4)
        eval_loaders['ref'] = DataLoader(eval_dset_ref, batch_size, shuffle=False, num_workers=4)

    if args.mode == "train":
        if args.mt_mode == 'sequential':
            train(args.task, model, train_loaders, eval_loaders, args.epochs, args.output)
        else:
            multitask_train(args.task, model, train_loaders, eval_loaders, args.epochs, args.output, args.run_as)

    elif args.mode == 'vqa2ref' or 'ref2vqa':
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint)
        train(args.task, model, train_loaders, eval_loaders, args.epochs, args.output)

    else:
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint)
        if args.task == 'vqa' or args.task == 'ref_vqa':
            score, analysis_log = evaluate(model, eval_loaders['vqa'], 'vqa')

            # obtain answers dictionary
            with open(args.output+'/analysis_vqa.json', "w+") as fout:
                for item in analysis_log:
                    answer = item[-1]
                    # q_tokens = [dictionary.idx2word[id] for id in item[-2]]
                    dict_ = {
                        "question_id": item[1],
                        "answer_id": answer,
                        "answer_tokens": eval_dset.label2ans[answer],
                        "image_id": item[0]
                    }
                    fout.write(json.dumps(dict_) + "\n")

        elif args.task == 'ref' or args.task == 'ref_vqa':
            score, analysis_log = evaluate(model, eval_loaders['ref'], 'ref')
            with open(args.output+'/analysis_ref.json', "w+") as fout:
                for item in analysis_log:
                    # q_tokens = [dictionary.idx2word[id] for id in item[-1]]
                    dict_ = {
                        "image_id" : item[0],
                        "annotation_id" : item[1],
                        "refexp_id" : item[2],
                        "predicted_id" : item[3],
                        "logits": list(map(float, item[5])),
                    }
                    fout.write(json.dumps(dict_) + "\n")
