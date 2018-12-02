import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary,RefExpFeatureDataset
import base_model
from train import train, evaluate
import utils
import pdb
import sys,os
import json

import scipy
import tempfile

sys.path.append('/home/ubuntu/11777/src/baselines/refexp/Google_Refexp_toolbox/google_refexp_py_lib')
from refexp_eval import RefexpEvalComprehension
from refexp_eval import RefexpEvalGeneration
from common_utils import draw_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('data_root', type=str, default=None)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='refex_baseline')
    parser.add_argument('--output', type=str, default='saved_models/exp3')
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--analysis_file', type=str, default="saved_models/analysis_exp1.json")
    parser.add_argument('--model_file', type=str, default="saved_models/exp3/model.pth")
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args

def load_evaluation_refexp():
    base_dir = '/home/ubuntu/11777/src/baselines/refexp/Google_Refexp_toolbox'
    coco_data_path = '%s/external/coco/annotations/instances_train2014.json' % base_dir
    refexp_dataset_path = '%s/google_refexp_dataset_release/google_refexp_val_201511_coco_aligned.json' % base_dir
    eval_compreh = RefexpEvalComprehension(refexp_dataset_path, coco_data_path)
    data_path = '/home/ubuntu/11777/data/refexp/'

    with open('%s/google_refexp_val_201511_coco_aligned_and_labeled.json' % data_path, 'r') as fp:
        labeled_data = json.load(fp)

    return eval_compreh, labeled_data

if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data_root = args.data_root

    dictionary = Dictionary.load_from_file(os.path.join(data_root, 'dictionary.pkl'))
    train_dset = RefExpFeatureDataset('train', dictionary, data_root)
    eval_dset = RefExpFeatureDataset('val_heldout' if args.mode == "eval_heldout" else 'val', dictionary, data_root)
    batch_size = args.batch_size
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid)
    if torch.cuda.is_available():
        model = model.cuda()
    #model = getattr(base_model, constructor)(train_dset, args.num_hid)
    model.w_emb.init_embedding(os.path.join(data_root, 'glove6b_init_300d.npy'))

    if torch.cuda.is_available():
        model = nn.DataParallel(model).cuda()
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    if args.mode == "train":
        eval_compreh, labeled_data = load_evaluation_refexp()
        train(model, train_loader, eval_loader, args.epochs, args.output, eval_compreh, labeled_data)
    else:
        checkpoint = torch.load(args.model_file)
        model.load_state_dict(checkpoint)
        score, analysis_log = evaluate(model, eval_loader)
        with open(args.analysis_file, "w+") as fout:
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
