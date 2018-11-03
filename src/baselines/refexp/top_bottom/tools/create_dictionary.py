from __future__ import print_function
import os
import sys
import json
import pdb
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    # Aligned Data from MSCOCO and Google Referential Expressions
    files = [
        'google_refexp_train_201511_coco_aligned.json',
        'google_refexp_val_201511_coco_aligned.json',
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['refexps']
        for exp in qs:
            dictionary.tokenize(qs[exp]['raw'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary(sys.argv[1])
    d.dump_to_file(os.path.join(sys.argv[1], "dictionary.pkl"))

    d = Dictionary.load_from_file(os.path.join(sys.argv[1], 'dictionary.pkl'))
    emb_dim = 300
    glove_file = os.path.join(sys.argv[1], 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(sys.argv[1], 'glove6b_init_%dd.npy' % emb_dim), weights)
