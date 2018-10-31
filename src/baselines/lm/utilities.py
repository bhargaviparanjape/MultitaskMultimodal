import codecs
import argparse
import torch
import numpy as np
import re

try:
    import cPickle as pickle
except:
    import pickle

import math
import os
import json
from torch import optim
import time
from collections import defaultdict
import operator

use_cuda = torch.cuda.is_available()
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

def pad_seq(seq, seq_length, pad_token):
    diff = max(seq_length - len(seq), 0)
    for i in diff:
        seq += pad_token
    return seq


# Build the vocabulary.
def file_split(f, delim=' \t\n', bufsize=1024):
    prev = ''
    while True:
        s = f.read(bufsize)
        if not s:
            break
        tokens = re.split('['+delim+']{1,}', s)
        if len(tokens) > 1:
            yield prev + tokens[0]
            prev = tokens[-1]
            for x in tokens[1:-1]:
                yield x
        else:
            prev += s
    if prev:
        yield prev