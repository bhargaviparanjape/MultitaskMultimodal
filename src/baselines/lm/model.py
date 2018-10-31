import torch.nn as nn
import torch
from torch import autograd

PAD_token = 0
SOS_token = 1
EOS_token = 2


class Model(nn.Module):
    def __init__(self, args, data_loader):
        super(Model, self).__init__()
        self.args = args
        self.data_loader = data_loader
        self.batch_size = args.batch_size
        self.bptt = args.bptt
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.word_vocab_size = len(self.data_loader.vocabulary.word2id)
        self.word_dim = self.args.word_dim
        self.is_cuda = torch.cuda.is_available()
        self.save_to = args.model_path
        self.load_from = self.save_to

    def save(self):
        if self.save_to is not None:
            torch.save(self, self.save_to)
            print("Saved model: " + self.save_to)
        else:
            print('Save to path not provided!')

    def pad_seq(self, seq, max_len, pad_token = 0):
        seq += [pad_token for i in range(max_len-len(seq))]
        # mask = [1]*len(seq) + [0]*(max_len-len(seq))
        return seq


    def load(self, path=None):
        if path is None:
            path = self.load_from
        if self.load_from is not None or path is not None:
            print('Load model parameters from %s!' % path)
            self.model.populate(path)
        else:
            print('Load from path not provided!')


