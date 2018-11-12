from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import pdb


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(img, image_id, annotation_id, refexp, gold_box):
    entry = {
        'annotation_id' : annotation_id,
        'refexp_id' : refexp['refexp_id'],
        'image_id' : image_id,
        'image' : img,
        'refexp' : refexp['raw'],
        'gold_box' : gold_box}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'val_heldout'
    """
    if name == "val":
        refex_path = os.path.join(dataroot, "google_refexp_val_201511_coco_aligned_and_labeled.json")
    elif name == "val_heldout":
        refex_path = os.path.join(dataroot, "google_refexp_val_201511_coco_aligned_and_labeled_heldout.json")
    else:
        refex_path = os.path.join(
            dataroot, 'google_refexp_%s_201511_coco_aligned_and_labeled_filtered.json' % name)
         # dataroot, 'google_refexp_val_201511_coco_aligned_and_labeled.json')
    data = json.load(open(refex_path))
    refexps = data['refexps']
    images = data['images']
    annotations = data['annotations']
    
    entries = []
    for annotation_id in annotations:
        image_id = annotations[annotation_id]['image_id']
        annotation_id_ = int(annotation_id)
        gold_box = annotations[annotation_id]['labels'].index(1)
        #gold_box = np.random.randint(36)
        if image_id not in img_id2val:
            continue
        img = img_id2val[image_id]
        refexp_ids = annotations[annotation_id]['refexp_ids']
        for id_ in refexp_ids:
            refexp = refexps[str(id_)]
            entries.append(_create_entry(img, image_id, annotation_id_, refexp, gold_box))

    return entries


class RefExpFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(RefExpFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'val_heldout']

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % ('train' if name == 'val_heldout' else name)))
            #open(os.path.join(dataroot, 'train36_imgid2idx_small.pkl')))


        print('loading features from h5 file')
        # Load the feature file provided by Zarana here
        h5_path = os.path.join(dataroot, '%s36.hdf5' % ('train' if name == 'val_heldout' else name))
        # h5_path = os.path.join(dataroot, 'train36_small.hdf5')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            # self.gold_boxes = np.array(hf.get('gold_box'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['refexp'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['r_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            refexp = torch.from_numpy(np.array(entry['r_token']))
            entry['r_token'] = refexp
            entry['gold_box'] = torch.LongTensor(1).fill_(entry['gold_box'])

    def __getitem__(self, index):
        entry = self.entries[index]
        features = self.features[entry['image']]
        spatials = self.spatials[entry['image']]

        refexp = entry['r_token']
        gold_box = entry['gold_box']

        refexp_id = entry['refexp_id']
        image_id = entry['image_id']
        annotation_id = entry['annotation_id']
        return features, spatials, refexp, gold_box, image_id, annotation_id, refexp_id

    def __len__(self):
        return len(self.entries)
