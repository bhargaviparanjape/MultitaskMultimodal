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


def _create_entry(img=None, question=None, answer=None,annotation_id=None, refexp=None, gold_box=None, image_id=None):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question.get('question_id',None),
        'image_id'    : question.get('image_id',None),
        'image'       : img,
        'question'    : question.get('question',None),
        'answer'      : answer,
        'annotation_id': annotation_id,
        'refexp_id': refexp.get('refexp_id',None),
        'refexp': refexp.get('raw',None),
        'gold_box': gold_box
    }
    return entry


def _load_dataset(task, dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == "val":
        question_path = os.path.join(dataroot, "v2_OpenEnded_mscoco_val2014_questions.json")
        refex_path = os.path.join(dataroot, "google_refexp_val_201511_coco_aligned_and_labeled.json")
    elif name == "val_heldout":
        refex_path = os.path.join(dataroot, "google_refexp_val_201511_coco_aligned_and_labeled_heldout.json")
    else:
        question_path = os.path.join(
            dataroot, 'v2_OpenEnded_mscoco_%s2014_questions_filtered.json' % name)
        refex_path = os.path.join(
            dataroot, 'google_refexp_%s_201511_coco_aligned_and_labeled_filtered.json' % name)

    #VQA
    vqa_entries = []
    if task == "vqa" or task == "ref_vqa":
        questions = sorted(json.load(open(question_path))['questions'],
                           key=lambda x: x['question_id'])
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])
        utils.assert_eq(len(questions), len(answers))

        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            vqa_entries.append(_create_entry(img_id2val[img_id], question, answer))

    #REF
    ref_entries = []
    if task == "ref" or task == "ref_vqa":
        data = json.load(open(refex_path))
        refexps = data['refexps']
        annotations = data['annotations']

        for annotation_id in annotations:
            image_id = annotations[annotation_id]['image_id']
            annotation_id_ = int(annotation_id)
            gold_box = annotations[annotation_id]['labels'].index(1)
            # gold_box = np.random.randint(36)
            if image_id not in img_id2val:
                continue
            img = img_id2val[image_id]
            refexp_ids = annotations[annotation_id]['refexp_ids']
            for id_ in refexp_ids:
                refexp = refexps[str(id_)]
                ref_entries.append(
                    _create_entry(img=img, image_id=image_id, annotation_id=annotation_id_, refexp=refexp,
                                  gold_box=gold_box))

    return vqa_entries, ref_entries

class FeatureDataset(Dataset):
    def __init__(self,task, name, dictionary, dataroot='data',):
        super(FeatureDataset, self).__init__()
        assert name in ['train', 'val','val_heldout']
        self.task = task

        #VQA
        if task == "vqa" or task == 'ref_vqa':
            ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
            label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
            self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
            self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
            self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % ('train' if name == 'val_heldout' else name))))

        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % ('train' if name == 'val_heldout' else name))
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))


        self.vqa_entries, self.ref_entries = _load_dataset(dataroot, name, self.img_id2idx)

        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(2)
        self.s_dim = self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.vqa_entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

        for entry in self.ref_entries:
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

        for entry in self.vqa_entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

        for entry in self.ref_entries:
            refexp = torch.from_numpy(np.array(entry['r_token']))
            entry['r_token'] = refexp
            entry['gold_box'] = torch.LongTensor(1).fill_(entry['gold_box'])

    def __getitem__(self, index):
        feats = {'question_id': None, 'question': None, 'target': None, 'image_id': None, 'annotation_id': None,
                 'refexp_id': None, 'refexp': None, 'gold_box': None}

        if len(self.vqa_entries) > 0:
            entry = self.vqa_entries[index]
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]

            question = entry['q_token']
            answer = entry['answer']
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            feats['question_id']=entry['question_id']
            feats['question']= question
            feats['target'] = target
            feats['image_id']= entry['image_id']


        if len(self.ref_entries) > 0:
            entry = self.ref_entries[index]
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]

            refexp = entry['r_token']
            gold_box = entry['gold_box']

            refexp_id = entry['refexp_id']
            image_id = entry['image_id']
            annotation_id = entry['annotation_id']
            feats['image_id'] = image_id
            feats['gold_box'] = gold_box
            feats['refexp']= refexp
            feats['refexp_id'] = refexp_id
            feats['annotation_id'] = annotation_id

        return features, spatials, feats



    def __len__(self):
        return len(self.vqa_entries) + len(self.ref_entries)


