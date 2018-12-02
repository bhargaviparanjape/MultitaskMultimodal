import torch
import torch.nn as nn
from attention import Attention, NewAttention
from context_attention import ContextAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
import pdb


class BaseRefexModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att):
        super(BaseRefexModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]
        
        vs = torch.cat((v,b), dim=2)
        logits = self.v_att(vs, q_emb)
        return logits

class BaseRefexModelAttn(nn.Module):
    def __init__(self, w_emb, q_emb, cv_att, v_att):
        super(BaseRefexModelAttn, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.cv_att = cv_att
        self.v_att = v_att

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)
        vs = torch.cat((v,b), dim=2)
        q_emb, _ = self.cv_att(vs, q_emb)
        
        logits = self.v_att(vs, q_emb)
        return logits

def build_refex_baseline(dataset, num_hid, bidirectional):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, bidirectional, 0.0)
    v_att = NewAttention(dataset.v_dim + dataset.s_dim, (q_emb.num_hid * (int(bidirectional) + 1)), num_hid)
    return BaseRefexModel(w_emb, q_emb, v_att)

def build_refex_baseline_attn(dataset, num_hid, bidirectional):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    bidirectional = True
    q_emb = QuestionEmbedding(300, num_hid, 1, bidirectional, 0.0)

    dim1, dim2 = dataset.v_dim + dataset.s_dim, (q_emb.num_hid * (int(bidirectional) + 1))
    cv_att = ContextAttention(dim1, dim2)
    v_att = NewAttention(dim1, dim2, num_hid)
    return BaseRefexModelAttn(w_emb, q_emb, cv_att, v_att)
