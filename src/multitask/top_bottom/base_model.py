import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class BaseModel(nn.Module):
    def __init__(self, task, w_emb, q_emb, v_att, q_net, v_net, classifier):
        super(BaseModel, self).__init__()
        self.task = task
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels, task='vqa'):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        logits = None
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        if task == 'vqa':
	    vs = torch.cat((v, b), dim=2)
            att_logits, att = self.v_att(vs, q_emb)
            v_emb = (att * v).sum(1)  # [batch, v_dim]

            q_repr = self.q_net(q_emb)
            v_repr = self.v_net(v_emb)
            joint_repr = q_repr * v_repr
            logits = self.classifier(joint_repr)

        elif task == 'ref':
            vs = torch.cat((v, b), dim=2)
            logits, att = self.v_att(vs, q_emb)

        return logits


def build_multitask(task, dataset, num_hid):

    ntokens = dataset.dictionary.ntoken
    num_ans_candidates = dataset.num_ans_candidates

    w_emb = WordEmbedding(ntokens, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, num_ans_candidates, 0.5)
    v_att = NewAttention(dataset.v_dim + dataset.s_dim, q_emb.num_hid, num_hid)
    return BaseModel(task, w_emb, q_emb, v_att, q_net, v_net, classifier)

