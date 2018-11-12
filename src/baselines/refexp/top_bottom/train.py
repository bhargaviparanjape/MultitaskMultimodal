import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import pdb


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def cross_entropy(logits, labels):
    loss = nn.functional.cross_entropy(logits, labels)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data #argmax
    return (logits == labels).sum(), logits


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a, image_id, annotation_id, refexp_id) in enumerate(train_loader):
            if torch.cuda.is_available():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()
            else:
                v = Variable(v)
                b = Variable(b)
                q = Variable(q)
                a = Variable(a)


            pred = model(v, b, q, a)

            pred = pred.squeeze(-1)
            a = a.squeeze(-1)
            ## Cross Entropy Loss
            loss = cross_entropy(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score, predicted_labels = compute_score_with_logits(pred, a.data)
            total_loss += loss.data[0] * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score = eval(model, eval_loader)
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f' % eval_score)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def eval(model, dataloader):
    score = 0
    num_data = 0
    for v, b, q, a, image_id, annotation_id, refexp_id in iter(dataloader):
        if torch.cuda.is_available():
            v = Variable(v, volatile=True).cuda()
            b = Variable(b, volatile=True).cuda()
            q = Variable(q, volatile=True).cuda()
            a = a.cuda()
        else:
            v = Variable(v, volatile=True)
            b = Variable(b, volatile=True)
            q = Variable(q, volatile=True)
        pred = model(v, b, q, None)

        pred = pred.squeeze(-1)
        a = a.squeeze(-1)

        batch_score, predicted_labels = compute_score_with_logits(pred, a)
        score += batch_score
        num_data += pred.size(0)

    score = 100 * score / len(dataloader.dataset)
    return score


def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    result_log = []
    for v, b, q, a, image_id, annotation_id, refexp_id in iter(dataloader):
        q_vector = q
        if torch.cuda.is_available():
            v = Variable(v, volatile=True).cuda()
            b = Variable(b, volatile=True).cuda()
            q = Variable(q, volatile=True).cuda()
            a = a.cuda()
        else:
            v = Variable(v, volatile=True)
            b = Variable(b, volatile=True)
            q = Variable(q, volatile=True)
        pred = model(v, b, q, None)

        pred = pred.squeeze(-1)
        a = a.squeeze(-1)

        batch_score, predicted_labels = compute_score_with_logits(pred, a)
        score += batch_score
        num_data += pred.size(0)

        # iterate over batch and populate result_log
        for i in range(v.size(0)):
            result_log.append([
                image_id[i].item(),
                annotation_id[i].item(),
                refexp_id[i].item(),
                predicted_labels[i].item(),
                q_vector[i].numpy(),
                pred[i].detach().numpy()
            ])

    score = 100 * score / len(dataloader.dataset)
    return score, result_log
