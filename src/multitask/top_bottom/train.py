import os
import time
import random
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import pdb

random.seed(123)


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


def train(model, train_loaders, eval_loaders, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0

    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        sampler = random.uniform(0, 1)

        # VQA training loop
        if sampler < 0.3:
            logger.write('epoch %d : VQA Training', epoch)
            train_loader = train_loaders['vqa']
            eval_loader = eval_loader['vqa']

            for i, (v, b, q, a, image_id, question_id) in enumerate(train_loader):
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()

                pred = model(v, b, q, a, 'vqa')
                loss = instance_bce_with_logits(pred, a)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)
                optim.step()
                optim.zero_grad()

                batch_score = compute_score_with_logits(pred, a.data).sum()
                total_loss += loss.data[0] * v.size(0)
                train_score += batch_score

            total_loss /= len(train_loader.dataset)
            train_score = 100 * train_score / len(train_loader.dataset)
            model.train(False)
            eval_score, bound = eval(model, eval_loader, 'vqa')
            model.train(True)

            if eval_score > best_eval_score:
                logger.write('Saving model for VQA')
                model_path = os.path.join(output, 'model_vqa.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score

        # Refexp training loop
        else:
            logger.write('epoch %d : RefExp Training', epoch)
            train_loader = train_loaders['ref']
            eval_loader = eval_loaders['ref']

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

                pred = model(v, b, q, a, 'ref')

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
            eval_score = eval(model, eval_loader, 'ref')
            model.train(True)

            if eval_score > best_eval_score:
                logger.write('Saving model for RefExp')
                model_path = os.path.join(output, 'model_refexp_best.pth')
                torch.save(model.state_dict(), model_path)
                best_eval_score = eval_score

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f' % eval_score)

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model_multi_best.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score


def eval(model, dataloader, task='vqa'):
    score = 0
    num_data = 0
    if task == 'vqa':
        upper_bound = 0
        for v, b, q, a, image_id, question_id in iter(dataloader):
            v = Variable(v, volatile=True).cuda()
            b = Variable(b, volatile=True).cuda()
            q = Variable(q, volatile=True).cuda()
            pred = model(v, b, q, None)
            batch_score, logits = compute_score_with_logits(pred, a.cuda())
            batch_score = batch_score.sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
        return score, upper_bound

    elif task == 'ref':
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


def evaluate(model, dataloader, task='vqa'):
    score = 0
    upper_bound = 0
    num_data = 0
    result_log = []

    if task == 'vqa':
        for v, b, q, a, image_id, question_id in iter(dataloader):
            v = Variable(v, volatile=True).cuda()
            b = Variable(b, volatile=True).cuda()
            q = Variable(q, volatile=True).cuda()
            pred = model(v, b, q, None)
            batch_score, predicted_logits = compute_score_with_logits(pred, a.cuda())
            batch_score = batch_score.sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)
            # iterate over batch and populate result_log
            for i in range(v.size(0)):
                result_log.append(
                    [image_id[i].item(), question_id[i].item(), predicted_logits[i].item()])

        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
        return score, result_log

    elif task == 'ref':
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
                    pred[i].detach().cpu().numpy()
                ])

        score = 100 * score / len(dataloader.dataset)
        return score, result_log
