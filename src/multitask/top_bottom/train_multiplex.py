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

def compute_score_with_logits_vqa(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores, logits

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data #argmax
    return (logits == labels).sum(), logits


def vqa_train(batch, model, optim, total_loss, train_score):
    v, b, feats = batch
    q = feats['question']
    a = feats['target']

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

    batch_score, predicted_logits = compute_score_with_logits_vqa(pred, a.data)
    batch_score = batch_score.sum()
    total_loss += loss.data[0] * v.size(0)
    train_score += batch_score

    return total_loss, train_score


def train_step(batch, model, optim, total_loss, train_score, train_loader, task):
    if task == 'vqa':
        total_loss, train_score = vqa_train(batch, model, optim, total_loss, train_score)
    else:
        total_loss, train_score = ref_train(batch, model, optim, total_loss, train_score)

    total_loss /= len(train_loader.dataset)
    train_score = 100 * train_score / len(train_loader.dataset)
    return total_loss, train_score


def ref_train(batch, model, optim, total_loss, train_score):
    v, b, feats = batch
    q = feats['refexp']
    a = feats['gold_box']

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

    return total_loss, train_score


def get_next_batch(data_iter, count):
    next_batch = None
    try:
        next_batch = next(data_iter)
    except StopIteration:
        #print('completed', count, 'examples')
        pass
    return next_batch


def multitask_train(task, model, train_loaders, eval_loaders, num_epochs, output, run_as):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_vqa_score = 0
    best_eval_ref_score = 0
    # debug total # examples
    if run_as == 'debug':
        total_train_vqa, total_train_ref = 7, 19
    else:
        total_train_vqa = 443283
        total_train_ref = 85140

    for epoch in range(num_epochs):
        total_vqa_loss, total_ref_loss = 0, 0
        train_vqa_score, train_ref_score = 0, 0
        count_train_vqa = 0
        count_train_ref = 0
        t = time.time()

        vqa_train_loader = train_loaders['vqa']
        vqa_eval_loader = eval_loaders['vqa']
        ref_train_loader = train_loaders['ref']
        ref_eval_loader = eval_loaders['ref']
	print(ref_eval_loader.dataset)

        vqa_train_iter = iter(vqa_train_loader)
        ref_train_iter = iter(ref_train_loader)

        while count_train_ref < total_train_ref and count_train_vqa < total_train_vqa:
            sampler = random.uniform(0, 1)
            if sampler < 0.5:
                #logger.write('epoch %d : VQA Training' % epoch)
                batch = get_next_batch(vqa_train_iter, count_train_vqa)
		if not batch:
		    break
                total_vqa_loss, train_vqa_score = train_step(batch, model, optim, total_vqa_loss, train_vqa_score,
                                                             vqa_train_loader, 'vqa')
                count_train_vqa += len(batch[0])
            else:
                #logger.write('epoch %d : RefExp Training' % epoch)
                batch = get_next_batch(ref_train_iter, count_train_ref)
		if not batch:
		    break
                total_ref_loss, train_ref_score = train_step(batch, model, optim, total_ref_loss, train_ref_score,
                                                             ref_train_loader, 'ref')
                count_train_ref += len(batch[0])

        while count_train_vqa < total_train_vqa:
            #logger.write('epoch %d : VQA Training' % epoch)
            batch = get_next_batch(vqa_train_iter, count_train_vqa)
	    if not batch:
	        break
            total_vqa_loss, train_vqa_score = train_step(batch, model, optim, total_vqa_loss, train_vqa_score,
                                                         vqa_train_loader, 'vqa')
            count_train_vqa += len(batch[0])

        while count_train_ref < total_train_ref:
            #logger.write('epoch %d : RefExp Training' % epoch)
            batch = get_next_batch(ref_train_iter, count_train_ref)
	    if not batch:
	        break
            total_ref_loss, train_ref_score = train_step(batch, model, optim, total_ref_loss, train_ref_score,
                                                         ref_train_loader, 'ref')
            count_train_ref += len(batch[0])

        model.train(False)
        eval_vqa_score, _ = eval(model, vqa_eval_loader, 'vqa')
        eval_ref_score, _ = eval(model, ref_eval_loader, 'ref')
        model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\tvqa: train loss: %.6f, score: %.6f' % (total_vqa_loss, train_vqa_score))
        logger.write('\teval score: %.6f' % eval_vqa_score)
        logger.write('\tref: train loss: %.6f, score: %.6f' % (total_ref_loss, train_ref_score))
        logger.write('\teval score: %.6f' % eval_ref_score)

        if eval_vqa_score > best_eval_vqa_score:
            logger.write('Saving model for vqa')
            model_path = os.path.join(output, 'model_vqa.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_vqa_score = eval_vqa_score

        if eval_ref_score > best_eval_ref_score:
            logger.write('Saving model for ref')
            model_path = os.path.join(output, 'model_ref.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_ref_score = eval_ref_score


def eval(model, dataloader, task='vqa'):
    score = 0
    num_data = 0
    if task == 'vqa':
        upper_bound = 0
        for i, (v, b, feats) in enumerate(dataloader):
            q = feats['question']
            a = feats['target']

            v = Variable(v, volatile=True).cuda()
            b = Variable(b, volatile=True).cuda()
            q = Variable(q, volatile=True).cuda()
            pred = model(v, b, q, None, 'vqa')
            batch_score, logits = compute_score_with_logits_vqa(pred, a.cuda())
            batch_score = batch_score.sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += pred.size(0)

        score = score / len(dataloader.dataset)
        upper_bound = upper_bound / len(dataloader.dataset)
        return score, upper_bound

    elif task == 'ref':
        for i, (v, b, feats) in enumerate(dataloader):
            q = feats['refexp']
            a = feats['gold_box']
            if torch.cuda.is_available():
                v = Variable(v, volatile=True).cuda()
                b = Variable(b, volatile=True).cuda()
                q = Variable(q, volatile=True).cuda()
                a = a.cuda()
            else:
                v = Variable(v, volatile=True)
                b = Variable(b, volatile=True)
                q = Variable(q, volatile=True)
            pred = model(v, b, q, None, 'ref')

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
            pred = model(v, b, q, None, 'vqa')
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
            pred = model(v, b, q, None, 'ref')

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
