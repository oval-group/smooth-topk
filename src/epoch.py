import torch.nn as nn
import torch.autograd as ag

from utils import regularization, accuracy, print_stats, dump_results
from tqdm import tqdm


def data_to_var(data, target, cuda, volatile=False):

    if cuda:
        data = data.cuda()
        target = target.cuda()
    data = ag.Variable(data, volatile=volatile)
    target = ag.Variable(target)

    return data, target


def train(model, loss, optimizer, loader, xp, args):

    if not len(loader):
        return 0

    model.train()

    xp.Parent_Train.reset()

    for batch_idx, (data, target) in tqdm(enumerate(loader), desc='Train Epoch',
                                          leave=False, total=len(loader)):
        data, target = data_to_var(data, target, args.cuda)

        output = model(data)
        obj = loss(output, target)

        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        prec1 = accuracy(output.data, target.data, topk=1)
        preck = accuracy(output.data, target.data, topk=xp.config['topk'])
        xp.Parent_Train.update(loss=obj.data[0], acck=preck, acc1=prec1, n=data.size(0))

    # compute objective function (including regularization)
    obj = xp.Loss_Train.get() + regularization(model, xp.mu)
    xp.Obj_Train.update(obj)
    # measure elapsed time
    xp.Timer_Train.update()

    xp.log_with_tag('train')

    if args.verbosity:
        print_stats(xp, 'train')


def test(model, loss, loader, xp, args):

    if not len(loader):
        return 0

    model.eval()

    metrics = xp.get_metric(tag=loader.tag, name='parent')
    timer = xp.get_metric(tag=loader.tag, name='timer')

    metrics.reset()

    if args.multiple_crops:
        epoch_test_multiple_crops(model, loader, xp, args.cuda)
    else:
        epoch_test(model, loader, xp, args.cuda)

    # measure elapsed time
    timer.update()
    xp.log_with_tag(loader.tag)

    if loader.tag == 'val':
        xp.Acc1_Val_Best.update(xp.acc1_val).log()
        xp.Acck_Val_Best.update(xp.acck_val).log()

    if args.verbosity:
        print_stats(xp, loader.tag)

    if args.eval:
        dump_results(xp, args)


def epoch_test(model, loader, xp, cuda):
    metrics = xp.get_metric(tag=loader.tag, name='parent')
    for batch_idx, (data, target) in tqdm(enumerate(loader), desc='Test Epoch',
                                          leave=False, total=len(loader)):
        data, target = data_to_var(data, target, cuda, volatile=True)
        output = model(data)

        prec1 = accuracy(output.data, target.data, topk=1)
        preck = accuracy(output.data, target.data, topk=xp.config['topk'])
        metrics.update(acck=preck, acc1=prec1, n=data.size(0))


def epoch_test_multiple_crops(model, loader, xp, cuda):
    metrics = xp.get_metric(tag=loader.tag, name='parent')
    xp.Temperature.update()
    for batch_idx, (data, target) in tqdm(enumerate(loader), desc='Test Epoch',
                                          leave=False, total=len(loader)):

        target = ag.Variable(target.cuda())
        avg = 0
        for img in data:
            img = ag.Variable(img.cuda(), volatile=True)
            output = model(img)
            # cross-entropy
            if xp.temperature == -1:
                avg += nn.functional.softmax(output).data
            # smooth-svm
            else:
                avg += output.data
                # avg += torch.exp(output.data / xp.temperature)

        prec1 = accuracy(avg, target.data, topk=1)
        preck = accuracy(avg, target.data, topk=xp.config['topk'])
        metrics.update(acck=preck, acc1=prec1, n=target.size(0))
