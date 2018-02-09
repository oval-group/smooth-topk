import os
import logger
import torch
import random
import numpy as np
import torch.optim as optim

from torch.autograd import Variable


def load_optimizer(optimizer, filename, lr_0=None):
    state_dict = torch.load(filename)
    optimizer.load_state_dict(state_dict['opt'])

    if lr_0 is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_0


def set_seed(seed, print_out=True):
    if print_out:
        print('Seed:\t {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def regularization(model, mu):
    reg = 0.5 * mu * sum(p.data.norm() ** 2 for p in model.parameters())
    return reg


def accuracy(out, targets, topk=1):
    if isinstance(out, Variable):
        out = out.data
    if isinstance(targets, Variable):
        targets = targets.data

    if topk == 1:
        _, pred = torch.max(out, 1)
        acc = torch.mean(torch.eq(pred, targets).float())
    else:
        _, pred = out.topk(topk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        acc = correct[:topk].view(-1).float().sum(0) / out.size(0)

    return acc


def decay_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1
    print('Switching lr to {}'.format(optimizer.param_groups[0]['lr']))
    return optimizer


def get_optimizer(model, mu, lr, xp):
    optimizer = optim.SGD(params=model.parameters(),
                          weight_decay=mu,
                          nesterov=True,
                          lr=lr,
                          momentum=0.9)

    # link metrics to new optimizer:

    # Learning rate and regularization hyper-parameters
    xp.Learning_Rate.set_fun(lambda: optimizer.param_groups[0]['lr'])
    xp.Mu.set_fun(lambda: optimizer.param_groups[0]['weight_decay'])

    # update saved model at every epoch
    xp.Epoch.reset_hooks()
    xp.Epoch.add_hook(
        lambda: save_out(xp, "{}".format(xp.name_and_dir), model, optimizer))

    # save best model
    xp.Acc1_Val_Best.reset_hooks()
    xp.Acc1_Val_Best.add_hook(
        lambda: save_out(xp, "{}_best1".format(xp.name_and_dir), model, optimizer))
    xp.Acck_Val_Best.reset_hooks()
    xp.Acck_Val_Best.add_hook(
        lambda: save_out(xp, "{}_bestk".format(xp.name_and_dir), model, optimizer))

    return optimizer


def update_optimizer(lr_schedule, optimizer, model, loss, xp):

    # beginning of training - nothing to monitor yet
    if int(xp.Epoch.value) in lr_schedule:
        optimizer = decay_lr(optimizer)

    return optimizer


def create_experiment(args):

    xp = logger.Experiment(args.out_name,
                           use_visdom=args.visdom,
                           visdom_opts=dict(server=args.server,
                                            port=args.port),
                           time_indexing=False,
                           xlabel='Epoch')
    xp.ParentWrapper(tag='train', name='parent',
                     children=(xp.AvgMetric(name='loss'),
                               xp.AvgMetric(name='acc1'),
                               xp.AvgMetric(name='acck'),
                               xp.SimpleMetric(name='obj'),
                               xp.TimeMetric(name='timer')))

    xp.ParentWrapper(tag='val', name='parent',
                     children=(xp.AvgMetric(name='acck'),
                               xp.AvgMetric(name='acc1'),
                               xp.TimeMetric(name='timer')))

    xp.ParentWrapper(tag='test', name='parent',
                     children=(xp.AvgMetric(name='acc1'),
                               xp.AvgMetric(name='acck'),
                               xp.TimeMetric(name='timer')))

    xp.SumMetric(name='epoch', to_plot=False)
    xp.DynamicMetric(name='learning_rate')
    xp.DynamicMetric(name='temperature', to_plot=False)
    xp.DynamicMetric(name='mu', to_plot=False)

    xp.BestMetric(tag='val_best', name='acc1', mode='max')
    xp.BestMetric(tag='val_best', name='acck', mode='max')

    if args.visdom:
        xp.plotter.set_win_opts(name='acc1', opts={'title': 'Accuracy@1'})
        xp.plotter.set_win_opts(name='acck', opts={'title': 'Accuracy@k'})
        xp.plotter.set_win_opts(name='loss', opts={'title': 'Loss Function'})
        xp.plotter.set_win_opts(name='obj', opts={'title': 'Objective Function'})
        xp.plotter.set_win_opts(name='learning_rate', opts={'title': 'Learning Rate'})
        xp.plotter.set_win_opts(name='Timer', opts={'title': 'Time (s) / epoch'})

    xp.log_config(vars(args))

    return xp


def save_out(xp, out_name, model, optimizer):
    # save out current logs, model and optimizer
    xp.to_json("{}.json".format(out_name))
    with open('{}.pkl'.format(out_name), 'w') as f:
        torch.save({'epoch': int(xp.Epoch.value),
                    'model': model.state_dict(),
                    'model_repr': repr(model),
                    'opt': optimizer.state_dict()}, f)


def dump_results(xp, args):
    xp_loaded = args.load_model
    acc1 = xp.acc1_test
    acck = xp.acck_test

    filename = '../xp/{}_results.txt'.format(args.dataset)
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            f.write('{0:60} {1:12} {2:12}\n'.format('Experiment name', 'Accuracy@1', 'Accuracy@k'))

    with open(filename, 'a') as f:
        f.write('{0:60} {1:12} {2:12}\n'.format(xp_loaded, acc1, acck))


def print_stats(xp, tag):

    stats = xp.get_metric(tag=tag, name='parent').get()

    if tag == 'train':
        stats['obj'] = '{0:.3g}'.format(stats['obj'])
        stats['loss'] = '{0:.3g}'.format(stats['loss'])
    else:
        stats['obj'] = stats['loss'] = '----'

    stats['tag'] = tag.title()
    stats['topk'] = xp.config['topk']

    stats['acc1'] *= 100.
    stats['acck'] *= 100.

    stats['epoch'] = int(xp.epoch) if xp.epoch is not None else 0

    print('\nEpoch: [{epoch}] ({tag})\t'
          'Time {timer:.1f}s\t'
          'Obj {obj:.4}\t'
          'Loss {loss:.4}\t'
          'Prec@1 {acc1:.2f}%\t'
          'Prec@{topk} {acck:.2f}%\t'.format(**stats))
