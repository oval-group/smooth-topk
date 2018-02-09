import torch.nn as nn
from losses.svm import SmoothSVM


def get_loss(xp, args):
    if args.loss == "svm":
        print("Using SVM loss")
        loss = SmoothSVM(n_classes=args.num_classes, k=args.topk, tau=args.tau, alpha=args.alpha)
    elif args.loss == 'ce':
        print("Using CE loss")
        loss = nn.CrossEntropyLoss()
        loss.tau = -1
    else:
        raise ValueError('Invalid choice of loss ({})'.format(args.loss))

    xp.Temperature.set_fun(lambda: loss.tau)

    return loss
