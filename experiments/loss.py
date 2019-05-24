import torch.nn as nn
from topk.svm import SmoothTop1SVM, SmoothTopkSVM


def get_loss(xp, args):
    if args.loss == "svm":
        print("Using SVM loss")
        if args.topk == 1:
            loss = SmoothTop1SVM(n_classes=args.num_classes, alpha=args.alpha,
                                 tau=args.tau)
        else:
            loss = SmoothTopkSVM(n_classes=args.num_classes, alpha=args.alpha,
                                 tau=args.tau, k=args.topk)
    elif args.loss == 'ce':
        print("Using CE loss")
        loss = nn.CrossEntropyLoss()
        loss.tau = -1
    else:
        raise ValueError('Invalid choice of loss ({})'.format(args.loss))

    xp.Temperature.set_fun(lambda: loss.tau)

    return loss
