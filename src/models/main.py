import torch
import models.cifar as cifar_models

from models.parser import parse_model

import torchvision.models as torch_models

from collections import OrderedDict


def get_model(args):
    parse_model(args)

    if args.dataset == 'imagenet':
        model = torch_models.__dict__[args.model]()
        args.model_name = args.model
    elif args.basic_model:
        model = cifar_models.BasicConvNet(args.dataset, args.planes)
        args.model_name = 'convnet_{}'.format(args.planes)
    else:
        model = cifar_models.DenseNet3(args.depth, args.num_classes, args.growth)
        args.model_name = 'densenet_{}_{}'.format(args.depth, args.growth)

    # Print the number of model parameters
    nparams = sum([p.data.nelement() for p in model.parameters()])
    print('Number of model parameters: \t {}'.format(nparams))

    return model


def load_model(model, filename):
    # map location allows to load on CPU weights originally on GPU
    state_dict = torch.load(filename, map_location=lambda storage, loc: storage)
    # map from DataParallel to timple module if needed
    if 'DataParallel' in state_dict['model_repr']:
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        state_dict['model'] = new_state_dict
    model.load_state_dict(state_dict['model'])
