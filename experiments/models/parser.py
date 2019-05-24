def parse_model(args):
    if args.dataset == 'imagenet':
        pass
    elif 'basic' in args.model:
        parse_basic_convnet(args)
    elif 'densenet' in args.model:
        parse_densenet(args)


def parse_basic_convnet(args):
    args.basic_model = 1
    args.densenet_model = 0

    param_str = args.model.replace("basic", "")
    param_str = param_str.replace("_", "-")
    args.planes = [int(p) for p in param_str.split("-") if p != ''].pop(0)


def parse_densenet(args):
    args.densenet_model = 1
    args.basic_model = 0

    param_str = args.model.replace("densenet", "")
    param_str = param_str.replace("_", "-")
    args.depth, args.growth = \
        [int(p) for p in param_str.split("-") if p != '']
