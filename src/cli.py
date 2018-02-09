import logger
import sys
import socket
import os
import torch


imagenet_defaults = dict(noise_labels=0,
                         topk=5,
                         num_classes=1000,
                         batch_size=256,
                         epochs=120,
                         train_size=1281167,
                         val_size=50000,
                         lr_schedule=(31, 61, 91),
                         model='resnet18')

cifar100_defaults = dict(noise_labels=1,
                         topk=5,
                         num_classes=100,
                         batch_size=64,
                         epochs=300,
                         train_size=45000,
                         val_size=5000,
                         lr_schedule=(151, 226))


def set_defaults(args):
    # force no logging in debug mode
    if args.debug:
        args.visdom = False
        args.log = False
        args.out_name = '../xp/debug'
        # remove previous log in debug mode
        if os.path.exists('../xp/debug_log.txt'):
            os.remove('../xp/debug_log.txt')

    elif args.eval:
        args.visdom = False
        args.log = False
        args.out_name = '../xp/{}'.format(args.dataset)
        args.epochs = 0
        # find settings of experiment
        _xp = logger.Experiment("")
        _xp.from_json(args.load_model.replace(".pkl", ".json"))
        for k in ('topk', 'model'):
            setattr(args, k, _xp.config[k])
        if args.multiple_crops:
            args.test_batch_size = 32

    assert args.dataset in ('imagenet', 'cifar100')
    if args.dataset == 'imagenet':
        my_dict = imagenet_defaults
    if args.dataset == 'cifar100':
        my_dict = cifar100_defaults

    # set number of classes
    args.num_classes = my_dict['num_classes']

    # replace None values with default ones
    for (k, v) in my_dict.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    # store full command line in args
    args.command_line = ' '.join(sys.argv)

    # store cuurent directory and hostname in args
    args.cwd = os.getcwd()
    args.hostname = socket.gethostname()


def add_all_parsers(parser):
    _add_dataset_parser(parser)
    _add_model_parser(parser)
    _add_loss_parser(parser)
    _add_hardware_parser(parser)
    _add_training_parser(parser)
    _add_misc_parser(parser)


def _add_dataset_parser(parser):
    d_parser = parser.add_argument_group(title='Dataset parameters')

    d_parser.add_argument('--dataset', required=True,
                          choices=('imagenet', 'cifar100'),
                          help='dataset (required)')
    d_parser.add_argument('--data_root', type=str, default=None,
                          help="root directory of data")
    d_parser.add_argument('--train-size', type=int,
                          help="training data size")
    d_parser.add_argument('--val-size', type=int,
                          help="validation data size")
    d_parser.add_argument('--noise-labels', type=float,
                          help="Noisy labels")
    d_parser.add_argument('--augment', type=int, default=1,
                          help="whether to use data augmentation")
    d_parser.add_argument('--multiple-crops', dest='multiple_crops',
                          action='store_true',
                          help="ten crops at evaluation time")
    d_parser.set_defaults(multiple_crops=False)


def _add_model_parser(parser):
    m_parser = parser.add_argument_group(title='Model parameters')

    m_parser.add_argument('--model', type=str, default=None,
                          help="Model description. Examples:\n"
                          "- densenet-40-12\n"
                          "- basic-64\n")
    m_parser.add_argument('--load-model', type=str, default=None,
                          help="Load model weights from file")
    m_parser.add_argument('--load-optimizer', type=str, default=None,
                          help="Load optimizer state from file")


def _add_hardware_parser(parser):
    c_parser = parser.add_argument_group(title='Cuda hardware parameters')

    c_parser.add_argument('--cuda', type=int,
                          default=torch.cuda.is_available())
    c_parser.add_argument('--device', type=int, default=0,
                          help="cuda device")
    c_parser.add_argument('--parallel-gpu', dest='parallel_gpu',
                          action='store_true',
                          help="parallel gpu computation")
    c_parser.set_defaults(parallel_gpu=False)


def _add_training_parser(parser):
    t_parser = parser.add_argument_group(title='Training parameters')

    t_parser.add_argument('--lr_0', type=float, default=0.1,
                          help="Initial learning rate")
    t_parser.add_argument('--batch-size', type=int,
                          help="batch size")
    t_parser.add_argument('--test-batch-size', type=int,
                          default=256, help="test batch size")
    t_parser.add_argument('--epochs', type=int,
                          default=None,
                          help="number of epochs")
    t_parser.add_argument('--lr-schedule', default=None,
                          help="number of epochs")


def _add_loss_parser(parser):
    l_parser = parser.add_argument_group(title='Loss parameters')

    l_parser.add_argument('--loss', type=str, required=True,
                          choices=['svm', 'ce', 'svm-lapin'],
                          help="loss ('svm' or 'ce')")
    l_parser.add_argument('--topk', type=int,
                          help="Top-k error to minimize")
    l_parser.add_argument('--alpha', type=float, default=1.,
                          help="scaling of classification margin")
    l_parser.add_argument('--tau', type=float, default=1,
                          help="temperature parameter")
    l_parser.add_argument('--mu', type=float, default=1e-4,
                          help="l2-regularization hyperparameter")


def _add_misc_parser(parser):
    m_parser = parser.add_argument_group(title='Miscellaneous parameters')

    m_parser.add_argument('--out-name', type=str, default=None,
                          help="output name")
    m_parser.add_argument('--seed', type=int, default=0)
    m_parser.add_argument('--server', type=str,
                          default="http://atlas.robots.ox.ac.uk",
                          help="server address for visdom")
    m_parser.add_argument('--port', type=int, default=9001,
                          help="server port for visdom")
    m_parser.add_argument('--verbosity', type=int, default=1)
    m_parser.add_argument('--no-visdom', dest='visdom', action='store_false',
                          help='use visdom (default: True)')
    m_parser.add_argument('--no-log', dest='log', action='store_false',
                          help='log model and results (default: True)')
    m_parser.add_argument('--debug', dest='debug', action='store_true',
                          help='activate debug mode (default: False)')
    m_parser.add_argument('--eval', dest='eval', action='store_true',
                          help='activate evaluation mode (default: False)')
    m_parser.set_defaults(visdom=True, log=True, debug=False, eval=False)
