import os

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from .utils import LabelNoise, split_dataset


def get_loaders(args):
    if args.dataset == 'cifar100':
        loaders = loaders_cifar
    elif args.dataset == 'imagenet':
        loaders = loaders_imagenet
    else:
        raise ValueError("dataset {} is not available".format(args.dataset))

    return loaders(dataset_name=args.dataset, batch_size=args.batch_size,
                   test_batch_size=args.test_batch_size,
                   cuda=args.cuda, topk=args.topk, train_size=args.train_size,
                   val_size=args.val_size, noise=args.noise_labels,
                   augment=args.augment, multiple_crops=args.multiple_crops,
                   data_root=args.data_root)


def loaders_cifar(dataset_name, batch_size, cuda,
                  train_size, augment=True, val_size=5000,
                  test_batch_size=1000, topk=None, noise=False,
                  multiple_crops=False, data_root=None):

    assert dataset_name == 'cifar100'
    assert not multiple_crops, "no multiple crops for CIFAR-100"

    data_root = data_root if data_root is not None else os.environ['VISION_DATA']
    root = '{}/{}'.format(data_root, dataset_name)

    # Data loading code
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    normalize = transforms.Normalize(mean=[x / 255.0 for x in mean],
                                     std=[x / 255.0 for x in std])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # define two datasets in order to have different transforms
    # on training and validation (no augmentation on validation)
    dataset_train = datasets.CIFAR100(root=root, train=True,
                                      transform=transform_train)
    dataset_val = datasets.CIFAR100(root=root, train=True,
                                    transform=transform_test)
    dataset_test = datasets.CIFAR100(root=root, train=False,
                                     transform=transform_test)

    # label noise
    if noise:
        dataset_train = LabelNoise(dataset_train, k=5, n_labels=100, p=noise)

    return create_loaders(dataset_name, dataset_train, dataset_val,
                          dataset_test, train_size, val_size, batch_size,
                          test_batch_size, cuda, num_workers=4, noise=noise)


def loaders_imagenet(dataset_name, batch_size, cuda,
                     train_size, augment=True, val_size=50000,
                     test_batch_size=256, topk=None, noise=False,
                     multiple_crops=False, data_root=None):

    assert dataset_name == 'imagenet'
    data_root = data_root if data_root is not None else os.environ['VISION_DATA_SSD']
    root = '{}/ILSVRC2012-prepr-split/images'.format(data_root)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    testdir = os.path.join(root, 'test')

    normalize = transforms.Normalize(mean=mean, std=std)

    if multiple_crops:
        print('Using multiple crops')
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            lambda x: [normalize(transforms.functional.to_tensor(img)) for img in x]])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        transform_train = transform_test

    dataset_train = datasets.ImageFolder(traindir, transform_train)
    dataset_val = datasets.ImageFolder(valdir, transform_test)
    dataset_test = datasets.ImageFolder(testdir, transform_test)

    return create_loaders(dataset_name, dataset_train, dataset_val,
                          dataset_test, train_size, val_size, batch_size,
                          test_batch_size, cuda, noise=noise, num_workers=4)


def create_loaders(dataset_name, dataset_train, dataset_val, dataset_test,
                   train_size, val_size, batch_size, test_batch_size, cuda,
                   num_workers, topk=None, noise=False):

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if cuda else {}

    dataset_train, dataset_val = split_dataset(dataset_train, dataset_val, train_size, val_size)

    print('Dataset sizes: \t train: {} \t val: {} \t test: {}'
          .format(len(dataset_train), len(dataset_val), len(dataset_test)))

    train_loader = data.DataLoader(dataset_train,
                                   batch_size=batch_size,
                                   shuffle=True, **kwargs)

    val_loader = data.DataLoader(dataset_val,
                                 batch_size=test_batch_size,
                                 shuffle=False, **kwargs)

    test_loader = data.DataLoader(dataset_test,
                                  batch_size=test_batch_size,
                                  shuffle=False, **kwargs)

    train_loader.tag = 'train'
    val_loader.tag = 'val'
    test_loader.tag = 'test'

    return train_loader, val_loader, test_loader
