import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets

from collections import defaultdict


class LabelNoise(data.Dataset):
    def __init__(self, dataset, k, n_labels, p=1):

        assert n_labels % k == 0

        self.dataset = dataset
        self.k = k
        # random label between 0 and k-1
        self.noise = np.random.choice(k, size=len(self.dataset))
        # noisy labels are introduced for each sample with probability p
        self.p = np.random.binomial(1, p, size=len(self.dataset))

        print('Noisy labels (p={})'.format(p))

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.p[idx]:
            label = label - label % self.k + self.noise[idx]
        return img, label

    def __len__(self):
        return len(self.dataset)


class Subset(data.Dataset):
    def __init__(self, dataset, indices=None):
        """
        Subset of dataset given by indices.
        """
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices

        if self.indices is None:
            self.n_samples = len(self.dataset)
        else:
            self.n_samples = len(self.indices)
            assert self.n_samples >= 0 and \
                self.n_samples <= len(self.dataset), \
                "length of {} incompatible with dataset of size {}"\
                .format(self.n_samples, len(self.dataset))

    def __getitem__(self, idx):
        if self.indices is None:
            return self.dataset[idx]
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.n_samples


def random_subsets(subset_sizes, n_total, seed=None, replace=False):
    """
    Return subsets of indices, with sizes given by the iterable
    subset_sizes, drawn from {0, ..., n_total - 1}
    Subsets may be distinct or not according to the replace option.
    Optional seed for deterministic draw.
    """
    # save current random state
    state = np.random.get_state()
    sum_sizes = sum(subset_sizes)
    assert sum_sizes <= n_total

    np.random.seed(seed)

    total_subset = np.random.choice(n_total, size=sum_sizes,
                                    replace=replace)
    perm = np.random.permutation(total_subset)
    res = []
    start = 0
    for size in subset_sizes:
        res.append(perm[start: start + size])
        start += size
    # restore initial random state
    np.random.set_state(state)
    return res


def split_dataset(dataset_train, dataset_val, train_size, val_size):
    if isinstance(dataset_train, datasets.ImageFolder):
        n_classes = len(dataset_train.classes)
        if train_size < len(dataset_train):
            train_size_per_class = train_size // n_classes
        else:
            train_size_per_class = float('inf')
        assert train_size_per_class > 0
        my_dict = defaultdict(list)
        [my_dict[e[1]].append(e[0]) for e in dataset_train.imgs]
        train_imgs = []
        for k in my_dict.keys():
            imgs = my_dict[k]
            adapted_train_size = min(train_size_per_class, len(imgs))
            train_indices, = random_subsets((adapted_train_size,),
                                            len(imgs),
                                            seed=1234 + int(k))
            train_imgs += [(imgs[idx], int(k)) for idx in train_indices]
        np.random.shuffle(train_imgs)
        dataset_train.imgs = train_imgs
    else:
        train_indices, val_indices = random_subsets((train_size, val_size),
                                                    len(dataset_train),
                                                    seed=1234)

        dataset_train = Subset(dataset_train, train_indices)
        dataset_val = Subset(dataset_val, val_indices)

    return dataset_train, dataset_val
