import os
import shutil
import torchvision.datasets as datasets

from collections import defaultdict
from data.utils import random_subsets


data_root = os.environ['VISION_DATA_SSD']


train_root = '{}/ILSVRC2012-prepr-split/images/train'.format(data_root)
val_root = '{}/ILSVRC2012-prepr-split/images/val'.format(data_root)
dataset_train = datasets.ImageFolder(train_root)

if not os.path.exists(val_root):
    os.makedirs(val_root)
else:
    assert len(os.listdir(val_root)) == 0, \
        "{} is not empty: split already performed?".format(val_root)
    print("{} initially empty".format(val_root))

n_classes = len(dataset_train.classes)
val_size_per_class = 50
assert val_size_per_class > 0
my_dict = defaultdict(list)
[my_dict[e[1]].append(e[0]) for e in dataset_train.imgs]
val_imgs = []
for k in my_dict.keys():
    imgs = sorted(my_dict[k])
    val_indices, = random_subsets((val_size_per_class,),
                                  len(imgs),
                                  seed=1234 + int(k))
    val_imgs += [imgs[idx] for idx in val_indices]

counter = dict()
for img in val_imgs:
    id_ = img.split('/')[-2]
    if id_ in counter.keys():
        counter[id_] += 1
    else:
        counter[id_] = 1

balanced = len(set(counter.values())) == 1
if balanced:
    print("data set is properly balanced")
else:
    raise ValueError("data set should be balanced")

print("Number of labels: {}".format(len(counter)))
print("Number of images per label: {}".format(counter.values()[0]))

print("Creating directories...")
for new_dir in os.listdir(train_root):
    os.makedirs(os.path.join(val_root, new_dir))

for img in val_imgs:
    new_img = img.replace("train", "val")
    print("Moving {} to {}".format(img, new_img))
    shutil.move(img, new_img)
