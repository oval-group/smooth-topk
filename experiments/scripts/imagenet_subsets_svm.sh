#!/bin/bash

lr_0=1
tau=0.1
mu=0.000025

python main.py --dataset imagenet --loss svm --out-name ../results/imagenet/im64k_svm \
    --parallel-gpu --train-size 64000 --lr_0 $lr_0 --tau $tau --mu $mu --no-visdom;

python main.py --dataset imagenet --loss svm --out-name ../results/imagenet/im128k_svm \
    --parallel-gpu --train-size 128000 --lr_0 $lr_0 --tau $tau --mu $mu --no-visdom;

python main.py --dataset imagenet --loss svm --out-name ../results/imagenet/im320k_svm \
    --parallel-gpu --train-size 320000 --lr_0 $lr_0 --tau $tau --mu $mu --no-visdom;

python main.py --dataset imagenet --loss svm --out-name ../results/imagenet/im640k_svm \
    --parallel-gpu --train-size 640000 --lr_0 $lr_0 --tau $tau --mu $mu --no-visdom;

python main.py --dataset imagenet --loss svm --out-name ../results/imagenet/imall_svm \
    --parallel-gpu --lr_0 $lr_0 --tau $tau --mu $mu --no-visdom;
