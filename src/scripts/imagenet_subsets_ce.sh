#!/bin/bash

lr_0=0.1

python main.py --dataset imagenet --loss ce --out-name ../xp/imagenet/im64k_ce \
    --parallel-gpu --train-size 64000 --lr_0 $lr_0 --no-visdom;

python main.py --dataset imagenet --loss ce --out-name ../xp/imagenet/im128k_ce \
    --parallel-gpu --train-size 128000 --lr_0 $lr_0 --no-visdom;

python main.py --dataset imagenet --loss ce --out-name ../xp/imagenet/im320k_ce \
    --parallel-gpu --train-size 320000 --lr_0 $lr_0 --no-visdom;

python main.py --dataset imagenet --loss ce --out-name ../xp/imagenet/im640k_ce \
    --parallel-gpu --train-size 640000 --lr_0 $lr_0 --no-visdom;

python main.py --dataset imagenet --loss ce --out-name ../xp/imagenet/imall_ce \
    --parallel-gpu --lr_0 $lr_0 --no-visdom;