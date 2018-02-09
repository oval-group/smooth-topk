# Smooth Loss Functions for Deep Top-k Classification

This repository contains the implementation of the paper [Smooth Loss Functions for Deep Top-k Classification](https://openreview.net/forum?id=Hk5elxbRW&noteId=Hk5elxbRW) in pytorch. If you use this work for your research, please cite the paper:

```
@Article{berrada2018smooth,
  author       = {Berrada, Leonard and Zisserman, Andrew and Kumar, M Pawan},
  title        = {Smooth Loss Functions for Deep Top-k Classification},
  journal      = {International Conference on Learning Representations},
  year         = {2018},
}
```

The core algorithm to compute the smooth top-k loss function with the elementary symmetric polynomials is contained in the `losses` module. Algorithms for the elementary symmetric polynomials and their gradients are in `losses/polynomial`.

## Requirements

This code has been tested for pytorch 0.3.0. Detailed requirements are available in `requirements.txt`. The code should be compatible with python 2 and 3 (developed in 2.7).

## Reproducing the results

### CIFAR-100

To reproduce the experiments with gpu `1`:
* `scripts/cifar100_noise_ce.sh 1`
* `scripts/cifar100_noise_svm.sh 1`

### ImageNet

We use the official validation set of ImageNet as a test set. Therefore we create our own balanced validation set made of 50,000 training images. This can be done with `scripts/imagenet_split.py`.

To reproduce the experiments:
* `scripts/imagenet_subsets_ce.sh`
* `scripts/imagenet_subsets_svm.sh`

Warning: these scripts will use all available GPUs. To restrict the devices used, use the environment variable `CUDA_VISIBLE_DEVICES`. For example, to train the SVM models on GPUS `0` and `1`, you can run `CUDA_VISIBLE_DEVICES=0,1 scripts/imagenet_subsets_svm.sh`.

The performance of the resulting models can then be obtained by executing `python scripts/eval.py`. This script evaluates the performance of the best models and writes the results in a text file.

### Algorithms Comparison

The script `scripts/perf.py` allows to compare the speed and numerical stability of different algorithms, including the standard algorithm to evaluate the Elementary Symmetric Functions (ESF).

## Acknowledgments

The DenseNet implementation is from [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch).

