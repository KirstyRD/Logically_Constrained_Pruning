#!/bin/bash

# CIFAR100 ResNet
./prune_fgsm.sh CIFAR100 resnet50 200 100  0.0 0.1 64 cifar100_resnet50_pruning cifar100_resnet50.weights
./prune_fgsm.sh CIFAR100 resnet50 200 100  0.05 0.1 64 cifar100_resnet50_pruning cifar100_resnet50.weights
./prune_fgsm.sh CIFAR100 resnet50 200 100  0.1 0.1 64 cifar100_resnet50_pruning cifar100_resnet50.weights

# CIFAR100 VGG
./prune_fgsm.sh CIFAR100 vgg 250 100  0.0 0.1 64 cifar100_vgg_pruning cifar100_vgg.weights
./prune_fgsm.sh CIFAR100 vgg 250 100  0.05 0.1 64 cifar100_vgg_pruning cifar100_vgg.weights
./prune_fgsm.sh CIFAR100 vgg 250 100  0.1 0.1 64 cifar100_vgg_pruning cifar100_vgg.weights

# GTSRB LeNet
./prune_fgsm.sh GTSRB LeNet 5 50  0.0 0.1 64 gtsrb_lenet_pruning gtsrb_lenet.weights
./prune_fgsm.sh GTSRB LeNet 5 50  0.05 0.1 64 gtsrb_lenet_pruning gtsrb_lenet.weights
./prune_fgsm.sh GTSRB LeNet 5 50  0.1 0.1 64 gtsrb_lenet_pruning gtsrb_lenet.weights
