# Logically Constrained Pruning

This repo contains code to prune neural network models using logical constaints.

## Install

Clone this repo:

```
git clone https://github.com/KirstyRD/Logically_Constrained_Pruning.git
cd Logically_Constrained_Pruning/
```

This code was tested with the libraries listed in requirements.txt and CUDA v10.0.

```
pip install -r requirements.txt
```

## Run

The file prune.sh produces pruned models from the paper. Note that only one constraint is used here. Any constraint from [DL2](https://github.com/eth-sri/dl2) can be used instead by modifying the --constraint parameter. The file plot.sh plots all graphs found in the paper e.g. the following figures; Prediction accuracy, robustess and deterioration of ResNet50 on CIFAR-100 with Global Robustness constrant and FGSM sampling ([Casadio et al [3]](https://github.com/aisec-private/training-with-constraints).)


<p float="left">
  <img src="/plots/CIFAR100_resnet50_accuracy.png" width="250" />
  <img src="/plots/CIFAR100_resnet50_robustness.png" width="250" />
  <img src="/plots/degen_CIFAR100_resnet50.png" width="250" />
</p>


There is a heavy reuse of code from paper references [DL2 [7]](https://github.com/eth-sri/dl2) and [NVIDIA Pruning [18]](https://github.com/NVlabs/Taylor_pruning)
