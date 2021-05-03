# Logically Constrained Pruning

This repo contains code to prune neural network models using logical constaints.

The file prune.sh roduces pruned models from the paper. Note that only one constraint is used here. Any constraint from [DL2](https://github.com/eth-sri/dl2) can be used instead. The file plot.sh plots results into graphs.

![Accuracy](https://github.com/KirstyRD/Logically_Constrained_Pruning/tree/main/plots/CIFAR100_resnet50_accuracy.png)
![Robutness](https://github.com/KirstyRD/Logically_Constrained_Pruning/tree/main/plots/CIFAR100_resnet50_robustnessy.png)
![Degeneracy](https://github.com/KirstyRD/Logically_Constrained_Pruning/tree/main/plots/degen_CIFAR100_resnet50.png)

This code was tested with the libraries listed in requirements.txt and CUDA v10.0.

There is a heavy reuse of code from paper references [DL2 [7]](https://github.com/eth-sri/dl2) and [NVIDIA Pruning [18]](https://github.com/NVlabs/Taylor_pruning)
