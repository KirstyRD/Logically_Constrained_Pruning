#!/bin/bash


# Accuracy
python3 plot_3_any.py text/run2_CIFAR100_vgg_250_100_0.0_FGSMconstr_64.txt testrun/log_CIFAR100_vgg_250_100_0.05_FGSM_64.txt text/run2_CIFAR100_vgg_250_100_0.1_FGSMconstr_64.txt run2_CIFAR100_vgg_250_100_vary_FGSMconstr_64_pred.png "Prediction Accuracy CIFAR100 with VGG" "Prediction Accuracy" 4 "Baseline" "weight 0.05" "weight 0.1"
python plot_3_any.py text/GTSRB_lenet3_5_50_0.0_FGSMconstr_128.txt testrun/log_GTSRB_lenet3_5_50_0.05_FGSM_128.txt text/GTSRB_lenet3_5_50_0.1_FGSMconstr_128.txt GTSRB_lenet3_5_50_0.1_FGSMconstr_pred.png "Prediction Accuracy - GTSRB with LeNet" "Prediction accuracy" 4 "Baseline" "weight 0.05" "weight 0.1"
python plot_3_any.py text/CIFAR100_resnet50_200_100_0.0_FGSMconstr_64.txt testrun/log_CIFAR100_resnet50_200_100_0.01_FGSM_64.txt text/CIFAR100_resnet50_200_100_0.05_FGSMconstr_64.txt CIFAR100_resnet50_200_100_vary_FGSMconstr_64_pred.png "Prediction Accuracy - CIFAR100 with ResNet" "Prediction accuracy" 4 "Baseline" "weight 0.05" "weight 0.1"

# Robustness
python3 plot_3_any.py text/run2_CIFAR100_vgg_250_100_0.0_FGSMconstr_64.txt testrun/log_CIFAR100_vgg_250_100_0.05_FGSM_64.txt text/run2_CIFAR100_vgg_250_100_0.1_FGSMconstr_64.txt run2_CIFAR100_vgg_250_100_vary_FGSMconstr_64_fgsm.png "Robustness - CIFAR100 with VGG" "Robustness" 7 "Baseline" "weight 0.05" "weight 0.1"
python plot_3_any.py text/GTSRB_lenet3_5_50_0.0_FGSMconstr_128.txt testrun/log_GTSRB_lenet3_5_50_0.05_FGSM_128.txt text/GTSRB_lenet3_5_50_0.1_FGSMconstr_128.txt GTSRB_lenet3_5_50_0.1_FGSMconstr_fgsm.png "Robustness - GTSRB with LeNet" "Robustness" 7 "Baseline" "weight 0.05" "weight 0.1"
python plot_3_any.py text/CIFAR100_resnet50_200_100_0.0_FGSMconstr_64.txt testrun/log_CIFAR100_resnet50_200_100_0.01_FGSM_64.txt text/CIFAR100_resnet50_200_100_0.05_FGSMconstr_64.txt CIFAR100_resnet50_200_100_vary_FGSMconstr_64_fgsm.png "Robustness - CIFAR100 with ResNet" "Robustness" 7 "Baseline" "weight 0.05" "weight 0.1"

# Degeneration 
python plot_3_degen.py text/degen_CIFAR100_densenet121_200_100_0.0_FGSMconstr_32.txt testrun/test_CIFAR100_densenet121_200_100_0.05_FGSM_32.txt text/degen_CIFAR100_densenet121_200_100_0.1_FGSMconstr_32.txt degen_CIFAR100_densenet121_200_100_vary_FGSMconstr_32.png "Deterioration of VGG on CIFAR100" "Percentage of Accecible Classes" 8 "Baseline" "weight 0.05" "weight 0.1" 100
python plot_3_degen.py text/degen_GTSRB_lenet3_5_50_0.0_FGSMconstr_128.txt testrun/test_GTSRB_lenet3_5_50_0.05_FGSM_128.txt text/degen_GTSRB_lenet3_5_50_0.1_FGSMconstr_128.txt degen_GTSRB_lenet3_5_50_vary_FGSMconstr_128.png "Deterioration of LeNet on GTSRB" "Percentage of Accecible Classes" 8 "Baseline" "weight 0.05" "weight 0.1" 43
python plot_3_degen.py text/degen_CIFAR100_resnet50_200_100_0.0_FGSMconstr_64.txt testrun/test_CIFAR100_resnet50_200_100_0.01_FGSM_64.txt text/degen_CIFAR100_resnet50_200_100_0.05_FGSMconstr_64.txt degen_CIFAR100_resnet50_200_100_vary_FGSMconstr_64_fgsm.png "Deterioration of ResNet on CIFAR100" "Percentage of Accecible Classes" 8 "Baseline" "weight 0.05" "weight 0.1" 100
