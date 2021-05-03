"""
0;256;0c0;256;0c0;256;0cCopyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from __future__ import print_function
import argparse
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

# ++++++for pruning
import os, sys
import csv
import  time
from utils.utils import save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, load_model_pytorch, dynamic_network_change_local, get_conv_sizes, connect_gates_with_parameters_for_flops
from tensorboardX import SummaryWriter

from logger import Logger
from models.lenet import LeNet
from models.vgg_bn import slimmingvgg as vgg11_bn
from models.preact_resnet import *
from pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list

from utils.group_lasso_optimizer import group_lasso_decay

import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.parallel

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

import numpy as np


#*k DL2 imports
from oracles import DL2_Oracle
from constraints import *
import dl2lib as dl2
import json
from sklearn.metrics import confusion_matrix

import time
import gc
import torchattacks
import gtsrb_dataset as dataset
#++++++++end

# code is based on Pytorch example for imagenet
# https://github.com/pytorch/examples/tree/master/imagenet


def str2bool(v):
    # from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/43357954#43357954
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate(args, test_loader, model, device, criterion, epoch, train_writer=None):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for data_test in test_loader:
            data, target = data_test

            data = data.to(device)

            output = model(data)

            if args.get_inference_time:
                iterations_get_inference_time = 100
                start_get_inference_time = time.time()
                for it in range(iterations_get_inference_time):
                    output = model(data)
                end_get_inference_time = time.time()
#                print("time taken for %d iterations, per-iteration is: "%(iterations_get_inference_time), (end_get_inference_time - start_get_inference_time)*1000.0/float(iterations_get_inference_time), "ms")

            target = target.to(device)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

#    print(' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'.format(top1=top1, top5=top5,batch_time=batch_time, losses = losses) )
    # log to TensorBoard
    if train_writer is not None:
        train_writer.add_scalar('val_loss', losses.avg, epoch)
        train_writer.add_scalar('val_acc', top1.avg, epoch)

    return top1.avg, losses.avg



#*k calculate constraint accuracy
def test_constraints(args, oracle, test_loader_constr, model, device):
    #
    loss = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    test_dl2_loss = 0
    correct, constr, num_steps, pgd_ok = 0, 0, 0, 0

    #count = 0
    for data, target in test_loader_constr:
        #count += 1
        #print("test: ", count)
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)
        n_batch = int(x_batch.size()[0])
        k = n_batch // oracle.constraint.n_tvars

        x_batches, y_batches = [], []
        assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(x_batch[i:(i + k)])
            y_batches.append(y_batch[i:(i + k)])
#        if size0 > 0:
        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=args.restarts, num_iters=args.num_iters, args=args)
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
        else:
            _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

        gc.collect()
        torch.cuda.empty_cache()
        output = model(x_batch)
        test_loss += loss(output, y_batch).item()  # sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        correct += pred.eq(y_batch.view_as(pred)).sum().item()
        constr += constr_acc.item()
        test_dl2_loss += dl2_batch_loss.item()

    test_loss /= len(test_loader_constr.dataset)
#    print('\nTest set: Average loss: {:.4f}, Pred. Accuracy: {}/{} ({:.0f}%)'.format(
#        test_loss, correct, len(test_loader_constr.dataset),
#        100. * correct / len(test_loader_constr.dataset)))
#    print('Constr. acc: %.4f' % (constr / float(num_steps)))

    return correct / len(test_loader_constr.dataset), constr / float(num_steps), test_loss, test_dl2_loss/len(test_loader_constr.dataset)

############### FGSM attack
# https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

# get adversarial accuracy
def fgsm_test( model, device, test_loader, epsilon ):

    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []
    length_testset = 0

    # Loop over all examples in test set
    #count = 0
    for data, target in test_loader:
        #count += 1
        #print("FGSM: ", count)
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True
        # Forward pass the data through the model
        gc.collect()
        torch.cuda.empty_cache()
        output = model(data)
        _, init_pred = torch.max(output, 1)
        #init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        #print(init_pred[1].item())
        #print(target[1].item())
        if init_pred[1].item() != target[1].item():
            continue
        else:
            length_testset = length_testset + 1
        
        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        gc.collect()
        torch.cuda.empty_cache()
        output = model(perturbed_data)

        # Check for success
        _, final_pred = torch.max(output, 1)
        #final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred[1].item() == target[1].item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred[1].item(), final_pred[1].item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred[1].item(), final_pred[1].item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(length_testset)
#    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, length_testset, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples







def RobustnessMany(eps1, eps2, eps3, eps4):
    return lambda model, use_cuda, network_output: ManyEpsilonsRobustnessDatasetConstraint(model, eps1, eps2, eps3, eps4, use_cuda=use_cuda, network_output=network_output)

def RobustnessDandR(eps1, eps2):
    return lambda model, use_cuda, network_output: DivergenceAndRobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)

def RobustnessAND(eps1, eps2):
    return lambda model, use_cuda, network_output: AND_RobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)

def RobustnessT(eps1, eps2):
    return lambda model, use_cuda, network_output: RobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)

def RobustnessG(eps, delta):
    return lambda model, use_cuda, network_output: RobustnessConstraint(model, eps, delta, use_cuda, network_output=network_output)

def RobustnessR(eps1, eps2):
    return lambda model, use_cuda, network_output: RobsRobustnessDatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)

def LipschitzT(L):
    return lambda model, use_cuda, network_output: LipschitzDatasetConstraint(model, L, use_cuda, network_output=network_output)

def LipschitzG(eps, L):
    return lambda model, use_cuda, network_output: LipschitzConstraint(model, eps=eps, l=L, use_cuda=use_cuda, network_output=network_output)

def CSimilarityT(delta):
    return lambda model, use_cuda, network_output: CifarDatasetConstraint(model, delta, use_cuda, network_output=network_output)

def CSimilarityG(eps, delta):
    return lambda model, use_cuda, network_output: CifarConstraint(model, eps, delta, use_cuda, network_output=network_output)

def SegmentG(eps, delta):
    return lambda model, use_cuda, network_output: PairLineRobustnessConstraint(model, eps, delta, use_cuda, network_output=network_output)




def main():
    starttime = time.time()
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    #*k
    parser = dl2.add_default_parser_args(parser)
    #
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--constr-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing constraints (default: 1000)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of GPUs to use')

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--lr-decay-every', type=int, default=100,
                        help='learning rate decay by 10 every X epochs')
    parser.add_argument('--lr-decay-scalar', type=float, default=0.1,
                        help='--')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--run_test', default=False,  type=str2bool, nargs='?',
                        help='run test only')

    parser.add_argument('--limit_training_batches', type=int, default=-1,
                        help='how many batches to do per training, -1 means as many as possible')

    parser.add_argument('--no_grad_clip', default=False,  type=str2bool, nargs='?',
                        help='turn off gradient clipping')

    parser.add_argument('--get_flops', default=False,  type=str2bool, nargs='?',
                        help='add hooks to compute flops')

    parser.add_argument('--get_inference_time', default=False,  type=str2bool, nargs='?',
                        help='runs valid multiple times and reports the result')

    parser.add_argument('--mgpu', default=False,  type=str2bool, nargs='?',
                        help='use data paralization via multiple GPUs')

    parser.add_argument('--dataset', default="MNIST", type=str,
                        help='dataset for experiment, choice: MNIST, CIFAR10', choices= ["MNIST", "CIFAR10", "CIFAR100", "Imagenet", "GTSRB"])

    parser.add_argument('--data', metavar='DIR', default='/imagenet', help='path to imagenet dataset')

    parser.add_argument('--model', default="lenet3", type=str,
                        help='model selection, choices: lenet3, vgg, mobilenetv2, resnet18',
                        choices=["lenet3", "vgg", "mobilenetv2", "resnet18", "resnet152", "resnet50", "resnet50_noskip",
                                 "resnet20", "resnet34", "resnet101", "resnet101_noskip", "densenet201_imagenet",
                                 'densenet121'])

    parser.add_argument('--tensorboard', type=str2bool, nargs='?',
                        help='Log progress to TensorBoard')

    parser.add_argument('--save_models', default=False, type=str2bool, nargs='?',
                        help='if True, models will be saved to the local folder')

    #*k ==========================Constraints
    parser.add_argument('--embed', action='store_true', help='Whether to embed the points.')
    parser.add_argument('--num-iters', type=int, default=10, help='Number of oracle iterations.')
    parser.add_argument('--constraint', type=str, required=True, help='the constraint to train with: LipschitzT(L), LipschitzG(eps, L), RobustnessT(eps1, eps2), RobustnessG(eps, delta), CSimiliarityT(), CSimilarityG(), LineSegmentG(), RobustnessR(eps1, eps2), RobustnessAND(eps1, eps2), RobustnessDandR(eps1, eps2)')
    parser.add_argument('--network-output', type=str, choices=['logits', 'prob', 'logprob'], default='logits', help='Wether to treat the output of the network as logits, probabilities or log(probabilities) in the constraints.')
    parser.add_argument('--batch-size-constr', type=int, default=128, help='Number of samples in a batch.')
    parser.add_argument('--print-after-epoch', type=int, default=0, help='Epoch to start constraint calculations.')
    parser.add_argument('--confusion', default=False, type=str2bool, nargs='?', help='Print confusion matrices?') 
    parser.add_argument('--print-conf-after-epoch', type=int, default=0, help='Epoch to start confusion matrices.')
    # FGSM Accuracy
    parser.add_argument('--attack_method', type=str, default="FGSM")
    parser.add_argument('--adv-after-epoch', type=int, default=0, help='Epoch to start adversarial calculations.')
    # *===========================DL2 training
    
    parser.add_argument('--delay', type=int, default=0, help='How many epochs to wait before training with constraints.')
    parser.add_argument('--print-freq', type=int, default=15, help='Print frequency.')
    parser.add_argument('--dl2-weight', type=float, default=0.0, help='Weight of DL2 loss.')


    # ============================PRUNING added
    parser.add_argument('--pruning_config', default=None, type=str,
                        help='path to pruning configuration file, will overwrite all pruning parameters in arguments')
    
    parser.add_argument('--group_wd_coeff', type=float, default=0.0,
                        help='group weight decay')
    parser.add_argument('--name', default='cuda', type=str,
                        help='experiment name(folder) to store logs')

    parser.add_argument('--augment', default=False, type=str2bool, nargs='?',
                            help='enable or not augmentation of training dataset, only for CIFAR, def False')

    parser.add_argument('--load_model', default='', type=str,
                        help='path to model weights')

    parser.add_argument('--pruning', default=False, type=str2bool, nargs='?',
                        help='enable or not pruning, def False')

    parser.add_argument('--pruning-threshold', '--pt', default=100.0, type=float,
                        help='Max error perc on validation set while pruning (default: 100.0 means always prune)')

    parser.add_argument('--pruning-momentum', default=0.0, type=float,
                        help='Use momentum on criteria between pruning iterations, def 0.0 means no momentum')

    parser.add_argument('--pruning-step', default=15, type=int,
                        help='How often to check loss and do pruning step')

    parser.add_argument('--prune_per_iteration', default=10, type=int,
                        help='How many neurons to remove at each iteration')

    parser.add_argument('--fixed_layer', default=-1, type=int,
                        help='Prune only a given layer with index, use -1 to prune all')

    parser.add_argument('--start_pruning_after_n_iterations', default=0, type=int,
                        help='from which iteration to start pruning')

    parser.add_argument('--maximum_pruning_iterations', default=1e8, type=int,
                        help='maximum pruning iterations')

    parser.add_argument('--starting_neuron', default=0, type=int,
                        help='starting position for oracle pruning')

    parser.add_argument('--prune_neurons_max', default=-1, type=int,
                        help='prune_neurons_max')

    parser.add_argument('--pruning-method', default=0, type=int,
                        help='pruning method to be used, see readme.md')

    parser.add_argument('--pruning_fixed_criteria', default=False, type=str2bool, nargs='?',
                        help='enable or not criteria reevaluation, def False')

    parser.add_argument('--fixed_network', default=False,  type=str2bool, nargs='?',
                        help='fix network for oracle or criteria computation')

    parser.add_argument('--zero_lr_for_epochs', default=-1, type=int,
                        help='Learning rate will be set to 0 for given number of updates')

    parser.add_argument('--dynamic_network', default=False,  type=str2bool, nargs='?',
                        help='Creates a new network graph from pruned model, works with ResNet-101 only')

    parser.add_argument('--use_test_as_train', default=False,  type=str2bool, nargs='?',
                        help='use testing dataset instead of training')

    parser.add_argument('--pruning_mask_from', default='', type=str,
                        help='path to mask file precomputed')

    parser.add_argument('--compute_flops', default=True,  type=str2bool, nargs='?',
                        help='if True, will run dummy inference of batch 1 before training to get conv sizes')

    parser.add_argument('--log_file', default='', type=str,
                        help='Pruning log file (without headers etc.)')
    parser.add_argument('--checkpoint_folder', default='', type=str,
                        help='Folder containing pruned model checkpoints')
    parser.add_argument('--starting_line', default=0, type=int,
                        help='First line in input file to begin validation')
    parser.add_argument('--restarts', default=1, type=int,
                        help='How many times to sample Box')

    # ============================END pruning added

    best_prec1 = 0
    global global_iteration
    global group_wd_optimizer
    global_iteration = 0

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=0)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model == "lenet3":
        model = LeNet(dataset=args.dataset)
    elif args.model == "vgg":
        model = vgg11_bn(pretrained=True)
    elif args.model == "resnet18":
        model = PreActResNet18()
    elif (args.model == "resnet50") or (args.model == "resnet50_noskip"):
        if args.dataset == "CIFAR10":
            model = PreActResNet50(dataset=args.dataset)
        else:
            from models.resnet import resnet50
            skip_gate = True
            if "noskip" in args.model:
                skip_gate = False

            if args.pruning_method not in [22, 40]:
                skip_gate = False
            model = resnet50(skip_gate=skip_gate)
    elif args.model == "resnet34":
        if not (args.dataset == "CIFAR10"):
            from models.resnet import resnet34
            model = resnet34()
    elif "resnet101" in args.model:
        if not (args.dataset == "CIFAR10"):
            from models.resnet import resnet101
            if args.dataset == "Imagenet":
                classes = 1000
            if args.dataset == "CIFAR100":
                classes = 100

            if "noskip" in args.model:
                model = resnet101(num_classes=classes, skip_gate=False)
            else:
                model = resnet101(num_classes=classes)

    elif args.model == "resnet20":
        if args.dataset == "CIFAR10":
            NotImplementedError("resnet20 is not implemented in the current project")
            # from models.resnet_cifar import resnet20
            # model = resnet20()
    elif args.model == "resnet152":
        model = PreActResNet152()
    elif args.model == "densenet201_imagenet":
        from models.densenet_imagenet import DenseNet201
        model = DenseNet201(gate_types=['output_bn'], pretrained=True)
    elif args.model == "densenet121":
        from models.densenet_cifar100 import DenseNet121
        model = DenseNet121(gate_types=['output_bn'], pretrained=True)
    else:
        print(args.model, "model is not supported")

    # dataset loading section
    if args.dataset == "MNIST":
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        #*k constraint arguments
        test_loader_constr = torch.utils.data.DataLoader(
            datasets.MNIST('../data/', train=False, transform=transforms.Compose([
            transforms.ToTensor()])),
            batch_size=args.constr_batch_size, shuffle=True, **kwargs)

        test_loader_adv = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=True)
        

        
    elif args.dataset == "GTSRB":
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        trans = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                                  (0.2724, 0.2608, 0.2669))
        ])

        # Create Datasets                   
        trainset = dataset.GTSRB(
            root_dir='./data', train=True,  transform=trans)
        testset = dataset.GTSRB(
            root_dir='./data', train=False,  transform=trans)

        # Load Datasets                                           
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)
        test_loader_constr = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)
        test_loader_adv = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2)



        
    elif args.dataset == "CIFAR10":
        # Data loading code
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

        kwargs = {'num_workers': 8, 'pin_memory': True}

        test_loader_constr = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=transform_test),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test_loader_adv = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])),
            batch_size=1, shuffle=True)

#    elif args.dataset == "CIFAR100":
#        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}##
#
#        trans = [transforms.ToTensor(),
#                 transforms.Normalize(mean=[n/255.
#                                            for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
#        trans = transforms.Compose(trans)
#        test_set = datasets.CIFAR100('../data', train=False, transform=trans, download=True)##
#
#        test_loader_constr = torch.utils.data.DataLoader(
#            test_set, batch_size=args.batch_size, shuffle=True, **kwargs)#
#
#        test_loader_adv = torch.utils.data.DataLoader(
#            test_set, batch_size=args.batch_size, shuffle=True, **kwargs)


    elif args.dataset == "CIFAR100":
        kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}
        trans = [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[n/255.
                                            for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
        trans = transforms.Compose(trans)
        train_set = datasets.CIFAR100('../data', train=True, transform=trans, download=True)

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs)


        trans = [transforms.ToTensor(),
                 transforms.Normalize(mean=[n/255.
                                            for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]
        trans = transforms.Compose(trans)
        test_set = datasets.CIFAR100('../data', train=False, transform=trans, download=True)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader_constr = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader_adv = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=True, **kwargs)





    elif args.dataset == "Imagenet":

        kwargs = {'num_workers': 16}


        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, **kwargs)

        ####end dataset preparation

#*k constraints
#    constraint = eval(args.constraint)(model, use_cuda, network_output=args.network_output)
#    oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda) 





    # aux function to get size of feature maps
    # First it adds hooks for each conv layer
    # Then runs inference with 1 image
#    output_sizes = get_conv_sizes(args, model)

    if use_cuda and not args.mgpu:
        model = model.to(device)
    elif args.distributed:
        model.cuda()
        print("\n\n WARNING: distributed pruning was not verified and might not work correctly")
        model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.mgpu:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.to(device)

#    print("model is set to device: use_cuda {}, args.mgpu {}, agrs.distributed {}".format(use_cuda, args.mgpu, args.distributed))

    weight_decay = args.wd
    if args.fixed_network:
        weight_decay = 0.0

    # remove updates from gate layers, because we want them to be 0 or 1 constantly
    if 1:
        parameters_for_update = []
        parameters_for_update_named = []
        for name, m in model.named_parameters():
            if "gate" not in name:
                parameters_for_update.append(m)
                parameters_for_update_named.append((name, m))
            #else:
                #print("skipping parameter", name, "shape:", m.shape)

    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
#    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

    optimizer = optim.SGD(parameters_for_update, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay)

#    if 1:
        # helping optimizer to implement group lasso (with very small weight that doesn't affect training)
        # will be used to calculate number of remaining flops and parameters in the network
#        group_wd_optimizer = group_lasso_decay(parameters_for_update, group_lasso_weight=args.group_wd_coeff, named_parameters=parameters_for_update_named, output_sizes=output_sizes)

    cudnn.benchmark = True

    # define objective
    criterion = nn.CrossEntropyLoss()


    # read pruning config file
    pruning_settings = dict()
    if not (args.pruning_config is None):
        pruning_settings_reader = PruningConfigReader()
        pruning_settings_reader.read_config(args.pruning_config)
        pruning_settings = pruning_settings_reader.get_parameters()


    ###=======================added for pruning
    # logging part
    log_save_folder = "%s"%args.name
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)

    if not os.path.exists("%s/models" % (log_save_folder)):
        os.makedirs("%s/models" % (log_save_folder))

    train_writer = None
    if args.tensorboard:
        try:
            # tensorboardX v1.6
            train_writer = SummaryWriter(log_dir="%s"%(log_save_folder))
        except:
            # tensorboardX v1.7
            train_writer = SummaryWriter(logdir="%s"%(log_save_folder))

    time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    run_params = "%s_%s_%d_%d_%s_%s_%s" % (args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), args.batch_size)
    print(run_params)
    textfile = "%s/test_%s.txt" % (log_save_folder, run_params)
    infile = "./process_output/textfiles/log_%s.txt"%(run_params)
    stdout = Logger(textfile)
    sys.stdout = stdout
    print(" ".join(sys.argv))


    ######## go through the log file
    y_params   = [] # number of parameters left
    y_loss     = []
    y_dl2_loss = []
    y_top1     = []
    y_top5     = []
    y_constr   = []
    y_iter     = [] # iteration
    y_time     = [] # time


    with open(args.log_file,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        headers = next(plots)
        for row in plots:
            y_params.append(int(row[1]))  # number of parameters left
            y_loss.append(float(row[2]))
            y_dl2_loss.append(float(row[3]))
            y_top1.append(float(row[4]))
            y_top5.append(float(row[5]))
            y_constr.append(float(row[7]))
            y_iter.append(int(row[8])) # iteration
            y_time.append(float(row[9])) # time

    
    print("print_iter, neuron_units, train_loss, train_dl2_loss, train_top1, train_top5, train_cons, train_time, test_loss, test_dl2_loss, test_acc, test_constr, adv_acc1, adv_acc2, inf_time, test_time")
    #for i in range(len(y_params)):
    for j in range(len(y_params)-args.starting_line):
        i = j + args.starting_line
        #if j % 2 == 0:
            #continue
        checkpoint_file1 = "%s/models/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%.3f000.weights" % (args.checkpoint_folder, args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), y_params[i], y_iter[i], y_top1[i] )
        checkpoint_file2 = "%s/models/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%.3f500.weights" % (args.checkpoint_folder, args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), y_params[i], y_iter[i], y_top1[i] )
        checkpoint_file3 = "%s/models/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%.3f250.weights" % (args.checkpoint_folder, args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), y_params[i], y_iter[i], y_top1[i] )

        lower_top = y_top1[i] - 0.001
        checkpoint_file4 = "%s/models/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%.3f500.weights" % (args.checkpoint_folder, args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), y_params[i], y_iter[i], lower_top )
        checkpoint_file5 = "%s/models/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%.3f750.weights" % (args.checkpoint_folder, args.dataset, args.model, pruning_settings['prune_per_iteration'], pruning_settings['frequency'], args.dl2_weight, args.constraint.replace(" ", "_").replace("(","_").replace(")","_"), y_params[i], y_iter[i], lower_top )


        # loading model file                                                                                                                             
        if os.path.isfile(checkpoint_file1):
            load_model_pytorch(model, checkpoint_file1, args.model)
        elif os.path.isfile(checkpoint_file2):
            load_model_pytorch(model, checkpoint_file2, args.model)
        elif os.path.isfile(checkpoint_file3):
            load_model_pytorch(model, checkpoint_file3, args.model)
        elif os.path.isfile(checkpoint_file4):
            load_model_pytorch(model, checkpoint_file4, args.model)
        elif os.path.isfile(checkpoint_file5):
            load_model_pytorch(model, checkpoint_file5, args.model)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint_file1))
            continue

        model.eval()

        if args.dataset == "CIFAR100":
            label_no = 100
        if args.dataset == "GTSRB":
            label_no = 43

        # array to hold number of inputs corresponding to each label
        num_inputs = [0] * label_no
        
        #num_inputs = torch.zeros(label_no, dtype=torch.int64)
        

        #all_preds = torch.tensor([])
        for data,target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            
            for p in pred:
                num_inputs[p.tolist()] += 1
#            print(num_inputs)

        
        degeneracy = num_inputs.count(0)#torch.count_nonzero(num_inputs)
        
            #num_inputs[] += 1 
            #allpreds = torch.cat((all_preds, output),dim=0)
        #print(all_preds)

#        # adversarial attack
#        if j==0:
#            count = 0
#            adversarial_images = []
#            labels = []
#            for data,target in train_loader:
#                #print(count)
#                count += 1
#                data, target = data_test
#                atk = torchattacks.PGD(model, eps=1/255, alpha=2/255, steps=2)
#                adversarial_images.append(atk(data,target))
#                labels.append(target)
#                
#            # test attack set
#        all_adv = 0
#        correct_adv = 0
#        for adv_count in range(len(adversarial_images)):
#            data = adversarial_images[adv_count]
#            target = labels[adv_count]
#            data, target = data.to(device), target.to(device)
#            
#            gc.collect()
#            torch.cuda.empty_cache()
#            output = model(data)
#            
#            _, pred = torch.max(output, 1)
#            for batch_item in range(len(pred)):
#                all_adv += 1
#                if pred[batch_item].item() == target[batch_item].item():
#                    correct_adv += 1
#            
#        correct_adv = correct_adv / all_adv
            
            
        runtime = time.time() - starttime
        print('{0}, '
              '{1}, '
              '{train_loss:.4f}, '
              '{train_dl2_loss:.4f}, '
              '{train_top1:.3f}, '
              '{train_top5:.3f}, '
              '{train_cons:.3f}, '
              '{train_time}, '
              '{degen}, '
              '{test_time} '
              .format(
                  i, y_params[i] , train_loss=y_loss[i], train_dl2_loss=y_dl2_loss[i],
                  train_top1=y_top1[i], train_top5=y_top5[i], train_cons=y_constr[i], train_time=y_time[i],
                  #precision_dl2, constr_acc, test_loss_dl2
                  #test_loss=test_loss , test_dl2_loss =test_loss_dl2 , test_acc =precision_dl2 , test_constr =constr_acc ,
                  #adv_acc1=accuracies[0], adv_acc2=accuracies[1], #adv_acc3=accuracies[2],
                  #adv_acc4=accuracies[3], adv_acc5=accuracies[4], adv_acc6=accuracies[5],
                  degen=degeneracy , test_time=runtime
              ))
        
        
if __name__ == '__main__':
    main()
                                    
