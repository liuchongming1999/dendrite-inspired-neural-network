import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")
import model.resnet_cifar10 as ResNet
import argparse
import matplotlib.pyplot as plt
import os
import imageio
import shutil
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import random

def savefig(name):
    plt.savefig(name,dpi=600, bbox_inches='tight')
    return

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.benchmark = True #for accelerating the running
    return

setup_seed(1)

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--gpu-id', default=[0], nargs='+', type=int, help='available GPU IDs')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 128),only used for train')
parser.add_argument('-w', '--workers', default=5, type=int, metavar='N', help='num_workers, at most 16, must be 0 on windows')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=60, type=int, metavar='N', help='print frequency (default: 10)')
#parser.add_argument('--resume', default='checkpoint/Alexnet/checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-t', '--train', dest='train', action='store_true', help='test model on test set')

args = parser.parse_known_args()[0]

Path_Name = 'resnet'
checkpoint_path = 'checkpoint/' + Path_Name
summary_path = 'summary/' + Path_Name
if args.train:
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    writer = SummaryWriter(summary_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])



# CIFAR10 dataset 
train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=train_transform, download=True)
valid_dataset = torchvision.datasets.CIFAR10(root='data', train=False, transform=test_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=10000)

model = ResNet.resnet18()
model = nn.DataParallel(model, device_ids=args.gpu_id).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
cudnn.benchmark = True
test_accuracy = []
best_prec = 0

if args.resume:
    if os.path.isfile(args.resume):
        print('=> loading checkpoint "{}"'.format(args.resume))
        checkpoint = torch.load(args.resume)
        #args.start_epoch = checkpoint['epoch']
        #best_prec = checkpoint['best_acc']
        checkpoint['state_dict']['module.classifier.bias_r'] = torch.arange(10)
        checkpoint['state_dict']['module.classifier.weight_a'] = torch.zeros(10,512,512)
        #checkpoint['state_dict']['module.classifier.bias_r'] = torch.arange(10)
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {} best_acc {})".format(args.resume, checkpoint['epoch'], best_prec))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model.module.linear)     






for epoch in range(args.start_epoch, args.epochs):
    if epoch < 150:
        lr = args.lr
    else:
        lr = args.lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()
    train_total = 0
    train_correct = 0
    train_loss = 0
    # train for one epoch
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input, target = input.cuda(), target.long().cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        ave_loss = train_loss/(i+1)

        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()

        prec = train_correct / train_total
        if (i+1) % args.print_freq == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, args.epochs, i+1, len(train_loader), loss, prec*100))


    # evaluate on test set
    # switch to evaluate mode
    model.eval()
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_loader):
            input, target = input.cuda(), target.long().cuda()

            output = model(input)

            _, predicted = torch.max(output.data, 1)
            valid_total = output.shape[0]
            valid_correct = (predicted == target).sum().item()
            prec = valid_correct / valid_total
            test_accuracy.append(prec)
            print('Accuary on test images:{:.2f}%'.format(prec*100))
            best_prec = max(prec, best_prec)

        print('Best accuracy: {:.2f}%'.format(best_prec*100))

    
data=open("weights0.txt",'w') 
for name, param in model.named_parameters():
    print(name, param, file=data)
data.close()
