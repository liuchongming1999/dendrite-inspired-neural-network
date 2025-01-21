import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sys
sys.path.append("..")
import model.simplenet as simplenet
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import random
%matplotlib inline
%config InlineBackend.figure_format = 'svg'

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
  
setup_seed(2)

device = torch.device('cpu')
train_dataset = torchvision.datasets.FashionMNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = torchvision.datasets.FashionMNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=10000)

model = simplenet.SimpleNet_2(num_eigens=1)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
test_accuracy = []
for epoch in range(0, 20):
    lr = 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    model.train()

    # train for one epoch
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        input, target = input, target.long()
        train_total = 0
        train_correct = 0
        train_loss = 0
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
        train_total = target.size(0)
        train_correct = (predicted == target).sum().item()

        prec = train_correct / train_total
        
        if i % 30 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, 20, i, len(train_loader), ave_loss, prec*100))
            
            model.eval()
            valid_correct = 0
            valid_total = 0
            with torch.no_grad():
                total_loss = 0
                for j, (input, target) in enumerate(valid_loader):
                    input, target = input, target.long()
                    output = model(input)
                    
                    _, predicted = torch.max(output.data, 1)
                    valid_total = output.shape[0]
                    valid_correct = (predicted == target).sum().item()
                    prec = valid_correct / valid_total
                    print('Accuary on test images:{:.2f}%, loss:{:.2f}'.format(prec*100,loss))
                    test_accuracy.append(prec)
                    best_prec = max(prec, best_prec)
                           

           
print('Best accuracy is: {:.2f}%'.format(best_prec*100))
