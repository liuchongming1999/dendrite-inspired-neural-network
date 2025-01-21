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
import matplotlib as mpl
import numpy as np
import torch.nn.functional as F
import random
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelweight'] = 'bold'

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
device = torch.device('cpu')
# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=10000)

num_eigen = 1
model = simplenet.SimpleNet_1(num_eigens = num_eigen)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
test_accuracy = []

for epoch in range(0, 10):

    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01

    model.train()
    # train for one epoch
    for i, (input, target) in enumerate(train_loader):
        train_total = 0
        train_correct = 0
        train_loss = 0
        # measure data loading time
        input, target = input, target.long()
                    
        
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        
        
        # compute training accuracy
        _, predicted = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += (predicted == target).sum().item()
        prec = train_correct / train_total

          if (i) % 20 == 0:
              train_accuracy.append(prec)
              print('Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}, Train_Acc:{:.2f}%'.format(epoch+1, 20, i, len(train_loader), loss, prec*100))

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
                      loss = criterion(output, target)

                      prec = valid_correct / valid_total
                      print('Accuary on test images:{:.2f}%, loss:{:.5f}'.format(prec*100, loss))
                      test_accuracy.append(prec)
                      best_prec = max(prec, best_prec)


print('Best accuracy is: {:.2f}%, Minimum loss is: {:.4f}'.format(best_prec*100, min_loss))
