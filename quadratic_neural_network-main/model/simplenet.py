from audioop import bias
import torch.nn as nn
import torch
import model.Linear_quadratic as Lq
#import model.Low_rank_conv2d as Lrc

'''
modified to fit dataset size
'''
NUM_CLASSES = 10

#Regression
class RegressionNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_eigens=1, EI_distribution=torch.zeros(4)):
        super(RegressionNet, self).__init__()
        self.classifier = nn.Sequential(
            #nn.Linear(512, 10),
            #nn.Linear(1,2000),
            #nn.Sigmoid(),
            #nn.ReLU(),
            #nn.Linear(2000,1),
            Lq.General_quadratic(1,5, bias=True),
            Lq.General_quadratic(5,5, bias=True),
            #Lq.General_quadratic(5,5, bias=True),
            #Lq.General_quadratic(5,5, bias=True),
            #Lq.General_quadratic(5,5, bias=True),
            Lq.General_quadratic(5,1, bias=True),
            #Lq.Low_dimensional_quadratic(512,10,num_eigens,bias=False),
            #Lq.Dales_General_quadratic(512,10,EI_distribution ,bias=False),
            #nn.ReLU(inplace = True),
            #nn.Linear(1, 1),
        )

    def forward(self, x):                             
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ONLY Linear layer
class SimpleNet_0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_eigens=1, EI_distribution=torch.zeros(4)):
        super(SimpleNet_0, self).__init__()
        self.classifier = nn.Sequential(
            #nn.Linear(512, 10),
            Lq.General_quadratic(512,10, bias=False),
            #Lq.Low_dimensional_quadratic(512,10,num_eigens,bias=False),
            #Lq.Dales_General_quadratic(512,10,EI_distribution ,bias=False),
            #nn.ReLU(inplace = True),
            #nn.Linear(1, 1),
        )

    def forward(self, x):                             
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2, num_eigens=1):
        super(SimpleNet, self).__init__()
        self.classifier = nn.Sequential(
            #nn.Linear(28*56, num_classes, bias=False),
            #Lq.Linear_quadratic(28*28, num_classes),
            #Lq.General_quadratic(28*56, num_classes, bias=False),
            #nn.ReLU(inplace = True),
            #nn.Linear(28*28, num_classes),
            Lq.Low_dimensional_quadratic(28*56, num_classes, num_eigens,bias=False),
            #nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
# ONLY quadratic Linear layer
class SimpleNet_1(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_eigens=1, EI_distribution=torch.zeros(784)):
        super(SimpleNet_1, self).__init__()
        self.classifier = nn.Sequential(
            #nn.Linear(784, 7500),
            #nn.ReLU(inplace = True),
            #nn.Linear(7500, num_classes),
            Lq.Low_dimensional_quadratic(28*28, num_classes, num_eigens,bias=False),
            #Lq.General_quadratic(28*28, num_classes, bias=False)
            #Lq.Dales_General_quadratic(28*28,num_classes,EI_distribution ,bias=False)
            #nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# One convolution layer
class SimpleNet_2(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_eigens=1, EI_distribution=torch.zeros(784)):
        super(SimpleNet_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=5, padding=2, stride=1),
            #Lrc.Conv2d_lowrank_quadratic(1, 1, kernel_size=5, padding=2, number_of_eigens=5)
            nn.ReLU (inplace=True),
            #nn.MaxPool2d(kernel_size=2),   
            #nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            #nn.Linear(28*28, num_classes),
            #Lq.General_quadratic(28*28, num_classes, bias=False),
            Lq.Low_dimensional_quadratic(28*28, num_classes, num_eigens,bias=False),
            #Lq.Dales_General_quadratic(28*28,num_classes,EI_distribution ,bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleNet_3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, num_eigens=1):
        super(SimpleNet_3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=4, padding=0, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            #Lrc.Conv2d_lowrank_quadratic(1, 1, kernel_size=5, padding=2, number_of_eigens=5)
            #nn.ReLU (inplace=True),4
            #nn.MaxPool2d(kernel_size=2),
            #nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*2*2, 2),
            #Lq.General_quadratic(1*28*28, num_classes),
            #nn.Linear(2, num_classes, bias=False),
            #Lq.General_quadratic(2, num_classes, bias=False),
            Lq.Low_dimensional_quadratic(2, num_classes, num_eigens, bias=False),
            #nn.ReLU(inplace = True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
