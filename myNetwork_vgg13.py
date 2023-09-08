import torch.nn as nn
import torch.nn.functional as F


class network(nn.Module):

    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        # 特征提取器
        self.feature = feature_extractor
        # 全连接层
        self.fc1 = nn.Linear(512, 4096)
        #self.bn1=nn.BatchNorm1d(4096)
        self.relu1=nn.ReLU(inplace=True)
        self.fc2=nn.Linear(4096, 4096)
        #self.bn2 = nn.BatchNorm1d(4096)
        self.relu2=nn.ReLU(inplace=True)
        self.fc3=nn.Linear(4096, numclass)
        '''
        self.classifier = nn.Sequential(
                            nn.Linear(512, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(),
                            nn.Linear(4096, numclass)
                          )
        '''
    def forward(self, input):
        x = self.feature(input)
        
        x = self.fc1(x)
        #x=self.bn1(x)
        x=self.relu1(x)
        x=F.dropout(x)
        x=self.fc2(x)
        #x=self.bn2(x)
        x=self.relu2(x)
        x=F.dropout(x)
        x=self.fc3(x)
        
        #x=self.classifier(x)
        return x

    def feature_extractor(self, inputs):
        return self.feature(inputs)

    def Incremental_learning(self, numclass):
        weight = self.fc3.weight.data
        bias = self.fc3.bias.data
        in_feature = self.fc3.in_features
        out_feature = self.fc3.out_features

        self.fc3 = nn.Linear(in_feature, numclass, bias=True)
        if numclass > out_feature:
            self.fc3.weight.data[:out_feature] = weight
            self.fc3.bias.data[:out_feature] = bias

'''
from vgg13 import  vgg13_bn
import torch
feature_exactor=vgg13_bn()
net=network(10,feature_exactor)
print(net(torch.randn((1,3,32,32))))
net.Incremental_learning(20)
print(net(torch.randn((1,3,32,32))))
'''
