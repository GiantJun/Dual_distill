import torch
from iCIFAR100 import iCIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.utils.data as Data
from myNetwork import network
from ResNet import resnet18
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import numpy as np
import benchmark as benchmark
import torch.nn as nn
import NMS as NMS
import os

logsoftmax=nn.LogSoftmax(dim=1)
softmax=nn.Softmax(dim=1)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
epochs=50


# knowledge distillation的超参数，计算公式查看附件一
T1=4.0
T2=2.5
Lambda = 1.0


learning_rate = 0.1
batchsize = 128

#旧类别个数
old_task_class = 0
#总共的类别数，旧类别的label为0到old_task_class-1，新类别的label为old_task_class到numclass
numclass = 0

#学习阶段数，例如有50个阶段
parser=10
#每次学习的类别数，例如有50个阶段，每个阶段学习2类，即task_size=2
task_size=int(100/parser)


train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=0.24705882352941178),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])


train_dataset = iCIFAR100(os.environ['DATA'], transform=train_transform, download=True)
test_dataset = iCIFAR100(os.environ['DATA'], test_transform=test_transform, train=False, download=True)


# 旧模型的文件名，例如现在是50分类，这个文件存储前一步40分类的模型，即前一增量的结果，存储origin模型
model_name_10 = 'model/0_%d_v1_NMS%d分类_dill_18_20_50.pkl'
# 新模型的文件名，例如现在是50分类，这个文件存储后10分类的模型，存储expert模型
model_name_20 = 'model/%d_%d_18_20_50.pkl'
savefilename = 'model/0_%d_v1_NMS%d分类_dill_18_20_50.pkl'


# 计算knowledge_distillation
def knowledge_distillation(new_output,old_output,T):
    x,y=new_output.to(device),old_output.to(device)
    x = logsoftmax(x / T)
    y = softmax(y / T)
    result = torch.mul(x, y).sum() / x.shape[0]
    return -1 * result.to(device)

# 得到模型的测试准确率
def _test(model, testloader):
    model.eval()
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(testloader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct.item() / total
    model.train()
    return accuracy

# 训练模型
def train():
    opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00001,momentum=0.9,nesterov=True)

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batchsize)

    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=True,
                             batch_size=batchsize)

    for epoch in range(epochs):

        # 变更学习率的代码有问题，但是影响不大，都能完全拟合训练集
        if epoch == 30:
            opt = optim.SGD(model.parameters(), lr=learning_rate /10,momentum=0.9,nesterov=True, weight_decay=0.00001)
            print("变更学习率为%.3f" % (learning_rate / 10))
        elif epoch == 40:
            opt = optim.SGD(model.parameters(), lr=learning_rate  / 100,momentum=0.9,nesterov=True,weight_decay=0.00001)
            print("变更学习率为%.3f" % (learning_rate / 100))


        for step, (indexs, batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # 计算loss
            with torch.no_grad():
                # model_10为origin模型，model_20为expert模型
                x1 = model_10(batch_x)
                x2 = model_20(batch_x)

            # model为new模型
            output = model(batch_x)
            opt.zero_grad()

            #计算交叉熵
            x1_loss=F.cross_entropy(output,batch_y)

            #计算knowledge distillation
            x2_loss=knowledge_distillation(output[...,:x1.shape[1]],x1,T1)
            x3_loss=knowledge_distillation(output[...,x1.shape[1]:],x2,T2)
            loss=x1_loss+Lambda*(x2_loss+x3_loss)
            print("x1_loss:%.3f " % x1_loss.item(), "x2_loss:%.3f " % x2_loss.item(),"x3_loss:%.3f " % x3_loss.item(), end='')
            print(" epoch:%d,step:%d,loss:%.5f" % (epoch, step, loss.item()))
            loss.backward()
            opt.step()

        train_accuracy=_test(model, train_loader)
        accuracy = _test(model, test_loader)
        print('stage 1 test accuracy:', accuracy,' stage 1 train accuracy:',train_accuracy)



# -------------------------------------------------------------------------------------------------------------------------
for i in range(0, parser):
    print("begin train：",i+1)

    feature_extractor=resnet18()
    model=network(task_size,feature_extractor).to(device) if i==0 else torch.load(model_name_10%(i*task_size,i*task_size))
    savefilenames=model_name_10 % (task_size, task_size) if i==0 else model_name_20 % (i * task_size, (i + 1) * task_size)

    # 步骤一训练的轮数以及变更学习率的epoch
    epochs1=100 if i==0 or i==1 else 30
    change1=48
    change2=62

    # 如果i!=0，需要增加全连接层的输出，例如现在是50分类，要把之前全连接层的40分类转变为50分类
    if i!=0 and i!=1:
        #Incremental_learning函数代码位于myNetwork.py
        model.Incremental_learning(task_size)
        model=model.to(device)
        change1=10
        change2=20

    # expert模型训练
    benchmark.train(model,i*task_size,(i+1)*task_size,savefilenames,epochs1,change1,change2)

    if i==0:
        NMS.NMS_classify(model,task_size,task_size)

    # 如果不是第一轮，，进行dual distillation
    if i!=0:

        # origin模型
        model_10 = torch.load(model_name_10 % (i * task_size, i * task_size))
        model_10.eval()

        # expert模型
        model_20 = torch.load(model_name_20 % (i * task_size, (i + 1) * task_size))
        model_20.eval()

        old_task_class = i * task_size
        numclass = (i + 1) * task_size

        # new模型
        model = torch.load(model_name_10 % (i * task_size, i * task_size))
        model.Incremental_learning(numclass)
        model = model.to(device)
        model.train()


        print('exemplar_result大小：',len(NMS.exemplar_result))

        train_dataset.getTrainData([old_task_class, numclass], NMS.exemplar_result)
        test_dataset.getTestData([0, numclass])

        print('begin: ',numclass,' stage 2')
        # dual distillation
        train()
        # NME分类
        NMS.NMS_classify(model,numclass,task_size)
        torch.save(model, savefilename % (numclass, numclass))

print('平均准确率为:',np.mean(np.array(NMS.accuracy_set[1:])))
'''

a=torch.tensor([1.0,2.0,3.0])
b=torch.tensor([0.0,1.0,0.0])
print(knowledge_distillation(a,b,1))
'''