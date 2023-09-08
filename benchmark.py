import torch
from torchvision import transforms
from torch.nn import functional as F
import torch.optim as optim
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
''''''
batch_size=128
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([transforms.RandomCrop((32,32),padding=4),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.ColorJitter(brightness=0.24705882352941178),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

train_dataset = iCIFAR100('dataset', transform=train_transform, download=True)
test_dataset = iCIFAR100('dataset', test_transform=test_transform, train=False, download=True)

#得到one_hot编码，用于计算loss
def get_one_hot(target,numclass):
    one_hot=torch.zeros(target.shape[0],numclass).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

def _test(testloader,beginclass,model):
    model.eval()
    correct, total = 0, 0
    for setp, (indexs, imgs, labels) in enumerate(testloader):
        labels=labels-beginclass
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += len(labels)
    accuracy = 100 * correct / total
    model.train()
    return accuracy

'''
beginclass:训练数据的起始label
endclass:训练数据的结束label
训练label从beginclass到endclass的类别

savefilename:存储模型的文件名

epoch1:训练总轮数
change1：改变学习率的训练轮数
change2：改变学习率的训练轮数
'''
def train(model,beginclass,endclass,savefilename,epochs,change1,change2):

    #得到训练数据
    train_dataset.getTrainData([beginclass,endclass], [])

    #得到测试数据
    test_dataset.getTestData([beginclass,endclass])

    #
    numclass=endclass-beginclass
    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size)

    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=True,
                             batch_size=batch_size)
    if epochs>=70:
        learning_rate=1.0
        opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    else:
        learning_rate=1.0
        opt = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.00001,momentum=0.9,nesterov=True)
    for epoch in range(epochs):

        #更改学习率的方式存在问题，但是不影响训练集上的拟合
        if epoch == change1:
            if epochs==70:
               opt = optim.SGD(model.parameters(), lr=learning_rate/5, weight_decay=0.00001)
            else:
               opt = optim.SGD(model.parameters(), lr=learning_rate / 5, weight_decay=0.00001,momentum=0.9,nesterov=True)
            print("变更学习率为%.3f" % (learning_rate / 5))
        elif epoch == change2:
            if epochs==70:
               opt = optim.SGD(model.parameters(), lr=learning_rate/25, weight_decay=0.00001)
            else:
               opt = optim.SGD(model.parameters(), lr=learning_rate / 25, weight_decay=0.00001,momentum=0.9,nesterov=True)
            print("变更学习率为%.3f" % (learning_rate / 25))
        elif epoch==80:
             opt = optim.SGD(model.parameters(), lr=learning_rate/125, weight_decay=0.00001)

        #常规的模型训练与测试
        for step, (indexs, images, target) in enumerate(train_loader):
            target=target-beginclass
            images, target = images.to(device), target.to(device)
            target=get_one_hot(target,numclass)
            output = model(images)
            loss_value = F.binary_cross_entropy_with_logits(output, target)
            opt.zero_grad()
            loss_value.backward()
            opt.step()
            print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
        accuracy = _test(test_loader,beginclass,model)
        print('benchmark epoch:%d,accuracy:%.3f' % (epoch, accuracy))
    torch.save(model,savefilename)

