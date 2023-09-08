from torchvision.datasets import CIFAR100
import numpy as np
from PIL import Image
import random


'''
需重点检查
1、getTestData函数
2、getTrainData函数
3、get_image_class函数
'''
class iCIFAR100(CIFAR100):

    #train参数决定是load训练集数据还是测试集数据
    def __init__(self,root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR100,self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)


        self.target_test_transform=target_test_transform
        self.test_transform=test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        #self.dataIndex=[i for i in range(100)]

        #打乱次序
        np.random.seed(32)
        self.dataIndex=np.random.permutation(100)
        print(self.dataIndex)

    #辅助函数，写啰嗦了，用于统一纬度，例如原纬度为（4，500，32，32），变为（2000，32，32）
    def concatenate(self,datas,labels):
        con_data=datas[0]
        con_label=labels[0]
        for i in range(1,len(datas)):
            con_data=np.concatenate((con_data,datas[i]),axis=0)
            con_label=np.concatenate((con_label,labels[i]),axis=0)
        return con_data,con_label

    #获得label为classes[0]到classes[1]的测试类别数据
    def getTestData(self, classes):
        datas,labels=[],[]
        for index in range(classes[0], classes[1]):
            #数据打乱了，需要获得原始数据的label
            label=self.dataIndex[index]
            print('get class %d test data' % (index), ' index is ', label)
            #获得label对应的数据
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            #重新分配label
            labels.append(np.full((data.shape[0]), index))
        datas,labels=self.concatenate(datas,labels)
        self.TestData,self.TestLabels=datas,labels
        print("the size of test set is %s"%(str(self.TestData.shape)))
        print("the size of test label is %s"%str(self.TestLabels.shape))


    # 获得label为classes[0]到classes[1]的训练类别数据，同时合并examplar的数据，得到最终的训练数据
    def getTrainData(self,classes,exemplar_set):
        datas,labels=[],[]

        # 合并examplar的数据，写啰嗦了
        if len(exemplar_set)!=0:
            datas=[exemplar for exemplar in exemplar_set ]
            length=len(datas[0])
            labels=[np.full((length),label) for label in range(len(exemplar_set))]
        if len(classes)!=0:

            # 获得label为classes[0]到classes[1]的训练类别数据
            for index in range(classes[0],classes[1]):
                #数据打乱了，需要获得原始数据的label
                label=self.dataIndex[index]
                print('get class %d train data' % (index), ' index is ', index)
                # 获得label对应的数据
                data=self.data[np.array(self.targets)==label]
                datas.append(data)
                # 重新分配label
                labels.append(np.full((data.shape[0]),index))
        self.TrainData,self.TrainLabels=self.concatenate(datas,labels)
        print("the size of train set is %s"%(str(self.TrainData.shape)))
        print("the size of train label is %s"%str(self.TrainLabels.shape))


    # 获得index对应的训练数据，并应用transform
    def getTrainItem(self,index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]

        if self.transform:
            img=self.transform(img)

        if self.target_transform:
            target=self.target_transform(target)

        return index,img,target

    # 获得index对应的测试数据，并应用transform
    def getTestItem(self,index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]

        if self.test_transform:
            img=self.test_transform(img)

        if self.target_test_transform:
            target=self.target_test_transform(target)

        return index, img, target

    # dataloader调用该函数获得数据
    def __getitem__(self, index):
        if self.TrainData!=[]:
            return self.getTrainItem(index)
        elif self.TestData!=[]:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData!=[]:
            return len(self.TrainData)
        elif self.TestData!=[]:
            return len(self.TestData)

    # 依据label获得对应的数据，用于iCaRL分类器
    def get_image_class(self,label):
        return self.data[np.array(self.targets)==self.dataIndex[label]]


