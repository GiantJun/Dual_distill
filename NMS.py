import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from iCIFAR100 import iCIFAR100

exemplar_result=[]
accuracy_set=[]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

test_dataset = iCIFAR100('dataset', test_transform=transform, train=False, download=True)
train_dataset = iCIFAR100('dataset', transform=transform, download=True)
class_mean_set_all={}
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

# 辅助函数，对images图像应用tansform，transform似乎只能一张图片一张图片处理，不支持多张图片同时处理
def Image_transform(images, transform):
    data = transform(Image.fromarray(images[0])).unsqueeze(0)
    for index in range(1, len(images)):
        data = torch.cat((data, transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
    return data

# 计算类别中心
def compute_class_mean(images,transform,model):
    x = Image_transform(images, transform).to(device)
    feature_extractor_output = F.normalize(model.feature_extractor(x).detach()).cpu().numpy()
    class_mean = np.mean(feature_extractor_output, axis=0)
    return class_mean, feature_extractor_output

# 构建新类别的examplar_set
def _construct_exemplar_set(images, m,model):
    class_mean, feature_extractor_output = compute_class_mean(images, transform,model)
    exemplar_all=[]
    now_class_mean = np.zeros((1, feature_extractor_output.shape[1]))
    for i in range(m):
        # shape：batch_size*512
        x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
        # shape：batch_size
        x = np.linalg.norm(x, axis=1)
        index = np.argmin(x)
        now_class_mean += feature_extractor_output[index]
        exemplar_all.append(images[index])
    print('exemplar_all的大小为:%s'%(str(len(exemplar_all))))

    #存入examplar
    exemplar_result.append(exemplar_all)

# 降低旧类别的examplar_set
def _reduce_exemplar_sets(m):
    for index in range(len(exemplar_result)):
        exemplar_result[index] = exemplar_result[index][:m]

        #验证examplar的大小
        print('第%d类exemplar的大小为%s' % (index, str(len(exemplar_result[index]))))

# 计算examplar_set的类别中心，用于分类
def compute_exemplar_class_mean(model):
    class_mean_set = []
    for index in range(len(exemplar_result)):
        #print用于验证examplar的大小是正确的
        print("计算第%s类的类别中心" % (str(index)),' 大小：',len(exemplar_result[index]))

        exemplar = exemplar_result[index]

        #计算类别中心
        class_mean, _ = compute_class_mean(exemplar, transform,model)
        class_mean_, _ = compute_class_mean(exemplar, classify_transform,model)
        class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2


        class_mean_set.append(class_mean)

    return np.array(class_mean_set)

'''
numclass：表示总类别个数。例如现在是50分类，numclass的值为50
task_size：表示每次增量的新类别数，例如10阶段学习，则task_size为10
'''
def NMS_classify(model,numclass,task_size):
    model.eval()
    #首先降低每个examplar的图片数
    _reduce_exemplar_sets(int(2000/numclass))

    #构建examplar
    for i in range(numclass-task_size,numclass):
        print("construct class %d examplar"%(i),end='')
        images=train_dataset.get_image_class(i)
        _construct_exemplar_set(images,int(2000/numclass),model)

    #计算所有类的class mean
    class_mean_set=compute_exemplar_class_mean(model) 


    print(class_mean_set.shape)

    for i in range(class_mean_set.shape[0]):
        # class_mean_set_all包含了所有旧类别
        if i not in class_mean_set_all.keys():
            #对于新类别，会利用该类的数据计算一份类别平均
            print("新类别：",i)
            exemplar=train_dataset.get_image_class(i)
            class_mean, _ = compute_class_mean(exemplar, transform,model)
            class_mean_, _ = compute_class_mean(exemplar, classify_transform,model)
            class_mean = (class_mean / np.linalg.norm(class_mean) + class_mean_ / np.linalg.norm(class_mean_)) / 2
            class_mean_set_all[i]=[class_mean]
        #如果不是新类别，class_mean_set_all存储旧的ensemble class mean，class_mean_set[i]即new class mean
        class_mean_set_all[i].append(class_mean_set[i])

    # 计算ensemble class mean
    class_mean_set=[]
    for i in class_mean_set_all.keys():
        print("计算类别平均:",i,' 大小为:',len(class_mean_set_all[i]),' 第%d次任务'%(int(i/task_size)),'乘数为：',(int(numclass/task_size)-int(i/task_size)))

        # int(numclass/task_size)-int(i/task_size)即"模型结构"文件中的index参数
        temp=(class_mean_set_all[i][0]*(int(numclass/task_size)-int(i/task_size))+class_mean_set_all[i][1])/(int(numclass/task_size)-int(i/task_size)+1)

        # temp即为ensemble class mean
        class_mean_set_all[i]=[temp]
        class_mean_set.append(temp)

    for i in class_mean_set_all.keys():
        print('第%d类的类别平均大小为%d'%(i,len(class_mean_set_all[i])))
    class_mean_set=np.array(class_mean_set)
       
    print(class_mean_set.shape)
    correct=0
    num=0

    # 用于存储每个task的准确率
    every_class_accuracy_set={}
    for i in range(0,int(numclass/task_size)):
        every_class_accuracy_set[i]=[]

    # 利用iCaRL的分类器计算准确率
    for i in range(0, numclass):
        every_class_accuracy=0
        imgs = test_dataset.get_image_class(i)
        print('compute accuracy:', i, ":", imgs.shape[0])
        num += imgs.shape[0]
        imgs = Image_transform(imgs, transform).to(device)
        with torch.no_grad():
            temp = F.normalize(model.feature_extractor(imgs)).cpu().detach().numpy()

        #利用iCaRL的分类器，得到分类结果
        for target in temp:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            if x == i:
                every_class_accuracy+=1
                correct += 1
        every_class_accuracy_set[int(i/task_size)].append(every_class_accuracy)
    print('every class accuracy:')
    print(correct)

    #输出每个任务的准确率
    for key in  every_class_accuracy_set.keys():
        print(np.sum(np.array(every_class_accuracy_set[key])))
        print(key,':',np.sum(np.array(every_class_accuracy_set[key]))/(len(every_class_accuracy_set[key])*100))

    #输出总准确率
    print(numclass,' NMS accuracy:',correct/num)
    accuracy_set.append(correct/num)
    print()
    model.train()
'''
for i in range(1,21):
        numclass=i*5
        model=torch.load('model/0_%d_v1_NMS%d分类_dill_18_20_normal.pkl'%(numclass,numclass),map_location='cpu')
        model=model.to(device)
        NMS_classify(model,numclass,5)
print(np.mean(np.array(accuracy_set[1:])))
'''
