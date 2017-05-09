# coding=utf-8
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import scipy.io as scio
import numpy as np
import random
import math


# 处理输入数据
with open('Radio_map_3D.mat','rb') as fileholder:
    trainset = scio.loadmat(fileholder)
    v = np.array(trainset['Radio_map_3D'])
print('Data read complete')


# 生成训练样本集
train_target = []
trainset = []
# 将250*250*169的tensor转换成62500*169的tensor
for i,row in enumerate(v,0):
    for j,vector in enumerate(row,0):
        target  = (i/5)*50 + (j/5)
        train_target.append(target)
        trainset.append([vector])

trainset = torch.FloatTensor(trainset)
train_target = torch.LongTensor(train_target)
trainloader = torch.utils.data.TensorDataset(trainset,train_target)
trainloader = torch.utils.data.DataLoader(trainloader, batch_size=4,shuffle=True, num_workers=4)
print('Train set complete')

# 准备测试集
testset = []
test_target = []
for rand in range(100):
    index_i = random.randint(0,249)
    index_j = random.randint(0,249)
    testdata = []
    for item in v[index_i][index_j]:
        testdata.append(item + random.uniform(-10,10))
    target = (index_i/5)*50+(index_j/5)
    test_target.append(target)
    testset.append([testdata])

testset = torch.FloatTensor(testset)
test_target = torch.LongTensor(test_target)
testloader = torch.utils.data.TensorDataset(testset,test_target)
testloader = torch.utils.data.DataLoader(testloader, batch_size=4,shuffle=True, num_workers=4)
print('Test set complete')


# 定义网络的结构
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(1,6,10,stride=1)
        self.pool = nn.MaxPool1d(4)
        self.linear = nn.Linear(240,2500)

    def forward(self, x):
        x = self.conv1(x)
        #print('Conv1:\n')
        #print(x)
        x = self.pool(x)
        #print('Pool:\n')
        #print(x)
        x = x.view(-1, 240)
        #print('View:\n')
        #print(x)
        x = self.linear(x)
        return x


# 生成网络
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)


# 利用训练集训练网络
# 并画出loss图

x_axis = []
y_axis = []
for epoch in range(4):
    running_loss = 0.0
    for index, data in enumerate(trainloader,0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        output = net(inputs)
        optimizer.zero_grad()
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        #print loss.data
        if index % 10 == 1:
            x_axis.append(epoch*16000 + index)
            y_axis.append(running_loss/10)
            #print('epoch ' + str(epoch) + ' index ' + str(index) + ': ')
            #print(running_loss/10)
            running_loss = 0.0
        if index % 1000 == 0:
            print index
print('\nTrain Finished!')
plt.figure(1)
plt.ylim(0.0, 15.0)
plt.xlabel('Training Times')
plt.ylabel('Cross Entropy Loss')
plt.plot(x_axis,y_axis)
plt.show()

# 利用测试集测试数据
# 画法1
'''
correct = 0
total = 0
ave_loss = 0
test_x1_axis = []
test_y1_axis = []
test_x2_axis = []
test_y2_axis = []
for index, data in enumerate(testloader,0):
    test_input, test_labels = data
    test_output = net(Variable(test_input))
    _, predicted = torch.max(test_output.data, 1)
    for i in range(4):
        if(test_labels[i] == predicted[i][0]):
            correct += 1
        x1 = predicted[i][0] % 50
        y1 = predicted[i][0] / 50
        x2 = test_labels[i] % 50
        y2 = test_labels[i] / 50
        test_x1_axis.append(x1)
        test_y1_axis.append(y1)
        test_x2_axis.append(x2)
        test_y2_axis.append(y2)
        ave_loss += math.sqrt((x1-x2)**2 + (y1-y2)**2)
        total += 1
plt.figure(2)
plt.plot(test_x1_axis,test_y1_axis,'go')
plt.plot(test_x2_axis,test_y2_axis,'ro')
plt.show()
print('Test Finished!')
print ('Accurce :' + str(100 * correct/total) + '%')
print ('Loss: ' + str(ave_loss/total))
'''
# 画法2
correct = 0
total = 0
ave_loss = 0
#plt.figure(2)
for index, data in enumerate(testloader,0):
    test_input, test_labels = data
    test_output = net(Variable(test_input))
    _, predicted = torch.max(test_output.data, 1)
    for i in range(4):
        if(test_labels[i] == predicted[i][0]):
            correct += 1
        x1 = predicted[i][0] % 50
        y1 = predicted[i][0] / 50
        x2 = test_labels[i] % 50
        y2 = test_labels[i] / 50
        plt.plot([x1, x2], [y1, y2], '-x')
        ave_loss += math.sqrt((x1-x2)**2 + (y1-y2)**2)
        total += 1

plt.figure(2)
plt.grid(True, which='major')
plt.show()
print('Test Finished!')
print ('Accurce :' + str(100 * correct/total) + '%')
print ('Loss: ' + str(ave_loss/total))












