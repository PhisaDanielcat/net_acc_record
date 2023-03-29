import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np
import math
# from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch.nn.init as init
import datetime
import copy
# 定义超参数
num_epochs = 5  #训练的总循环周期
batch_size = 2048  #一个撮（批次）的大小，64张图片

# 训练集
train_dataset = datasets.CIFAR10(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 测试集
test_dataset = datasets.CIFAR10(root='./data',
                           train=False,
                           transform=transforms.ToTensor())
# 构建batch数据
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size)
def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(3*32*32, 256, bias=True),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256,10,bias=True),
            nn.ReLU()
        )

        # init.xavier_uniform_(self.fc1.Linear.weight)
        # init.xavier_uniform_(self.fc2.Linear.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

lr = 1
device = torch.device("cuda:0")
model = My_model()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum=0.9)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

train_acc_list=[]
valid_acc_list=[]
loss_list=[]
log_infos=[]
start_time = time.time()
best_training_acc=0
best_validing_acc=0
best_train_model = copy.deepcopy(model)
best_valid_model = copy.deepcopy(model)

for epoch in range(num_epochs):


    # 当前epoch的结果保存下来
    train_rights = []
    

    ##training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        right = accuracy(output, target)
        train_rights.append(right)

    ##validing
    model.eval()
    val_rights = []
    for (data, target) in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        right = accuracy(output, target)
        val_rights.append(right)


    ##printing
    # 准确率计算
    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    print('current epoch: {} \tloss: {:.6f}\ttraining acc: {:.2f}%\tvalid acc: {:.2f}%'.format(
        epoch, loss.data,
               100. * train_r[0].cpu().numpy() / train_r[1],
               100. * val_r[0].cpu().numpy() / val_r[1]))
    
    # file storage
    current_epoch = epoch+1
    loss_data = loss.data.cpu().numpy().tolist()
    train_acc = train_r[0].cpu().numpy() / train_r[1]
    valid_acc = val_r[0].cpu().numpy() / val_r[1]
    
    log_info = 'current epoch: {} \tloss: {:.6f}\ttraining acc: {:.2f}%\tvalid acc: {:.2f}%'.format(
        epoch, loss.data,
               100. * train_r[0].cpu().numpy() / train_r[1],
               100. * val_r[0].cpu().numpy() / val_r[1])
    
    train_acc_list.append(train_acc)
    valid_acc_list.append(valid_acc)
    loss_list.append(loss_data)
    log_infos.append(log_info)

    ## store the best acc
    if(train_acc>best_training_acc):
        best_training_acc=train_acc
        best_train_model=copy.deepcopy(model)
        print("now best training acc changed to",best_training_acc)
    if(valid_acc>best_validing_acc):
        best_validing_acc=valid_acc
        best_valid_model=copy.deepcopy(model)
        print("now best validing acc changed to",best_validing_acc)

end_time = time.time()

overall_time = end_time - start_time

## files write
# print(train_acc_list)
# print(valid_acc_list)
# print(loss_list)
# print(overall_time)

file_folder = "runs/"
runs_time =  datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
file_path = file_folder + runs_time + "/"

if not os.path.exists(file_path):
    os.mkdir(file_path)

with open(file_path+"logs.txt","w") as f:
    for item in log_infos:
        f.write(str(item))
        f.write("\n")
    f.write("overall time is ")
    f.write(str(overall_time))

with open(file_path+"train_acc.txt","w") as f:
    for item in train_acc_list:
        f.write(str(item))
        f.write("\n")
with open(file_path+"valid_acc.txt","w") as f:
    for item in valid_acc_list:
        f.write(str(item))
        f.write("\n")
with open(file_path+"loss.txt","w") as f:
    for item in loss_list:
        f.write(str(item))
        f.write("\n")
with open(file_path+"train_acc.txt","w") as f:
    for item in train_acc_list:
        f.write(str(item))
        f.write("\n")
with open(file_path+"netshape.txt","w") as f:
    f.write(str(model))

torch.save(best_train_model.state_dict(),file_path+"best_train_"+str(best_training_acc)[2:4].ljust(2,'0')+"_"+str(best_training_acc)[4:6].ljust(2,'0')+".pkl")
torch.save(best_valid_model.state_dict(),file_path+"best_valid_"+str(best_validing_acc)[2:4].ljust(2,'0')+"_"+str(best_validing_acc)[4:6].ljust(2,'0')+".pkl")