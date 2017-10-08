import torch

import torchvision.datasets as dataSets
import torchvision.transforms as Trans

import torch.optim as Optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

import DensenetArch as Dense

import time

# Hyper parameters
epoch = 6
batchsz = 100
lr = 0.01
growthRate = 6
reduction = 0.5
nLayers = 3
nClasses = 10

print('running')
#load data
TrainTransform=Trans.Compose([
    Trans.RandomHorizontalFlip(),
    Trans.ToTensor()
])

tr_data=dataSets.MNIST(
    root='./MNIST/train',
    train=True,
    transform=TrainTransform,
    download=True
    )

train_loader=Data.DataLoader(tr_data,batchsz,True)



# print the size of the training dataset
print('training size: ', (tr_data.train_data.size()))    # 60000*28*28
print('label size: ',(tr_data.train_labels.size()))

te_data=dataSets.MNIST(
    root='./MNIST/test',
    train=False,
    transform=Trans.ToTensor(),
    download=True
    )

test_loader=Data.DataLoader(te_data,batchsz,False)

densenet=Dense.Densenet(growthRate,reduction,nLayers,nClasses)
print(densenet)
optimizer=Optim.SGD(densenet.parameters(),lr,momentum=0.9,weight_decay=1e-4)

# training process
for epc in range(epoch):
    print('training...')
    pre_time=time.time()

    for bath_index,(data, target) in enumerate(train_loader):
        x = Variable(data)
        target = Variable(target)

        out = densenet(x)
        optimizer.zero_grad()
        loss = F.cross_entropy(out,target)
        loss.backward()
        optimizer.step()      # update weights

        if bath_index % 50 == 0:
            pred=torch.max(out,1)[1].data  # the second return value of torch max is index;
            accuracy=100*sum(pred==target.data)/len(target)
            print('Epoch: ', epc+1, '| train loss: %.4f' % loss.data[0], '| training accuracy: %.2f%%' % accuracy)

    cur_time=time.time()
    print('training time: %.2f'%(cur_time-pre_time))

    #testing process
    print('testing...')
    incorrect = 0
    total = 0
    for data, target in test_loader:
        x = Variable(data,volatile=True)    # not used in inference mode(no need to do backprop)
        out = densenet(x)

        pred = out.data.max(1)[1]  # the same as torch.max(out,1)[1].data
        incorrect += pred.ne(target).sum()


    acc = 100.*(1-incorrect/len(te_data.test_data))

    print('Test set: Accuracy: %.4f %%' %(acc))






















