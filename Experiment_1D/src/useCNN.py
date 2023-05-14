# coding: utf-8
# data shape:  conv1d: [batch, channel,length]
#             LSTM:   [batch, length, channel] batch_first=true
#                     [length, batch, channel] batch_first=false (default)
# network choice:
#              self.blstm=True for lstm others for cnn
# In[5]:


import pandas as pd
import numpy as np
import gc
import time
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder

from blitz.modules import BayesianLSTM, BayesianConv1d, BayesianLinear
from blitz.utils import variational_estimator

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# get_ipython().run_line_magic('matplotlib', 'inline')

from collections import deque

import math
import scipy.io as sio

#########---------dataloaderb
# device = torch.device("cuda:0")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# root = '.\RData\SimulationData'
root = './sample_16000'


def ReadData(path):
    # file=h5py.File(path)
    file = sio.loadmat(path)
    # echoS=file[list(file.keys())[0]]
    # s=echoS['real']+echoS['imag']*1j
    s = file['hbreal']
    # s=torch.from_numpy(np.abs(s)).float()
    return s


def Standardize(sample):
    scaler = StandardScaler().fit_transform(sample.transpose())
    return torch.from_numpy(scaler).float()


BatchSize = 1600
MyDataSet = DatasetFolder(root=root, loader=ReadData, transform=Standardize, extensions='.mat')
# MyDataSet=DatasetFolder(root=root,loader=ReadData,extensions='.mat')
# TargetValue=np.array(MyDataSet.classes).astype(float) # convert class to number for regression
# TargetValue=torch.from_numpy(TargetValue).float()
Total = len(MyDataSet)
TrainNum = math.floor(0.6 * Total)
TestNum = math.floor(0.4 * Total)
ValNum = Total - TrainNum - TestNum

TrainSet, TestSet, ValSet = torch.utils.data.random_split(MyDataSet, [TrainNum, TestNum, ValNum])
TrainDataLoader = DataLoader(TrainSet, batch_size=BatchSize, shuffle=True)
TestDataLoader = DataLoader(TestSet, batch_size=BatchSize, shuffle=False)
im, label = next(iter(TrainDataLoader))


# plt.plot(im[0,:,:])
# plt.show()

class Logger(object):
    def __init__(self, filename='hbreal-cnn.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(stream=sys.stdout)


@variational_estimator
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # self.conv1d=BayesianConv1d(1,64,21,stride=11)
        self.model = nn.Sequential(
            nn.Conv1d(1, 64, 11),
            nn.ReLU(),
            nn.MaxPool1d(5, 3),
            nn.Conv1d(64, 128, 11),
            nn.ReLU(),
            nn.MaxPool1d(5, 3),
            nn.Conv1d(128, 256, 11),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 3),
            nn.Conv1d(256, 512, 11),
            nn.LeakyReLU(),
            nn.MaxPool1d(5, 3)
        )
        self.lstm_1 = BayesianLSTM(512, 1024, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        # self.linear1 = BayesianLinear(9216, 200) #cnn
        # self.linear2 = BayesianLinear(200, 3)
        #  self.linear  = BayesianLinear(18432,100) #LSTM
        self.linearlstm = nn.Linear(1024, 100)
        self.linearlast = nn.Linear(100, 7)
        self.blstm = True  # using lstm

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channel, length]
        # x=self.conv1d(x)
        x = self.model(x)
        if self.blstm:  # lstm
            x = x.permute(0, 2, 1)
            x_, _ = self.lstm_1(x)
            # gathering only the latent end-of-sequence for the linear layer
            x_ = x_[:, -1, :]
            x_ = self.linearlstm(x_)
            # x_ = self.linear(x_.view(-1, x_[0].shape[0] * x_[0].shape[1]))
            x_ = self.linearlast(x_)
        else:  # cnn
            # x_ = nn.Sigmoid()(self.linear1(x.view(-1,x[0].shape[0]*x[0].shape[1])))
            x_ = nn.ReLU()(self.linear1(x.view(-1, x[0].shape[0] * x[0].shape[1])))
            x_ = self.linear2(x_)
            # x_ = self.linear(x_)
        return x_


net = NN().to(device)

criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
# optimizer = optim.Adam(net.parameters(), lr=0.0002) # lstm: 0.002; cnn: 0.0015
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)
# In[11]:
# 训练
numepoch = 30
iteration = 0
history = {'TrainLoss': [], 'TestLoss': [], 'TestAccuracy': []}
for epoch in range(numepoch):
    TLoss = 0
    iteration = 0
    for i, (datapoints, labels) in enumerate(TrainDataLoader, 0):
        # print('label:{}\n'.format(labels))
        # labels=TargetValue[labels].unsqueeze(1)
        optimizer.zero_grad()
        # plt.imshow(datapoints[0,:,:])
        # plt.show()
        loss = net.sample_elbo(inputs=datapoints.to(device),
                               labels=labels.to(device),
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight=0.0000001 / datapoints.shape[0])
        loss.backward()
        optimizer.step()

        iteration += 1
        TLoss += loss
    history['TrainLoss'].append(TLoss / len(TrainSet))
    if epoch < numepoch:
        #  net.freeze_()
        loss_test = 0
        correct = 0
        with torch.no_grad():
            for j, (X_test, y_test) in enumerate(TestDataLoader):
                # y_test=TargetValue[y_test].unsqueeze(1)
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # net.freeze_()
                preds_test = net(X_test)
                # net.unfreeze_()
                loss_test += criterion(preds_test, y_test)
                corr = (preds_test.max(dim=1).indices == y_test).sum()
                correct += corr

        Curr_LR = optimizer.param_groups[0]['lr']
        print("Epoch: {} Train-loss: {:.4f} Val-loss: {:.4f} accuracy: {:.4f} LR: {:.6f}\n".format(str(epoch),
                                                                                                   TLoss / len(
                                                                                                       TrainSet),
                                                                                                   loss_test / len(
                                                                                                       TestSet),
                                                                                                   correct / len(
                                                                                                       TestSet),
                                                                                                   Curr_LR))
        print("Iteration: {} accuracy: {:.4f} LR: {}\n".format(str(iteration), loss_test / len(TestSet), Curr_LR))
        scheduler.step(loss_test / len(TestSet))
        net.unfreeze_()
        history['TestLoss'].append(loss_test / len(TestSet))
        history['TestAccuracy'].append(correct / len(TestSet))

# 保存权重并画图
torch.save(net.state_dict(), './SavedModel/OriginalModel-CNN.pth')
fig = plt.figure()
plt.plot(history['TrainLoss'])
plt.plot(history['TestLoss'])
plt.plot(history['TestAccuracy'])
plt.legend(['TrainLoss', 'TestLoss', 'TestAccuracy'])
plt.xlabel('epoch')
plt.savefig('./Experiments/TrainTestLossAccuracyCurve-2.png')
plt.close(fig)
plt.clf()


def threshold(mu, rho, ratio):
    var = torch.log(1 + torch.exp(rho))
    snr = mu.abs() / var
    ws = torch.flatten(snr)
    ws, ind = torch.sort(ws)
    length = len(ws)
    ind = ratio * length
    thr = ws[np.int(ind)]
    mu[snr < thr] = 0
    return mu


def CompressModel(modelfile, net, ratio):
    state = torch.load(modelfile)
    net.load_state_dict(state)
    net.model[0].freeze = True
    net.model[3].freeze = True
    net.model[6].freeze = True
    net.model[9].freeze = True
    net.lstm_1.freeze = True
    net.linearlstm.freeze = True
    net.linearlast.freeze = True

    # ratio=0.9
    neginf = -float('inf')
    mu = net.model[0].weight_mu.cpu()
    rho = net.model[0].weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.model[0].weight_mu = nn.parameter.Parameter(mu)
    net.model[0].weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    #  net.model[0].weight_sampler = \
    #     TrainableRandomDistribution(net.model[0].weight_mu, net.model[0].weight_rho)

    mu = net.model[3].weight_mu.cpu()
    rho = net.model[3].weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.model[3].weight_mu = nn.parameter.Parameter(mu)
    net.model[3].weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    #  net.model[3].weight_sampler = \
    #     TrainableRandomDistribution(net.model[3].weight_mu, net.model[3].weight_rho)

    mu = net.model[6].weight_mu.cpu()
    rho = net.model[6].weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.model[6].weight_mu = nn.parameter.Parameter(mu)
    net.model[6].weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    # net.model[6].weight_sampler = \
    #    TrainableRandomDistribution(net.model[6].weight_mu, net.model[6].weight_rho)

    mu = net.model[9].weight_mu.cpu()
    rho = net.model[9].weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.model[9].weight_mu = nn.parameter.Parameter(mu)
    net.model[9].weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    #  net.model[9].weight_sampler = \
    #     TrainableRandomDistribution(net.model[9].weight_mu, net.model[9].weight_rho)

    mu = net.lstm_1.weight_ih_mu.cpu()
    rho = net.lstm_1.weight_ih_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.lstm_1.weight_ih_mu = nn.parameter.Parameter(mu)
    net.lstm_1.weight_ih_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    # net.lstm_1.weight_ih_sampler = \
    #    TrainableRandomDistribution(net.lstm_1.weight_ih_mu, net.lstm_1.weight_ih_rho)

    mu = net.lstm_1.weight_hh_mu.cpu()
    rho = net.lstm_1.weight_hh_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.lstm_1.weight_hh_mu = nn.parameter.Parameter(mu)
    net.lstm_1.weight_hh_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    # net.lstm_1.weight_hh_sampler = \
    #    TrainableRandomDistribution(net.lstm_1.weight_hh_mu, net.lstm_1.weight_hh_rho)

    mu = net.linearlstm.weight_mu.cpu()
    rho = net.linearlstm.weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.linearlstm.weight_mu = nn.parameter.Parameter(mu)
    net.linearlstm.weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    # net.linearlstm.weight_sampler = \
    #    TrainableRandomDistribution(net.linearlstm.weight_mu, net.linearlstm.weight_rho)

    mu = net.linearlast.weight_mu.cpu()
    rho = net.linearlast.weight_rho.cpu()
    mu = threshold(mu, rho, ratio)
    net.linearlast.weight_mu = nn.parameter.Parameter(mu)
    net.linearlast.weight_rho = nn.parameter.Parameter(torch.ones_like(rho) * neginf)
    # net.linearlast.weight_sampler = \
    #     TrainableRandomDistribution(net.linearlast.weight_mu, net.linearlast.weight_rho)
    return net


numimage = 0
keepErrorRecognition = False


def TestCompressedModel(net, TestDataLoader, device, criterion):
    global numimage
    loss_test = 0
    correct = 0
    with torch.no_grad():
        for j, (X_test, y_test) in enumerate(TestDataLoader):
            # y_test=TargetValue[y_test].unsqueeze(1)
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            preds_test = net.to(device)(X_test)
            loss_test += criterion(preds_test, y_test)
            corr = (preds_test.max(dim=1).indices == y_test).sum()
            correct += corr

            '''   if keepErrorRecognition:
                pred_y_test = preds_test.max(dim=1).indices
                errorind = torch.from_numpy(np.array((np.where(pred_y_test.cpu() != y_test.cpu()))))
                errorSeq = X_test[errorind, :, :]
                epred_y = pred_y_test[errorind]
                e_y = y_test[errorind]
                errorSeq = np.squeeze(errorSeq)
                errorind=errorind.squeeze()
                errorSeq=errorSeq.squeeze()
                epred_y=epred_y.squeeze()
                e_y=e_y.squeeze()
                errorind=np.array(errorind)
                for k in range(errorind.size):
                    print('k={}'.format(k))
                    #fig=plt.figure(1)
                    numimage+=1
                    if errorSeq.dim()>1:
                        print(errorSeq.shape[0])
                        plt.plot(errorSeq[k, :].cpu())
                    else:
                        print(errorSeq.shape[0])
                        plt.plot(errorSeq.cpu())
                    if errorind.size==1:
                        pred=epred_y.cpu().item()
                        truth=e_y.cpu().item()
                    else:
                        pred = epred_y[k].cpu().item()
                        truth = e_y[k].cpu().item()
                    plt.savefig('./Experiments/error-pred-{:d}-true-{:d}-image-{:d}.png'
                                    .format(pred, truth,numimage),dpi=1200)
                    #plt.show()
                    time.sleep(0.01)
                    #plt.close(fig)
                    plt.clf()
'''
    return loss_test, correct


# now it works

print("TrainLoss:")
print(history['TrainLoss'])
print("TestLoss:")
print(history['TestLoss'])
print("TestAccuracy:")
print(history['TestAccuracy'])
print("done")
