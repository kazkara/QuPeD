
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np

import copy
import random
import time
import math

from utils import quantize
from utils import init_centers
from utils import update_prox


import torch.backends.cudnn as cudnn
import resnet as resnet

ROOT = '.data'
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

torch.manual_seed(1234)
np.random.seed(1234)

cudnn.benchmark = True

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])

train_data = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True)

test_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))




n_train_examples = int(len(train_data))

BATCH_SIZE = 128

train_iterator = data.DataLoader(train_data, 
                                 shuffle = True, 
                                 batch_size = BATCH_SIZE)

test_iterator = data.DataLoader(test_data, 
                                batch_size = BATCH_SIZE)
OUTPUT_DIM = 10

device = torch.device('cuda:0')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
resn = 8   #Choose whether you will run for ResNet20 or ResNet32

if resn == 20:
    model = resnet.resnet20()
elif resn == 32:
    model = resnet.resnet32()
else:
    model = resnet.resnet8()

model.to(device)
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)


cold_start = True  #Choose whether Cold or Warm Start (results stated in the paper are for warm start)
fp = False          #True for full precision training, False for quantized training
lambda_coef = 1e-4 #1e-5 for resn8, 1e-4 for resnet20,32,  1e-6 for cnet
weight_decay = 0
l1 = True
adam = True
lr2 = 1e-4
freeze_epoch = 200  #Epoch number after which fine tuning is done
EPOCHS = 250        #Total Epochs
n_centroids = 2 #Choose the number of centers
#Don't need finetuning if fp is chosen
if fp:
    Epochs = 200
    freeze_epoch = 500
lambda_coef2 = lambda_coef

optimizer = optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay = weight_decay)

optimizer_adam = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

if not adam:
    lr1 = optimizer.param_groups[0]['lr']
else:
    lr1 = optimizer_adam.param_groups[0]['lr']

lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=[100, 150], last_epoch=-1)

# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, #87.2% best test accuracy
#                     gamma = 0.99, last_epoch=-1)

if not cold_start:
    if resn == 20:
        model.load_state_dict(torch.load('resnet20_fp_fpinit.pt',map_location='cuda:0'))
    elif resn == 32:
        model.load_state_dict(torch.load('resnet32_fp_sgd1init.pt',map_location='cuda:0'))

#Initialize Centers
centers = init_centers(model,n_centroids)

if adam:
    centers[:,0] = -1
    centers[:,-1] = 1

def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

         
def evaluate(model, iterator, criterion, device):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (x, y) in iterator:

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


layer_glb = 0
count_layer = 0
first_layer = True


def train(model, iterator, optimizer, criterion, device, centers, epoch, count_layer,
          lr1, lr2, lambda_coef, lambda_coef2, iteration, weight_decay, tune_mode, l1):
    
    epoch_loss = 0
    epoch_acc = 0
    batch_count = 0
    model.train()
    no_of_members_old = torch.zeros_like(centers)
    changing_weights_ep = torch.zeros(18)
    if fp:
        no_of_members = torch.zeros_like(centers)

    for (x, y) in iterator:
        
        
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()
                
        y_pred = model(x)
        
        loss = criterion(y_pred, y)

        acc = calculate_accuracy(y_pred, y)
        batch_count += 1
        loss.backward()
   
        lamb = (lambda_coef*(epoch+1))
        lamb2 = (lambda_coef2*(epoch+1))

        if fp:
            optimizer.step()
        else:
            if not adam:
                optimizer.step()
                no_of_members  = update_prox(False, device, model, 0, 0, centers, n_centroids, lr1, lr2, lamb, lamb2, 0,
                             x, y, criterion, 0, tune_mode, l1, weight_decay ,False, False, 0, 0 ,False, False, resn)
            elif adam:
                optimizer_adam.step()
                no_of_members  = update_prox(False, device, model, 0, 0, centers, n_centroids, lr1, lr2, lamb, lamb2, 0,
                             x, y, criterion, 0, tune_mode, l1, weight_decay ,False, False, 0, 0 ,False, False, resn)

    

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_perc = 100*epoch_acc/batch_count
        iteration += 1
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator),no_of_members, lamb, lamb2, iteration


    


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


best_test_acc = 0
best_test_qacc = 0


m = torch.zeros_like(centers)
v = torch.zeros_like(centers)
count = 0 
iteration_centroid = 10



train_acc_list =[]
valid_acc_list =[]
train_loss_list =[]
valid_loss_list =[]
test_loss_list =[]
test_acc_list =[]

train_qacc_list =[]
valid_qacc_list =[]
train_qloss_list =[]
valid_qloss_list =[]
test_qloss_list =[]
test_qacc_list =[]
no_of_members = torch.zeros_like(centers)



test_loss, test_acc = evaluate(model, test_iterator, criterion, device)
train_loss, train_acc = evaluate(model, train_iterator, criterion, device)
iteration = 0
tune_mode = False

for epoch in range(EPOCHS):
    
    if resn!=8:
        if not adam:
            if lr1 != optimizer.param_groups[0]['lr']:
                lambda_coef *= 10
        else:
            if lr1 != optimizer_adam.param_groups[0]['lr']:
                lambda_coef *= 10

    
    if not adam:
        lr1 = optimizer.param_groups[0]['lr']
    else:
        lr1 = optimizer_adam.param_groups[0]['lr']
     
    if lr2 != 0 and lr2 != 1e-6:
        if epoch<80:
            lr2 = 1e-4
        elif epoch<140:
            lr2 = 1e-5
        else:
            lr2 = 1e-6

    first_layer = True
    if not fp:
        if epoch+1 == freeze_epoch:
            for p in model.parameters():
                if p.data.dim() == 4 and (not first_layer):
                    p.requires_grad = False
                first_layer = False
            lambda_coef2 = 0
    
    if not fp:
        if epoch > freeze_epoch:
            if lr2 == 0:
                lr2 = 0
            else:
                lr2 = 1e-6
            tune_mode = True
            lambda_coef2 = 0
            
        if epoch+1 == freeze_epoch:
            quantize(model,centers,l1)

    start_time = time.time()

    train_loss, train_acc, no_of_members, lamb, lamb2, iteration = train(model, train_iterator, optimizer, criterion, device,
                                                             centers, epoch, count_layer,lr1, lr2, lambda_coef, lambda_coef2, iteration,
                                                             weight_decay, tune_mode, l1)
    
    
    
    lr_old = lr1
    if not adam:
        lr_scheduler.step()

    test_loss, test_acc = evaluate(model, test_iterator, criterion, device)


    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)

    if resn == 20:
        model_q = resnet.resnet20()
    elif resn == 32:
        model_q = resnet.resnet32()
    else:
        model_q = resnet.resnet8()
    model_q.load_state_dict(model.state_dict())
    quantize(model_q,centers,l1)

    test_qloss, test_qacc = evaluate(model_q, test_iterator, criterion, device)
    train_qloss, train_qacc = evaluate(model_q, train_iterator, criterion, device)

    test_qacc_list.append(test_qacc)
    test_qloss_list.append(test_qloss)

    train_qacc_list.append(train_qloss)
    train_qloss_list.append(train_qloss)

    if resn == 20:
        folder = 'Resnet20/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        if fp:
            save_place =  folder +  'resnet20_fp_'
        else:
            save_place =  folder + 'resnet20_binary_' + str(n_centroids) + 'centers_l12_'
    elif resn == 32:
        folder = 'Resnet32/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        if fp:
            save_place = folder + 'resnet32_fp_' 
        else:
            save_place = folder + 'resnet32_binary_' + str(n_centroids) + 'centers_'
    else:
        folder = 'Resnet8/'
        Path(folder).mkdir(parents=True, exist_ok=True)
        if fp:
            save_place = folder + 'resnet8_fp_' 
        else:
            save_place = folder + 'resnet8_binary_' + str(n_centroids) + 'centers_lamb' + str(lambda_coef) + '_'

    if adam:
        save_place = save_place + 'adam_' 
    else:
        save_place = save_place + 'sgd_' 
    
    save_place_q_pt = save_place + ' quantized' + '.pt'
    save_place_pt = save_place + '.pt'

    if test_qacc > best_test_qacc:
        best_test_qacc = test_qacc
        torch.save(model_q.state_dict(), save_place_q_pt)
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), save_place_pt)


    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    save_place_chn_pt =  save_place + '_changing_weights' + '.pt'
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    print(f'\t Best Test Acc: {best_test_acc*100:.2f}%')
    print(f'\t Quantized Test Loss: {test_qloss:.3f} |  Test Acc: {test_qacc*100:.2f}%')
    print(f'\t Quantizeed Best Test Acc: {best_test_qacc*100:.2f}%')

    save_place_test_txt =  save_place + '_test' + '.txt'
    save_place_trn_txt =  save_place + '_train' + '.txt'
    save_place_q_test_txt =  save_place + '_q_testacc' + '.txt'
    save_place_q_trainloss_txt = save_place + '_q_trainloss' + '.txt'

    with open(save_place_test_txt, 'w') as filehandle:
        json.dump(test_acc_list, filehandle)
        
    with open(save_place_trn_txt, 'w') as filehandle:
        json.dump(train_acc_list, filehandle)

    with open(save_place_q_test_txt, 'w') as filehandle:
        json.dump(test_qacc_list, filehandle)
    
    with open(save_place_q_trainloss_txt, 'w') as filehandle:
        json.dump(train_qloss_list, filehandle)


