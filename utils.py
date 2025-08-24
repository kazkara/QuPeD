# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 19:03:38 2020

@author: ukaan
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.distributed

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import numpy as np

from torch.autograd import Variable

import copy
import random
import time
import math
import cifarnet as cifarnet
import resnet as resnet
import nlp_model as nlp_model


def get_mean(model,n_centroids, cnet =True):
    init_coef = 1
    init_params = []
    count_layer = 0
    
    first_layer = True
    for p in model.parameters():
        if cnet:
            if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                first_layer = False 
                continue                
        else:
            if p.data.dim() != 4 or first_layer:
                first_layer = False 
                continue
        mask_pos = p.data >= 0
        mask_neg = p.data < 0   
        boundmin = torch.sum(p.data*mask_neg)/torch.sum(mask_neg)
        boundmin = boundmin.cpu().item()
        boundmax = torch.sum(p.data*mask_pos)/torch.sum(mask_pos)
        boundmax = boundmax.cpu().item()                
        init_params.append(torch.linspace(boundmin,boundmax,n_centroids))
        count_layer += 1    
    centers = torch.zeros(count_layer,n_centroids)
        

    for ind in range(count_layer):
        centers[ind,:] = init_params[ind]
    
    
    return centers

def init_centers(model,n_centroids, cnet = True , nlp=False):
    init_coef = 1
    init_params = []
    count_layer = 0
    
    first_layer = True
    if n_centroids>2:  ##WARNING
        for p in model.parameters():
            if cnet:
                if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                    first_layer = False 
                    continue                
            else:
                if not nlp:
                    if p.data.dim() != 4 or first_layer:
                        first_layer = False 
                        continue
                else:
                    if p.data.shape[0] == 4:
                        continue  
            boundmin = torch.min(p.data).cpu().item()
            boundmax = torch.max(p.data).cpu().item()
            init_params.append(torch.linspace(boundmin,boundmax,n_centroids))
            count_layer += 1

        centers = torch.zeros(count_layer,n_centroids)
        

        for ind in range(count_layer):
            centers[ind,:] = init_params[ind]
        

    else:
        for p in model.parameters():
            if cnet:
                if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                    first_layer = False 
                    continue                
            else:
                if not nlp:
                    if p.data.dim() != 4 or first_layer:
                        first_layer = False 
                        continue
                else:
                    if p.data.shape[0] == 4:
                        continue  

            mask_pos = p.data >= 0
            mask_neg = p.data < 0   
            boundmin = torch.sum(p.data*mask_neg)/torch.sum(mask_neg)
            boundmin = boundmin.cpu().item()
            boundmax = torch.sum(p.data*mask_pos)/torch.sum(mask_pos)
            boundmax = boundmax.cpu().item()                
            init_params.append(torch.linspace(boundmin,boundmax,n_centroids))
            count_layer += 1    
        centers = torch.zeros(count_layer,n_centroids)
        

        for ind in range(count_layer):
            centers[ind,:] = init_params[ind]
    
    
    return centers


def quantize(model,centers, l1, cnet = True  , nlp=False):
    n = centers.shape[1]
    if n>2:  #IMPORTANT 
        layer = 0
        first_layer = True
        if nlp:
            first_layer = False
        for p in model.parameters():
            if p.data.dim() == 4:
                distances = p.data.repeat(n, 1,1,1,1)
            elif p.data.dim() == 3:
                distances = p.data.repeat(n, 1,1,1)
            elif p.data.dim() == 2:
                distances = p.data.repeat(n, 1,1)
            elif p.data.dim() == 1:
                distances = p.data.repeat(n, 1)

            if cnet:
                if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                    first_layer = False 
                    continue                
            else:
                if not nlp:
                    if p.data.dim() != 4 or first_layer:
                        first_layer = False 
                        continue
                else:
                    if p.data.shape[0] == 4:
                        continue


            for ind in range(n):
                if l1:
                    distances[ind] = abs(p.data-centers[layer,ind])
                else:
                    distances[ind] = (p.data-centers[layer,ind])**2
            
            indx_tensor = torch.argmin(distances,0)
            layer_centers = centers[layer,:]
            p.data.copy_(layer_centers[indx_tensor.long()])    
            layer += 1
    else:
        layer = 0
        first_layer = True
        if nlp:
            first_layer = False

        for p in model.parameters():
            if p.data.dim() == 4:
                distances = p.data.repeat(n, 1,1,1,1)
            elif p.data.dim() == 3:
                distances = p.data.repeat(n, 1,1,1)
            elif p.data.dim() == 2:
                distances = p.data.repeat(n, 1,1)
            elif p.data.dim() == 1:
                distances = p.data.repeat(n, 1)

            if cnet:
                if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                    first_layer = False 
                    continue                
            else:
                if not nlp:
                    if p.data.dim() != 4 or first_layer:
                        first_layer = False 
                        continue
                else:
                    if p.data.shape[0] == 4:
                        continue
        
            mask_pos = p.data >= 0
            mask_neg = p.data < 0
            
            p.data.copy_(mask_neg*centers[layer,0] + mask_pos*centers[layer,1])
            layer += 1

# update_prox(last_iter, device, model, model_p, lambda_p, centers, n_centroids, lr1, lr2, lamb, lamb2,
#                                 temp, x, y, criterion, criterion_KL, tune_mode, l1, weight_decay, cnet, models_type[rank], rank)
def update_prox(last_iter, device, m, model_p, lambda_p, centers, n, lr1, lr2, lamb, lamb2, temp,
                        x, y, criterion, criterion_KL, tune_mode, l1, weight_decay, cnet, mnist, models_type, 
                        rank = 0, qupel = False, imagenet=False, resn = 0, nlp = False, offs=None, vocab=None):

    with torch.no_grad():
        no_of_members = torch.zeros_like(centers,device=device)
        pos_members = torch.zeros_like(centers,device=device)
        neg_members = torch.zeros_like(centers,device=device)

        if nlp:
            vocab_size = len(vocab)

        if imagenet:
            model_comp = resnet.resnet18()
        elif cnet:
            if models_type == 0:
                if not mnist:
                    model_comp = cifarnet.Cifarnet()
                else:
                    model_comp = cifarnet.Mnistnet()
            elif models_type == 1:
                if not mnist:
                    model_comp = cifarnet.Cifarnet2()
                else:
                    model_comp = cifarnet.Mnistnet2()
        elif resn != 0:
            if resn == 20:
                model_comp = resnet.resnet20()
            elif resn == 32:
                model_comp = resnet.resnet32()
            else:
                model_comp = resnet.resnet8()
        else:
            model_comp = nlp_model.TextClassificationModel(vocab_size, 64, 4)

        model_comp.load_state_dict(m.state_dict())
        model_comp.to(device)
        quantize(model_comp,centers,l1,cnet,nlp)
        first_layer = True
        layer = 0
        
        for p,pc in zip(m.parameters(),model_comp.parameters()):

            if cnet:
                if p.data.shape[0] == 10 or first_layer or p.data.dim() == 1:
                    first_layer = False 
                    continue                
            else:
                if not nlp:
                    if p.data.dim() != 4 or first_layer:
                        first_layer = False 
                        continue
                else:
                    if p.data.shape[0] == 10:
                        continue               
            
            if l1:
                if lr2 == 0:

                    p_abs = torch.abs(p.data)
                    p_sign = torch.sign(p.data)
                    p.data.copy_(p_sign*(torch.relu(torch.abs(p_abs-1)-lamb*lr1)*torch.sign(p_abs-1)+1))
                else:
                    mask_greater = (p.data) > pc.data + lamb*lr1
                    mask_less = (p.data) < pc.data - lamb*lr1
                    mask_btw =(((p.data) >= pc.data - lamb*lr1) & ((p.data) <= pc.data + lamb*lr1))
                    p.data.copy_((p.data-lr1*lamb)*mask_greater+(p.data+lr1*lamb)*mask_less+
                                                                            (pc.data)*mask_btw)                
                p.data.clamp_(-1, 1)


            else:                    
                p.data.copy_((p.data+2*lamb*lr1*pc.data)/(1+2*lamb*lr1))
        
                
            layer += 1


    if imagenet:
        model_comp = resnet.resnet18()
    elif cnet:
        if models_type == 0:
            if not mnist:
                model_comp = cifarnet.Cifarnet()
            else:
                model_comp = cifarnet.Mnistnet()
        elif models_type == 1:
            if not mnist:
                model_comp = cifarnet.Cifarnet2()
            else:
                model_comp = cifarnet.Mnistnet2()
    elif resn !=0:
        if resn == 20:
            model_comp = resnet.resnet20()
        elif resn == 32:
            model_comp = resnet.resnet32()
        else:
            model_comp = resnet.resnet8()
    else:
        model_comp = nlp_model.TextClassificationModel(vocab_size, 64, 4)
    
    model_comp.to(device)
    model_comp.load_state_dict(m.state_dict())
    quantize(model_comp,centers, l1,cnet, nlp)

    model_comp.zero_grad()
    if nlp:
        y_pred = model_comp(x,offs)
    else:
        y_pred = model_comp(x)

    if lambda_p != 0:
        if nlp:
            y_pred_p = model_p(x,offs)
        else:
            y_pred_p = model_p(x)
            
        if not qupel:
            loss_kl = criterion_KL(F.log_softmax(y_pred, dim = 1), 
                                                                F.softmax(Variable(y_pred_p/temp), dim=1))
    loss_ce = criterion(y_pred,y)
    
    

    if lambda_p != 0:
        if qupel:
            # reg = 0
            # for p_p, p in zip(model_p.parameters(), model_comp.parameters()):
            #     reg += 0.5*((p-p_p) ** 2).sum()
            loss = loss_ce# + lambda_p*reg
        else:
            loss = (1-lambda_p)*loss_ce + (lambda_p)*(temp**2)*loss_kl #+ weight_decay*reg
    else:
        loss = loss_ce #+ weight_decay*reg
    
    # loss_ce = criterion(y_pred,y)
    # loss = loss_ce
    loss.backward()
    if qupel:
        for p_p, p in zip(model_p.parameters(), model_comp.parameters()):
            p.grad.copy_(p.grad+lambda_p*(p.data-p_p.data))

    last_iter = False
    #center updates:
    with torch.no_grad():
        no_of_members = torch.zeros_like(centers,device=device)
        if not last_iter:
            center_gradients = torch.zeros_like(centers,device=device)
            
            layer = 0    
            
            
            sum_of_members = torch.zeros_like(centers,device=device,)
            first_layer = True
            for p,pc in zip(m.parameters(), model_comp.parameters()):
                
                

                if cnet:
                    if pc.data.shape[0] == 10 or first_layer or pc.data.dim() == 1:
                        first_layer = False 
                        continue                
                else:
                    if not nlp:
                        if pc.data.dim() != 4 or first_layer:
                            first_layer = False 
                            continue
                    else:
                        if pc.data.shape[0] == 4:
                            continue     

                if pc.grad.is_sparse:
                    pc.grad = pc.grad.to_dense()

                if pc.data.dim() == 4:
                    p_rep = pc.grad.repeat(n, 1,1,1,1)
                elif pc.data.dim() == 3:
                    p_rep = pc.grad.repeat(n, 1,1,1)
                elif pc.data.dim() == 2:
                    p_rep = pc.grad.repeat(n, 1,1)
                elif pc.data.dim() == 1:
                    p_rep = pc.grad.repeat(n, 1)
                mask = torch.zeros_like(p_rep)

                for ind in range(n):
                    if pc.data.dim() == 4:
                        mask[ind:ind+1, :,:,:,:] = pc.data == centers[layer,ind]
                        masked_data = mask*p_rep
                        sum_grad = torch.sum(masked_data[ind:ind+1,:,:,:,:])
                    elif pc.data.dim() == 3:
                        mask[ind:ind+1, :,:,:] = pc.data == centers[layer,ind]
                        masked_data = mask*p_rep
                        sum_grad = torch.sum(masked_data[ind:ind+1,:,:,:])
                    elif pc.data.dim() == 2:
                        mask[ind:ind+1, :,:] = pc.data == centers[layer,ind]
                        masked_data = mask*p_rep
                        sum_grad = torch.sum(masked_data[ind:ind+1,:,:])
                    elif pc.data.dim() == 1:
                        mask[ind:ind+1, :] = pc.data == centers[layer,ind]                    
                        masked_data = mask*p_rep
                        sum_grad = torch.sum(masked_data[ind:ind+1,:])
  
                    center_gradients[layer,ind:ind+1] = center_gradients[layer,ind:ind+1] + sum_grad
                    
                    
                    mask_upd_1 = pc.data == centers[layer,ind]
                    mask_pos = ((p.data-pc.data)*mask_upd_1)>0
                    mask_neg = ((p.data-pc.data)*mask_upd_1)<0

                    no_of_members[layer,ind] += torch.sum(mask_upd_1)

                    pos_members[layer,ind] += torch.sum(mask_pos)
                    neg_members[layer,ind] += torch.sum(mask_neg)
                    
                    sum_of_members[layer,ind] += torch.sum(mask_upd_1*p.data)

                layer += 1
            
            center_gradient_means = center_gradients/no_of_members

            nans = torch.isnan(center_gradient_means)
            center_gradient_means.masked_fill_(nans,0)
            if not nlp:
                grads_clipped = torch.clamp(center_gradients, -10,10)
            else:
                grads_clipped = torch.clamp(center_gradients, -1,1)
            centers_old = centers.clone()
            # pos_members[-2,:] = pos_members[-2,:]/5
            # neg_members[-2,:] = neg_members[-2,:]/5
            centers.to(device)
            if l1:
                centers += - (lr2*(grads_clipped)+(lamb2)*lr2*(pos_members-neg_members)).cpu()
            else:
                centers += (-lr2*center_gradients + 2*lamb2*lr2*sum_of_members).cpu()
                centers /= (1 + 2*lamb2*lr2*no_of_members)

            centers_sorted, indices = torch.sort(centers)
            centers.copy_(centers_sorted)

        return no_of_members


