import torch
from torch import optim
import torch.nn.functional as F
import random
import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
import torchsummary
import torchvision.transforms as T
import Train
import time
import copy
import pandas as pd
from scipy.stats import norm

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns 
import pandas as pd
from torch.nn.functional import softmax
from sklearn.model_selection import train_test_split
from sklearn import linear_model, model_selection    
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scipy.stats
from numpy.linalg import norm
from sklearn.utils import shuffle



def extractor_loss_functionnmse(e_out, e):

    loss = nn.MSELoss(reduction='none')
    MSE = loss(e_out, e)
    MSE = MSE.sum(dim=1)
    #print(MSE)
    MSE = torch.sum(torch.exp(MSE/(MSE + 1)))
    return MSE


def extractor_loss_functioncosine(s_e, s_e_teacher, u_e, u_e_teacher, temp):
    
    loss1 = nn.CosineEmbeddingLoss(reduction='none')

    COS1 = loss1(s_e, s_e_teacher, torch.ones(s_e.shape[0]).cuda())
    
    loss2 = nn.CosineEmbeddingLoss(reduction='none')

    COS2 = loss2(u_e, u_e_teacher, torch.ones(u_e.shape[0]).cuda())
    
    SIM = torch.sum(torch.log(torch.exp(COS1)/torch.sum(torch.exp(COS2/temp))))
    
    return SIM




def extractor_unlearning(trained_model, s_vae, u_vae, loaders, epochs, lr_ue, lr_ra, temp):
    trained_model = trained_model.cuda()
    s_vae = s_vae.cuda()
    s_vae.eval()
    u_vae = u_vae.cuda()
    u_vae.eval()
    teacher_model = copy.deepcopy(trained_model)
    teacher_model.eval()
    optimizer1 = torch.optim.Adam(trained_model.parameters(), lr = lr_ue)
    optimizer2 = torch.optim.Adam(trained_model.parameters(), lr = lr_ra)
    all_loss_list = []
    all_loss2_list = []
    for epoch in range(epochs):
        all_loss = 0
        all_loss2 = 0
        all_loss4 = 0
        all_loss5 = 0
        count = 0
        
        for i, ((u_data, u_targets),(s_data, s_targets)) in enumerate(zip(loaders['unlearn'],loaders['remain'])):
            
            u_data = u_data.cuda()
            u_targets = u_targets.cuda()
            s_data = s_data.cuda()
            s_targets = s_targets.cuda()
            
            y_out, s_e = trained_model(s_data)
            
            s_e_out, z, s_mu, s_sigma = s_vae(s_e)

            u_y_out, u_e = trained_model(u_data)
            
            u_u_e_out, z, u_u_mu, u_u_sigma = u_vae(u_e)
            
            loss2 = extractor_loss_functionnmse(s_e_out, s_e)  - extractor_loss_functionnmse(u_u_e_out, u_e)
            
            loss =  1 * loss2
            optimizer1.zero_grad()
            loss.backward()
            all_loss += loss.cpu().item() 
            count += 1
            optimizer1.step()
              
            y_out, s_e = trained_model(s_data)
            y_out_teacher, s_e_teacher = teacher_model(s_data)
            
            y_out, u_e = trained_model(u_data)
            y_out_teacher, u_e_teacher = teacher_model(u_data)
            
            loss3 = 1 * extractor_loss_functioncosine(s_e, s_e_teacher, u_e, u_e_teacher, temp)
            optimizer2.zero_grad()
            loss3.backward()
            optimizer2.step()
            
            all_loss2 += loss3.item()
        all_loss_list.append(all_loss/count)
        all_loss2_list.append(all_loss2/count)    
        print(all_loss/count)
        print(all_loss2/count)
            

    return trained_model, all_loss_list, all_loss2_list
            


def classifier_unlearning(trained_model, loaders, epochs, lr):
    trained_model = trained_model.cuda()
    loss_func1 = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(trained_model.fc2.parameters(), lr = lr) 
    for epoch in range(epochs):
        all_loss = 0
        count = 0

        for i, ((u_data, u_targets),(x, y)) in enumerate(zip(loaders['unlearn'],loaders['remain'])):

            x = x.cuda()
            y = y.cuda() 
            y_out, _ = trained_model(x)

            loss = loss_func1(y_out, y)
                
            
            optimizer.zero_grad()
            loss.backward()
            all_loss += loss.cpu().item()
            count += 1
            optimizer.step()

        epoch_loss = all_loss/count
        print('Epoch: ', epoch, ' Loss: ', epoch_loss)   
            
    return trained_model

    
