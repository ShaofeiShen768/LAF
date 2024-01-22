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
from torchvision import models, transforms 
from  torchvision.models import ResNet18_Weights


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        embedding = self.fc1(x)
        output = self.fc2(embedding)
        return output, embedding
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        
        resnet = models.resnet18() 
        num_features = resnet.fc.in_features 
        features = list(resnet.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*features)
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        embedding = self.feature_extractor(x).squeeze(3).squeeze(2)
        embedding = self.fc1(embedding)
        output = self.fc2(embedding)
        return output, embedding       
        
'''
encoder of vae for unlearning
'''    
class CNN_VAE_Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(CNN_VAE_Encoder, self).__init__()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3_1 = nn.Linear(32, latent_dims)
        self.linear3_2 = nn.Linear(32, latent_dims)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu =  self.linear3_1(x)
        sigma = torch.exp(0.5 * self.linear3_2(x))
        z = mu + sigma*torch.randn_like(sigma)
        return z, mu, sigma
    
    
'''
decoder of vae for unlearning
'''
class CNN_VAE_Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(CNN_VAE_Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 256)

    def forward(self, z):
        
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        output = torch.sigmoid(self.linear3(z))
        return output



    
'''
decoder of vae for unlearning
'''
class ResNet_VAE_Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(ResNet_VAE_Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 32)
        self.linear2 = nn.Linear(32, 128)
        self.linear3 = nn.Linear(128, 256)

    def forward(self, z):
        
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        output = torch.sigmoid(self.linear3(z))
        return output   
    
     
'''
encoder of vae for unlearning
'''    
class ResNet_VAE_Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(ResNet_VAE_Encoder, self).__init__()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, 32)
        self.linear3_1 = nn.Linear(32, latent_dims)
        self.linear3_2 = nn.Linear(32, latent_dims)

    def forward(self, x):

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu =  self.linear3_1(x)
        sigma = torch.exp(0.5 * self.linear3_2(x))
        z = mu + sigma*torch.randn_like(sigma)
        return z, mu, sigma


class VariationalAutoencoder(nn.Module):
    def __init__(self, dataset, latent_dims):
        super(VariationalAutoencoder, self).__init__()

        if dataset == 'Digit' or dataset == 'Fashion':
            self.encoder = CNN_VAE_Encoder(latent_dims)
            self.decoder = CNN_VAE_Decoder(latent_dims)
        elif dataset == 'CIFAR10' or dataset == 'SVHN':
            self.encoder = ResNet_VAE_Encoder(latent_dims)
            self.decoder = ResNet_VAE_Decoder(latent_dims)

    def forward(self, x):
  
        z, mu, sigma = self.encoder(x)
        return self.decoder(z), z, mu, sigma  