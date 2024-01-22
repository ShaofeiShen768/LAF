import torch
from torch import optim
import torch.nn.functional as F
import random
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset
from torch.utils.data import Subset
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
import torchvision.transforms as T
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models, transforms 
import pandas


'''
Construct the dataset used in unlearning.
The dataset contains three senarios
1. some data in the training dataset are misclassified. We aim to remove such data
2. We aim to remove a imbalanced dataset from the training data
3. We aim to remove some catagories from the training data
'''    
class data_construction():
    
    def __init__(self, dataset_name):
        
        # Load Fashion-MNIST        
        if dataset_name == 'Fashion':
            self.train_data = datasets.FashionMNIST(
                root = '../data',
                train = True,                         
                transform = ToTensor(), 
                download = True,            
            )

            self.test_data = datasets.FashionMNIST(
                root = '../data', 
                train = False, 
                transform = ToTensor()
            )
            
        # Load CIFAR10       
        elif dataset_name == 'CIFAR10':
            
            transforms_cifar_train = transforms.Compose([
                transforms.Resize((224, 224)),   
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            transforms_cifar_test = transforms.Compose([
                transforms.Resize((224, 224)),   
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        
            self.train_data = datasets.CIFAR10(
                root = '../data',
                train = True,                         
                transform = transforms_cifar_train, 
                download = True,            
            )
            
            self.test_data = datasets.CIFAR10(
                root = '../data', 
                train = False, 
                transform = transforms_cifar_test,
                download = True,  
            )
            
            
        # Load SVHN       
        elif dataset_name == 'SVHN':
            
            transforms_cifar_train = transforms.Compose([
                transforms.Resize((224, 224)),   
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            transforms_cifar_test = transforms.Compose([
                transforms.Resize((224, 224)),   
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        
            self.train_data = datasets.SVHN(
                root = '../data',
                split = 'train',                        
                transform = transforms_cifar_train, 
                download = True,            
            )
            
            self.test_data = datasets.SVHN(
                root = '../data', 
                split = 'test', 
                transform = transforms_cifar_test,
                download = True,  
            )
            
            self.train_data.targets = self.train_data.labels
            self.test_data.targets = self.test_data.labels
            
        # Load MNIST 
        elif dataset_name == 'Digit':
            
            self.train_data = datasets.MNIST(
                root = '../data',
                train = True,                         
                transform = ToTensor(), 
                download = True,    
            )        

            self.test_data = datasets.MNIST(
                root = '../data', 
                train = False, 
                transform = ToTensor()
            )        
        print('Train data length: ', len(self.train_data))
        print('Test data length: ', len(self.test_data))
        

    
    # Construct a dataset contains imbalanced data to be unlearned
    def class_data(self, num = None):
        
        # set random proportion of unlearning data points for each class
        if num is None:
            random_array = np.random.random(len(set(self.train_data.targets.tolist())))
        else:
            random_array = num

        class_count = np.zeros(len(set(np.array(self.train_data.targets).tolist())))
        
        # set random number of unlearning data points for each class
        class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        random_class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        
        # record the indice of data points in each class
        for i in range(len(np.array(self.train_data.targets).tolist())):
            
            class_count[np.array(self.train_data.targets).tolist()[i]] += 1
            class_idx[np.array(self.train_data.targets).tolist()[i]].append(i)
            
        # set random number of unlearning data points for each class    
        random_counts = np.array(random_array) * class_count
        
        for i in range(len(random_counts)):
            random_counts[i] = int(random_counts[i])
        print(class_count)
        print(random_counts)  
        # random select the data points based on the above random number      
        for k in range(len(class_idx)):
            imbalance_idx = random.sample(class_idx[k], int(random_counts[k]))
            random_class_idx[k] = imbalance_idx
        
        # record the total indice of the unlearned data    
        unlearn_idx = []
        for k in range(len(class_idx)):
            unlearn_idx += random_class_idx[k]
        
        # record the unlearn data index and remove them    
        normal_idx = [idx for idx in range(0, len(self.train_data)) if idx not in unlearn_idx]    
        # sample the remain data index and remove them 
        sample_idx = random.sample(normal_idx, len(unlearn_idx))
            
            
        # obtain unlearn remain sample data from train data
        unlearn_data = Subset(self.train_data, unlearn_idx) 
        remain_data = Subset(self.train_data, normal_idx)
        sample_data = Subset(self.train_data, sample_idx)

        return unlearn_data, remain_data, sample_data

    def mislabel_data(self):
        

        class_count = np.zeros(len(set(np.array(self.train_data.targets).tolist())))
        
        # set random number of unlearning data points for each class
        class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        random_class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        
        # record the indice of data points in each class
        for i in range(len(np.array(self.train_data.targets).tolist())):
            
            class_count[np.array(self.train_data.targets).tolist()[i]] += 1
            class_idx[np.array(self.train_data.targets).tolist()[i]].append(i)
            
        # set random number of unlearning data points for each class    
        random_counts = np.array([0.6,0.6,0.6,0.6,0.6,0,0,0,0,0]) * class_count
        
        for i in range(len(random_counts)):
            random_counts[i] = int(random_counts[i])
        print(class_count)
        print(random_counts)  
        # random select the data points based on the above random number      
        for k in range(len(class_idx)):
            imbalance_idx = random.sample(class_idx[k], int(random_counts[k]))
            random_class_idx[k] = imbalance_idx
        
        # record the total indice of the unlearned data    
        unlearn_idx = []
        for k in range(len(class_idx)):
            unlearn_idx += random_class_idx[k]
        
        # record the unlearn data index and remove them    
        normal_idx = [idx for idx in range(0, len(self.train_data)) if idx not in unlearn_idx]    
        # sample the remain data index and remove them 
        sample_idx = random.sample(normal_idx, len(unlearn_idx))
        
        for idx in unlearn_idx:
            self.train_data.targets[idx] = random.sample([i for i in range(0,9) if i != self.train_data.targets[idx]],1)[0]
        
        # obtain unlearn remain sample data from train data
        unlearn_data = Subset(self.train_data, unlearn_idx) 
        remain_data = Subset(self.train_data, normal_idx)
        sample_data = Subset(self.train_data, sample_idx)

        return unlearn_data, remain_data, sample_data
    
    def random_data(self):
        

        class_count = np.zeros(len(set(np.array(self.train_data.targets).tolist())))
        
        # set random number of unlearning data points for each class
        class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        random_class_idx = [[] for i in range(len(set(np.array(self.train_data.targets).tolist())))]
        
        # record the indice of data points in each class
        for i in range(len(np.array(self.train_data.targets).tolist())):
            
            class_count[np.array(self.train_data.targets).tolist()[i]] += 1
            class_idx[np.array(self.train_data.targets).tolist()[i]].append(i)
            
        # set random number of unlearning data points for each class    
        random_counts = np.array([0,0,0,0,0,0.4,0.4,0.4,0.4,0.4]) * class_count
        
        for i in range(len(random_counts)):
            random_counts[i] = int(random_counts[i])
        print(class_count)
        print(random_counts)  
        # random select the data points based on the above random number      
        for k in range(len(class_idx)):
            imbalance_idx = random.sample(class_idx[k], int(random_counts[k]))
            random_class_idx[k] = imbalance_idx
        
        # record the total indice of the unlearned data    
        unlearn_idx = []
        for k in range(len(class_idx)):
            unlearn_idx += random_class_idx[k]
        
        # record the unlearn data index and remove them    
        normal_idx = [idx for idx in range(0, len(self.train_data)) if idx not in unlearn_idx]    
        # sample the remain data index and remove them 
        sample_idx = random.sample(normal_idx, len(unlearn_idx))
        
        # obtain unlearn remain sample data from train data
        unlearn_data = Subset(self.train_data, unlearn_idx) 
        remain_data = Subset(self.train_data, normal_idx)
        sample_data = Subset(self.train_data, sample_idx)

        return unlearn_data, remain_data, sample_data
    
    
    def construct_data(self, type, num, batch_size = 32):
                     
        if type == 'class':
            unlearn_data, remain_data, sample_data = self.class_data(num)
        elif type == 'random':
            unlearn_data, remain_data, sample_data = self.random_data()
        elif type == 'noise':
            unlearn_data, remain_data, sample_data = self.mislabel_data()
        # build data loaders   
        print(set(np.array(self.train_data.targets).tolist()))         
        loaders = {
            'train' : torch.utils.data.DataLoader(self.train_data, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=1),
            
            'unlearn'  : torch.utils.data.DataLoader(unlearn_data, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=1),
            'remain'  : torch.utils.data.DataLoader(remain_data, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=1),
            'test'  : torch.utils.data.DataLoader(self.test_data, 
                                                batch_size=batch_size, 
                                                shuffle=False, 
                                                num_workers=1),
            'sample' : torch.utils.data.DataLoader(sample_data, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=1),
        }        
        return loaders