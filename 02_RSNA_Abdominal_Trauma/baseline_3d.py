#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2
from matplotlib import pyplot as plt
import sklearn.metrics

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from resnet3d import generate_model
from timm.utils import AverageMeter

from tqdm import tqdm
import sys
import glob
import gc
import os

os.environ['CUDA_LAUNCH_BLOCKING']='1'


# # Parameters

# In[ ]:


BASE_PATH = '/home/junseonglee/01_codes/input/rsna-2023-abdominal-trauma-detection'
RESOL = 256
BATCH_SIZE = 8
LR = 0.001
N_EPOCHS = 30
train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
train_meta_df = pd.read_csv(f'{BASE_PATH}/train_meta.csv')
train_df = train_df.sort_values(by=['patient_id'])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE = 'cpu'


# # Dataset

# In[ ]:


class AbdominalCTDataset(Dataset):
    def __init__(self, meta_df):
        self.meta_df = meta_df
    
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        label = row[['bowel_healthy','bowel_injury',
                    'extravasation_healthy','extravasation_injury',
                    'kidney_healthy','kidney_low','kidney_high',
                    'liver_healthy','liver_low','liver_high',
                    'spleen_healthy','spleen_low','spleen_high', 'any_injury']]
        data_3d = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        data_3d = cv2.equalizeHist(data_3d)
        data_3d = data_3d.reshape(1, RESOL, RESOL, RESOL).astype(float)  # channel, 3D
        #avg std
        data_3d -=47.5739
        data_3d /=34.6175
        data_3d = torch.from_numpy(data_3d.astype(np.float32))
        label = label.to_numpy().astype(np.float32)
                
        #any_injury = label[-1]
        #nu_any_injury = np.zeros(2)
        #nu_any_injury[int(any_injury)]= 1
        
        #label = np.hstack([label[:-1], nu_any_injury])
        label = torch.from_numpy(label)
        return data_3d, label        

train_dataset = AbdominalCTDataset(train_meta_df)
data_3d, label = train_dataset[0]
print(label)

del train_dataset, data_3d, label
gc.collect()


# In[ ]:


'''
#normalizatino parameter
train_dataset = AbdominalCTDataset(train_meta_df)
data_3d, label = train_dataset[0]

avgs = np.zeros(len(train_dataset))
stds = np.zeros(len(train_dataset))
for i in tqdm(range(0, len(train_dataset))):
    data_3d, label = train_dataset[i]
    data_3d = data_3d.numpy()
    avgs[i] = np.average(data_3d)
    stds[i] = np.std(data_3d)
print(np.average(avgs))
print(np.average(stds))    

del train_dataset, data_3d, label, avgs, stds
gc.collect()
'''


# # Model

# In[ ]:


class AbdominalClassifier(nn.Module):
    def __init__(self, model_depth, device = DEVICE):
        super().__init__()
        self.device = device
        self.resnet3d = generate_model(model_depth = model_depth, n_input_channels = 1)
        self.flatten  = nn.Flatten()
        self.dropout  = nn.Dropout(p=0.5)
        self.softmax  = nn.Softmax(dim=1)
        self.sigmoid  = nn.Sigmoid()
        size_res_out  = 56832
        self.fc_bowel = nn.Linear(size_res_out, 2)
        self.fc_extrav= nn.Linear(size_res_out, 2)
        self.fc_kidney= nn.Linear(size_res_out, 3)
        self.fc_liver = nn.Linear(size_res_out, 3)
        self.fc_spleen= nn.Linear(size_res_out, 3)
        
        self.maxpool  = nn.MaxPool1d(5, 1)

    def forward(self, x):
        x = self.resnet3d(x)
        for i in range(0, 4):
            x[i] = self.flatten(x[i])
        x = torch.cat(x, axis = 1)
        x     = self.dropout(x)
        bowel = self.fc_bowel(x)
        extrav= self.fc_extrav(x)
        kidney= self.fc_kidney(x)
        liver = self.fc_liver(x)
        spleen= self.fc_spleen(x)

        labels = torch.cat([bowel, extrav, kidney, liver, spleen], dim = 1)

        bowel_soft = self.softmax(bowel)
        extrav_soft = self.softmax(extrav)
        kidney_soft = self.softmax(kidney)
        liver_soft = self.softmax(liver)
        spleen_soft = self.softmax(spleen)

        any_in = torch.cat([1-bowel_soft[:,0:1], 1-extrav_soft[:,0:1], 
                            1-kidney_soft[:,0:1], 1-liver_soft[:,0:1], 1-spleen_soft[:,0:1]], dim = 1) 
        any_in = self.maxpool(any_in)
        any_not_in = 1-any_in
        any_in = torch.cat([any_not_in, any_in], dim = 1)

        return labels, any_in


# In[ ]:


model = AbdominalClassifier(10)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(get_n_params(model))
del model
gc.collect()


# In[ ]:


train_dataset = AbdominalCTDataset(train_meta_df[train_meta_df['fold']!=0])
valid_dataset = AbdominalCTDataset(train_meta_df[train_meta_df['fold']==0])

train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = BATCH_SIZE, pin_memory = True, 
                          num_workers = 8, drop_last = False)

valid_loader = DataLoader(dataset = valid_dataset, shuffle = False, batch_size = BATCH_SIZE, pin_memory = True, 
                          num_workers = 8, drop_last = False)                        


# # Train

# In[ ]:


model = AbdominalClassifier(18)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
ttl_iters = N_EPOCHS * len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                                steps_per_epoch=len(train_loader), epochs = N_EPOCHS)
#scheduler = CosineAnnealingLR(optimizer, T_max=ttl_iters, eta_min=1e-6)


weights = np.ones(2)
weights[1] = 2
crit_bowel  = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))
weights[1] = 6
crit_extrav = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))
crit_any = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))

weights = np.ones((3))
weights[1] = 2
weights[2] = 4
crit_kidney = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))
crit_liver  = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))
crit_spleen = nn.CrossEntropyLoss(weight = torch.from_numpy(weights).to(DEVICE))


# In[ ]:


scaler = torch.cuda.amp.GradScaler(enabled=True)

val_metrics = np.ones(N_EPOCHS)*100

for epoch in range(0, N_EPOCHS):
    train_meters = {'loss': AverageMeter()}
    val_meters   = {'loss': AverageMeter()}
    
    model.train()
    pbar = tqdm(train_loader, leave=False)    
    for X, y in pbar:
        batch_size = X.shape[0]
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            X_out, X_any  = model(X)
            loss  = crit_bowel(X_out[:,:2], y[:,:2])
            loss += crit_extrav(X_out[:,2:4], y[:,2:4])
            loss += crit_kidney(X_out[:,4:7], y[:,4:7])
            loss += crit_liver(X_out[:,7:10], y[:,7:10])
            loss += crit_spleen(X_out[:,10:13], y[:,10:13])

            loss += crit_any(X_any,  torch.cat([torch.ones(batch_size, 1).to(DEVICE)- y[:,13:14],y[:,13:14]], dim = 1))  
            loss /= 6
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()          

        trn_loss = loss.item()      
        train_meters['loss'].update(trn_loss, n=X.size(0))     
        pbar.set_description(f'Train loss: {trn_loss}')   
    print('Epoch {:d} / trn/loss={:.4f}'.format(epoch+1, train_meters['loss'].avg))    
    
    softmax = nn.Softmax(dim=1)

    X_outs=[]
    ys=[]
    model.eval()
    for X, y in tqdm(valid_loader, leave=False):
        batch_size = X.shape[0]        
        X, y = X.to(DEVICE), y.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=True):        
            with torch.no_grad():                 
                X_out, X_any = model(X)                                       
                loss  = crit_bowel(X_out[:,:2], y[:,:2])
                loss += crit_extrav(X_out[:,2:4], y[:,2:4])
                loss += crit_kidney(X_out[:,4:7], y[:,4:7])
                loss += crit_liver(X_out[:,7:10], y[:,7:10])
                loss += crit_spleen(X_out[:,10:13], y[:,10:13])
                loss += crit_any(X_any,  torch.cat([torch.ones(batch_size, 1).to(DEVICE)- y[:,13:14],y[:,13:14]], dim = 1))                                
                loss /=6
                val_loss = loss.item()
        val_meters['loss'].update(val_loss, n=X.size(0))
              

    metric = val_meters['loss'].avg
    print('Epoch {:d} / val/loss={:.4f}'.format(epoch+1, metric))   
    
    #Save the best model    
    if(metric < np.min(val_metrics)):
        try:
            os.makedirs(f'{BASE_PATH}/weights')
        except:
            a = 1
        best_metric = metric
        print(f'Best val_metric {best_metric} at epoch {epoch+1}!')
        torch.save(model, f'{BASE_PATH}/weights/best.pt')    
    
    val_metrics[epoch] = val_meters['loss'].avg


# In[ ]:




