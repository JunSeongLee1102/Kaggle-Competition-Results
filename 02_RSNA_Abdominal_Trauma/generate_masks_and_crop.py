#!/usr/bin/env python
# coding: utf-8

# # Modification of the original code
# I just took the code of the winner of the RSNA 2022 cervical spine fracture detection competition.  
# Link: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage1

# # 1st Place Solution Training 3D Semantic Segmentation (Stage1)
# 
# Hi all,
# 
# I'm very exciting to writing this notebook and the summary of our solution here.
# 
# This is FULL version of training my final models (stage1), using resnet18d as backbone, unet as decoder and using 128x128x128 as input.
# 
# NOTE: **You need to run this code locally because the RAM is not enough here.**
# 
# NOTE2: **It is highly recommended to pre-process the 3D semantic segmentation training data first and save it locally, which can greatly speed up the loading of the data.**
# 
# My brief summary of winning solution: https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/362607
# 
# * Train Stage1 Notebook: This notebook
# * Train Stage2 (Type1) Notebook: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage2-type1
# * Train Stage2 (Type2) Notebook: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage2-type2
# * Inference Notebook: https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-inference
# 
# **If you find these notebooks helpful please upvote. Thanks! **

# In[1]:


#!pip -q install monai
#!pip -q install segmentation-models-pytorch==0.2.1


# In[2]:


DEBUG = False

import os
import sys
#sys.path = [
#    '../input/covn3d-same',
#] + sys.path


# In[3]:


import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import pydicom
import dicomsdl
import argparse
import warnings
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import gzip
import pickle
from joblib import Parallel, delayed
import lz4.frame
import mgzip

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from monai.transforms import Resize
import  monai.transforms as transforms

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 20, 8
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

sys.path.append('./lib_models')


# # Config

# In[4]:


RESOL = 128

BASE_PATH = '/home/junseonglee/Desktop/01_codes/inputs/rsna-2023-abdominal-trauma-detection'
MASK_SAVE_PATH = f'{BASE_PATH}/mask_preprocessed'
MASK_VALID_PATH = f'{BASE_PATH}/mask_validation'
TRAIN_PATH = f'{BASE_PATH}/train_images'
N_PREPROCESS_CHUNKS = 12

kernel_type = 'timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep'
load_kernel = None
load_last = True
n_blocks = 4
n_folds = 5
backbone = 'resnet18d'

image_sizes = [128, 128, 128]
R = Resize(image_sizes)

init_lr = 3e-3
batch_size = 4
drop_rate = 0.
drop_path_rate = 0.
loss_weights = [1, 1]
p_mixup = 0.1

data_dir = '../input/rsna-2022-cervical-spine-fracture-detection'
use_amp = True
num_workers = 24
out_dim = 5

n_epochs = 1000

log_dir = f'{BASE_PATH}/seg_models_backup'
model_dir = f'{BASE_PATH}/seg_models_backup'
seg_inference_dir = f'{BASE_PATH}/seg_infer_results'
cropped_img_dir   = f'{BASE_PATH}/3d_preprocessed_crop'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(MASK_VALID_PATH, exist_ok=True)
os.makedirs(seg_inference_dir, exist_ok = True)
os.makedirs(cropped_img_dir, exist_ok = True)



# In[5]:


transforms_valid = transforms.Compose([
])


# # DataFrame

# In[6]:


df_train = pd.read_csv(f'{BASE_PATH}/train_meta.csv')
mask_paths = []
cropped_paths = []
for i in range(0, len(df_train)):
    row = df_train.iloc[i]
    file_name = row['path'].split('/')[-1]
    mask_paths.append(f'{seg_inference_dir}/{file_name}')
    cropped_paths.append(f'{cropped_img_dir}/{file_name}')
df_train['mask_path'] = mask_paths
df_train['cropped_path'] = cropped_paths
df_train.tail()

df_train.to_csv(f'{BASE_PATH}/train_meta.csv', index = False)


# # Dataset

# In[7]:


def compress(name, data):
    with gzip.open(name, 'wb') as f:
        pickle.dump(data, f)

def decompress(name):
    with gzip.open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def compress_fast(name, data):
    data = torch.from_numpy(data)
    torch.save(data, name)    
    #np.save(name, data)

def decompress_fast(name):
    #data = np.load(f'{name}.npy')
    data = torch.load(name)
    return data


def save_pickle(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data    

def load_sample(row, has_mask=True):
    image = decompress(row['img_path'])[None]

    if has_mask:
        mask = load_pickle(row['mask_path'])
        
        return image, mask
    else:
        return image



class SEGDataset(Dataset):
    def __init__(self, df, mode, transform):

        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        image = decompress(row['path'])[None]
        image = torch.from_numpy(image).to(torch.float32)
        #file_name = row['path'].split('/')[-1]
        #save_path = f'{seg_inference_dir}/{file_name}'
        save_path = row['mask_path']

        return image, save_path


# # Model

# In[8]:


class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=1,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 1, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], out_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features


# In[9]:


from timm.models.layers.conv2d_same import Conv2dSame
from conv3d_same import Conv3dSame


def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


m = TimmSegModel(backbone)
m = convert_3d(m)
m(torch.rand(1, 1, 128,128,128)).shape


# # Valid func

# In[10]:


def infer_func(model, loader_valid):
    model.eval()
    valid_loss = []
    outputs = []
    ths = [0.2]
    batch_metrics = [[]]
    bar = tqdm(loader_valid)

    counter = 0
    
    with torch.no_grad():
        for images, save_paths in bar:
            images = images.cuda()

            logits = model(images)

            for thi, th in enumerate(ths):
                pred = (logits.sigmoid() > th).float().detach()
                for i in range(logits.shape[0]):                    
                    y_pred = (logits[i].sigmoid().cpu().numpy()+0.1).astype(np.uint8)
                    compress(save_paths[i], y_pred)



# # Training

# In[11]:


def run(fold):

    log_file = os.path.join(log_dir, f'{kernel_type}.txt')
    model_file = os.path.join(model_dir, f'{kernel_type}_fold{fold}_best.pth')


    dataset_train = SEGDataset(df_train, 'valid', transform=transforms_valid)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=4, shuffle=False, num_workers=4)

    model = TimmSegModel(backbone, pretrained=True)
    model = convert_3d(model)

    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    print(len(dataset_train))

    infer_func(model, loader_train)

    del model
    torch.cuda.empty_cache()
    gc.collect()


# In[12]:


#run(0)


# # Postprocss to get crop regions

# In[13]:


#The order of the crop region data format
#Z start/end, Y start/end, X start/end for each mask channels + total region for the extravasation prediction
def calc_crop_region(mask):
    crop_range = np.zeros((6, 6))
    crop_range[:,::2]=10000
    mask_z = np.max(mask, axis = (2, 3)).astype(bool)
    mask_y = np.max(mask, axis = (1, 3)).astype(bool)
    mask_x = np.max(mask, axis = (1, 2)).astype(bool)
    
    template_range = np.arange(0, RESOL)

    for mi in range(0, 5):
        zrange = template_range[mask_z[mi]]
        yrange = template_range[mask_y[mi]]
        xrange = template_range[mask_x[mi]]
        # For incomplete organ
        if(len(zrange)==0):
            zrange = template_range.copy()
            yrange = template_range.copy()
            xrange = template_range.copy()

        crop_range[mi] = np.min(zrange), np.max(zrange)+1, np.min(yrange), np.max(yrange)+1, np.min(xrange), np.max(xrange)+1

    crop_range[5] = np.min(crop_range[:5, 0]), np.max(crop_range[:5, 1]), np.min(crop_range[:5, 2]), \
                    np.max(crop_range[:5, 3]), np.min(crop_range[:5,4]), np.max(crop_range[:5, 5])
    
    crop_range[:,:2]/=len(mask_z[0])
    crop_range[:,2:4]/=len(mask_y[0])
    crop_range[:,4:6]/=len(mask_x[0])

    # Then make extravasation (# 5 mask) to reference one and convert other mask's crop respective to it
    # --> To minimize the loading size due to speed issue.
    zmin, rel_zrange = crop_range[5,0], crop_range[5,1]-crop_range[5,0]
    ymin, rel_yrange = crop_range[5,2], crop_range[5,3]-crop_range[5,2]
    xmin, rel_xrange = crop_range[5,4], crop_range[5,5]-crop_range[5,4]

    crop_range[:5,:2] = (crop_range[:5,:2]-zmin)/rel_zrange
    crop_range[:5,2:4] = (crop_range[:5,2:4]-ymin)/rel_yrange
    crop_range[:5,4:6] = (crop_range[:5,4:6]-xmin)/rel_xrange

    return crop_range

def crop_resize_avg_and_std_3d(data, region):
    shapes = np.shape(data)
    region[:2]*=shapes[0]
    region[2:4]*=shapes[1]
    region[4:6]*=shapes[2]
    region = region.astype(int)

    cropped = data[region[0]:region[1], region[2]:region[3], region[4]:region[5]]
    slices = []
    for i in range(0, len(cropped)):
        slices.append(cv2.resize(cropped[i], (RESOL, RESOL))[None])
    
    slices = np.vstack(slices)
    
    resized_cropped = np.zeros((RESOL, RESOL, RESOL))
    for i in range(0, len(slices[0,0])):
        resized_cropped[:,:,i] = cv2.resize(slices[:,:,i], (RESOL, RESOL))
    
    std = np.std(resized_cropped)
    avg = np.average(resized_cropped)
    resized_cropped = (resized_cropped-avg)/std
    resized_cropped = resized_cropped.astype(np.float32)

    del cropped, slices
    gc.collect()
    return resized_cropped

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    pixel_rep = dcm.PixelRepresentation

    return pixel_array

# Read each slice and stack them to make 3d data
def process_3d_crop(save_path, mask_path, data_path = TRAIN_PATH):
    tmp = save_path.split('/')[-1][:-4]
    tmp = tmp.split('_')
    patient, study = int(tmp[0]), int(tmp[1])
    
    mask = decompress(mask_path)
    crop_regions = calc_crop_region(mask)
    absolute_crop = crop_regions[5].copy() # To load minimum pixels...

    crop_regions[5] = 0, 1, 0, 1, 0, 1

    imgs = {}    
    
    for f in sorted(glob.glob(data_path + f'/{patient}/{study}/*.dcm')):      
        pixel_rep = 0
        bit_shift = 0
        dtype = 0
        try:            
            dicom = pydicom.dcmread(f)        
            img = standardize_pixel_array(dicom)
            img_shape = np.shape(img)
            xy_crop_range = absolute_crop[2:].copy()   
            xy_crop_range[0:2]*=img_shape[0]
            xy_crop_range[2:4]*=img_shape[1]            
            xy_crop_range = xy_crop_range.astype(int)
            img = img.astype(float)
            break
        except:
            continue
            
    for f in sorted(glob.glob(data_path + f'/{patient}/{study}/*.dcm')):
        #For the case that some of the image can't be read -> error without this though don't know why  
        img = dicomsdl.open(f).pixelData(storedvalue=True)[xy_crop_range[0]:xy_crop_range[1], xy_crop_range[2]:xy_crop_range[3]]
        img = img.astype(float)
        
        #dicom = pydicom.dcmread(f)
        #img = standardize_pixel_array(dicom).astype(float)
        #ind = int((f.split('/')[-1])[:-4])
        pos_z = -int((f.split('/')[-1])[:-4])
        imgs[pos_z] = img


    #sample_z = np.linspace(0, len(imgs)-1, RESOL, dtype=int)

    imgs_3d = []
    n_imgs = len(imgs)    
    z_crop_range= (absolute_crop[0:2]*n_imgs).astype(int)

    #print(z_crop_range)
    for i, k in enumerate(sorted(imgs.keys())):
        #if i in sample_z:
        if(i >= z_crop_range[0] and i < z_crop_range[1]):
            img = imgs[k]
            imgs_3d.append(img[None])
        
    imgs_3d = np.vstack(imgs_3d)
    imgs_3d = ((imgs_3d - imgs_3d.min()) / (imgs_3d.max() - imgs_3d.min()))

    if dicom.PhotometricInterpretation == "MONOCHROME1":
        imgs_3d = 1.0 - imgs_3d

    #Loaded original imgs_3d    
    processed_img_3d = np.zeros((6, RESOL, RESOL, RESOL))

    for i in range(0, 6):     
        #To deal with almost not detected slices
        try:   
            processed_img_3d[i] = crop_resize_avg_and_std_3d(imgs_3d, crop_regions[i])
        except:
            processed_img_3d[i] = crop_resize_avg_and_std_3d(imgs_3d, np.array([0, 1, 0, 1, 0, 1]))

    #here to
    #gzip too slow maybe I should divide the inference process to chunks or do not save in the inference notebooks\
    
    
    processed_img_3d = (processed_img_3d).astype(np.float16)
    #save_png(save_path, processed_img_3d)
    compress_fast(save_path, processed_img_3d)                      
    #save_pickle(save_path, processed_img_3d)

    del imgs, img
    gc.collect()



# crop_regions = np.zeros((len(df_train), 6))
# for i in range(0, len(df_train)):
#     row = df_train.iloc[i]
#     #file_name = row['path'].split('/')[-1]
#     #save_path = f'{seg_inference_dir}/{file_name}'
#     mask = decompress(row['mask_path'])
#     crop_region = calc_crop_region(mask)
#     print(crop_region)
#     #plt.imshow(mask_file[4,:,60,:])
#     if (i==0):
#         break
# 
# 

# In[14]:


# Preprocess dataset
rng_samples = np.linspace(0, len(df_train), N_PREPROCESS_CHUNKS+1, dtype = int)
def process_3d_wrapper(process_ind, rng_samples = rng_samples, train_meta_df = df_train):
    for i in tqdm(range(rng_samples[process_ind], rng_samples[process_ind+1])):
        if not os.path.isfile(f"{df_train.iloc[i]['cropped_path']}"):
            process_3d_crop(train_meta_df.iloc[i]['cropped_path'], train_meta_df.iloc[i]['mask_path'])     
        #For error correction
        
        else:
            try:
                #data = decompress_fast(f"{train_meta_df.iloc[i]['cropped_path']}")
                data = decompress_fast(f"{train_meta_df.iloc[i]['cropped_path']}")                
                if (len(data) != 6*RESOL*RESOL*RESOL):
                    dummy_fail_function()
            except: 
                process_3d_crop(train_meta_df.iloc[i]['cropped_path'], train_meta_df.iloc[i]['mask_path'])  
        


# In[15]:


get_ipython().run_cell_magic('time', '', '#For debugging\n#for i in range(0, N_PREPROCESS_CHUNKS):\n#for i in range(0, len(df_train)):\n#     process_3d_wrapper(i)\nParallel(n_jobs = N_PREPROCESS_CHUNKS)(delayed(process_3d_wrapper)(i) for i in range(N_PREPROCESS_CHUNKS))\n')


# In[ ]:




