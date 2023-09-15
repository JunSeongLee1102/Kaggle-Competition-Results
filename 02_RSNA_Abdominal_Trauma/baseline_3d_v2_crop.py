#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from tqdm import tqdm
import sys
import glob
import gc
import os
sys.path.append('./lib_models')

import pandas as pd
import numpy as np
import scipy as sp
import cv2
from matplotlib import pyplot as plt
import sklearn.metrics
import warnings
import pydicom
import dicomsdl
from joblib import Parallel, delayed
#import h5py
import bz2
import pickle
import gzip
import mgzip
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from multiprocessing import Pool
import lz4.frame


import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn

import segmentation_models_pytorch as smp
import timm
from timm.utils import AverageMeter
from timm.models import resnet

#sys.path.append('/home/junseonglee/Desktop/01_codes/inputs/rsna-2023-abdominal-trauma-detection')
import timm_new

from monai.transforms import Resize
import  monai.transforms as transforms

import wandb
sys.path.append('./lib_models')

wandb.login(key = '585f58f321685308f7933861d9dde7488de0970b')
#warnings.filterwarnings('ignore', category=UserWarning)
#os.environ['CUDA_LAUNCH_BLOCKING']='1'


# In[2]:


timm_new.__version__


# # Parameters

# In[3]:


backbone = 'timm/resnet10t.c3_in1k'

IS_WANDB = True
PROJECT_NAME = 'RSNA_ABTD'
GROUP_NAME= 'backbone_test'
RUN_NAME=   f'{backbone}'

if not IS_WANDB:
    PROJECT_NAME = 'Dummy_Project'

BASE_PATH  = '/home/junseonglee/Desktop/01_codes/inputs/rsna-2023-abdominal-trauma-detection'
TRAIN_PATH = f'{BASE_PATH}/train_images'
DATA_PATH = f'{BASE_PATH}/3d_preprocessed'

seg_inference_dir = f'{BASE_PATH}/seg_infer_results'
cropped_img_dir   = f'{BASE_PATH}/3d_preprocessed_crop'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

RESOL = 128
UP_RESOL = 128
N_CHANNELS = 6
BATCH_SIZE = 24
ACCUM_STEPS = 1
N_WORKERS  = 10
LR = 0.001
N_EPOCHS = 100
EARLY_STOP_COUNT = 20
N_FOLDS  = 5
N_PREPROCESS_CHUNKS = 12
train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
train_df = train_df.sort_values(by=['patient_id'])
n_blocks = 4
drop_rate = 0.2
drop_path_rate = 0.2
p_mixup = 0.0


#backbone = 'efficientnet_b1'



wandb_config = {
    'RESOL': RESOL,
    'BACKBONE': backbone,
    'N_CHANNELS': N_CHANNELS,
    'N_EPOCHS': N_EPOCHS,
    'N_FOLDS': N_FOLDS,
    'EARLY_STOP_COUNT': EARLY_STOP_COUNT,
    'BATCH_SIZE': BATCH_SIZE,    
    'LR': LR,
    'N_EPOCHS': N_EPOCHS,
    'DROP_RATE': drop_rate,
    'DROP_PATH_RATE': drop_path_rate,
    'MIXUP_RATE': p_mixup
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE
#DEVICE = 'cpu'


# # Data split

# In[4]:


train_df = pd.read_csv(f'{BASE_PATH}/train.csv')
train_meta = pd.read_csv(f'{BASE_PATH}/train_series_meta.csv')
train_df = train_df.sort_values(by=['patient_id'])
train_df

TRAIN_PATH = BASE_PATH + "/train_images/"
n_chunk = 8
patients = os.listdir(TRAIN_PATH)
n_patients = len(patients)
rng_patients = np.linspace(0, n_patients+1, n_chunk+1, dtype = int)
patients_cts = glob.glob(f'{TRAIN_PATH}/*/*')
n_cts = len(patients_cts)
patients_cts_arr = np.zeros((n_cts, 2), int)
data_paths=[]
for i in range(0, n_cts):
    patient, ct = patients_cts[i].split('/')[-2:]
    patients_cts_arr[i] = patient, ct
    data_paths.append(f'{BASE_PATH}/3d_preprocessed/{patients_cts_arr[i,0]}_{patients_cts_arr[i,1]}.pkl')
TRAIN_IMG_PATH = BASE_PATH + '/processed' 

#Generate tables for training
train_meta_df = pd.DataFrame(patients_cts_arr, columns = ['patient_id', 'series'])

#5-fold splitting
train_df['fold'] = 0
labels = train_df[['bowel_healthy','bowel_injury',
                    'extravasation_healthy','extravasation_injury',
                    'kidney_healthy','kidney_low','kidney_high',
                    'liver_healthy','liver_low','liver_high',
                    'spleen_healthy','spleen_low','spleen_high',
                    'any_injury']].to_numpy()

mskf = MultilabelStratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
counter = 0
for train_index, test_index in mskf.split(np.ones(len(train_df)), labels):
    for i in range(0, len(test_index)):
        train_df['fold'][test_index[i]] = counter
    counter+=1

train_meta_df = train_meta_df.join(train_df.set_index('patient_id'), on='patient_id')
train_meta_df['path']=data_paths

#For mask paths
mask_paths = []
cropped_paths = []
for i in range(0, len(train_meta_df)):
    row = train_meta_df.iloc[i]
    file_name = row['path'].split('/')[-1]
    mask_paths.append(f'{seg_inference_dir}/{file_name}')
    cropped_paths.append(f'{cropped_img_dir}/{file_name}')
train_meta_df['mask_path'] = mask_paths
train_meta_df['cropped_path'] = cropped_paths

train_meta_df.to_csv(f'{BASE_PATH}/train_meta.csv', index = False)
np.unique(train_df['fold'].to_numpy(), return_counts = True)


# # Dataset

# In[5]:


def compress(name, data):
    with gzip.open(name, 'wb') as f:
        pickle.dump(data, f)

def decompress(name):
    with gzip.open(name, 'rb') as f:
        data = pickle.load(f)
    return data


def compress_fast(name, data):  
    np.save(name, data)

def decompress_fast(name):
    data = np.load(f'{name}.npy')
    return data

def save_png(name, data):
    cv2.imwrite(f'{name}.png', data)


def load_png(name):
    data = cv2.imread(f'{name}.png', cv2.IMREAD_UNCHANGED)
    return data


# In[6]:


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


# In[7]:


def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Source : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/discussion/427217
    """
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    pixel_rep = dcm.PixelRepresentation

    return pixel_array

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
    #compress(save_path, processed_img_3d)
    compress_fast(save_path, processed_img_3d)                      
    #save_pickle(save_path, processed_img_3d)

    del imgs, img
    gc.collect()
    return processed_img_3d



# In[8]:


# Preprocess dataset
rng_samples = np.linspace(0, len(train_meta_df), N_PREPROCESS_CHUNKS+1, dtype = int)
def process_3d_wrapper(process_ind, rng_samples = rng_samples, train_meta_df = train_meta_df):
    for i in tqdm(range(rng_samples[process_ind], rng_samples[process_ind+1])):
        if not os.path.isfile(train_meta_df.iloc[i]['path']):
            process_3d(train_meta_df.iloc[i]['path'])


# In[9]:


class AbdominalCTDataset(Dataset):
    def __init__(self, meta_df, is_train = True, transform_set = None):
        self.meta_df = meta_df
        self.is_train = is_train
        self.transform_set = transform_set
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        label = row[['bowel_healthy','bowel_injury',
                    'extravasation_healthy','extravasation_injury',
                    'kidney_healthy','kidney_low','kidney_high',
                    'liver_healthy','liver_low','liver_high',
                    'spleen_healthy','spleen_low','spleen_high', 'any_injury']]

        #To avoid loading issue when applying multiprocessing to the unzip module
        try:
        #data_3d = decompress_fast(row['cropped_path'])  
            data_3d = decompress_fast(row['cropped_path'])            
            #data_3d = data_3d.reshape(6, RESOL, RESOL, RESOL).astype(np.float32)  # channel, 3D             
        except:                
            while(1):
                try:
                    data_3d = decompress_fast(row['cropped_path'])        
                    break                        
                except:
                    continue
                
            #data_3d = process_3d_crop(row['cropped_path'], row['mask_path'])           
            #data_3d = data_3d.reshape(6, RESOL, RESOL, RESOL).astype(np.float32)  # channel, 3D                 

        data_3d = torch.from_numpy(data_3d)
        if self.transform_set is not None:
            data_3d = self.transform_set({'image':data_3d})
            data_3d = data_3d['image']
        #augmentation  
        #if self.is_train:            
        #    random_angle = np.random.rand(1)[0]*360.0-180.0
        #    data_3d = transforms.functional.rotate(data_3d, random_angle, transforms.InterpolationMode.BILINEAR)
            

        label = label.to_numpy().astype(np.float32)
                
        label = torch.from_numpy(label)
        return data_3d, label        

train_dataset = AbdominalCTDataset(train_meta_df)
data_3d, label = train_dataset[0]
print(label)

del train_dataset, data_3d, label
gc.collect()


# In[10]:


import timm.models.layers


# # Model

# In[11]:


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


#m = TimmSegModel(backbone)
#m = convert_3d(m)
#out = m(torch.rand(1, 1, 128,128,128))
#for i in range(0, len(out)):
#    print(out[i].shape)


# In[12]:


class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm_new.create_model(
            backbone,
            in_chans=N_CHANNELS,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, N_CHANNELS, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        
        #if segtype == 'unet':
        #    self.decoder = smp.unet.decoder.UnetDecoder(
        #        encoder_channels=encoder_channels[:n_blocks+1],
        #        decoder_channels=decoder_channels[:n_blocks],
        #        n_blocks=n_blocks,
        #    )
        self.avgpool = nn.AvgPool2d(5, 4, 2)
        
        [_.shape[1] for _ in g]
        self.convs1x1 = nn.ModuleList()    
        self.batchnorms = nn.ModuleList()    
        self.batchnorms13 = nn.ModuleList()
        for i in range(0, len(g)):
            self.convs1x1.append(nn.Conv2d(g[i].shape[1], 13, 1))
            self.batchnorms.append(nn.BatchNorm2d(g[i].shape[1]))
            self.batchnorms13.append(nn.BatchNorm2d(13))

        del g
        gc.collect()
    def forward(self,x):
        global_features = self.encoder(x)[:n_blocks]        
        for i in range(0, len(global_features)):
            #global_features[i] = self.batchnorms[i](global_features[i])
            global_features[i] = self.convs1x1[i](global_features[i])
            #global_features[i] = self.batchnorms13[i](global_features[i])
            
            #global_features[i] = self.avgpool(global_features[i])
        return global_features
        #seg_features = self.decoder(*global_features)
        #seg_features = self.segmentation_head(seg_features)


# In[13]:


class AbdominalClassifier(nn.Module):
    def __init__(self, device = DEVICE):
        super().__init__()
        self.device = device
        self.upsample = torch.nn.Upsample(size = [UP_RESOL, UP_RESOL, UP_RESOL])
        self.resnet3d = TimmSegModel(backbone)
        self.resnet3d = convert_3d(self.resnet3d)
        #self.resnet3d.load_state_dict(torch.load(f'{BASE_PATH}/seg_models_backup/timm3d_res18d_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_bs4_lr3e4_20x50ep_fold0_best.pth'), strict=False)
        self.flatten  = nn.Flatten()
        self.dropout  = nn.Dropout(p=0.5)
        self.softmax  = nn.Softmax(dim=1)        
        self.maxpool  = nn.MaxPool1d(5, 1)
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.upsample(x)
        x = self.resnet3d(x)
        pooled_features = []
        for i in range(0, len(x)):        
            #pooled_features.append(self.flatten(torch.sum(x[i], dim = 1)))
            pooled_features.append(torch.reshape(torch.mean(x[i], dim = (2, 3, 4)), (batch_size, 13, 1)))
            
        x = torch.cat(pooled_features, dim=2)
        labels = torch.mean(x, dim=2)
        
        bowel_soft = self.softmax(labels[:,0:2])
        extrav_soft = self.softmax(labels[:,2:4])
        kidney_soft = self.softmax(labels[:,4:7])
        liver_soft = self.softmax(labels[:,7:10])
        spleen_soft = self.softmax(labels[:,10:13])

        any_in = torch.cat([1-bowel_soft[:,0:1], 1-extrav_soft[:,0:1], 
                            1-kidney_soft[:,0:1], 1-liver_soft[:,0:1], 1-spleen_soft[:,0:1]], dim = 1) 
        any_in = self.maxpool(any_in)
        any_not_in = 1-any_in
        any_in = torch.cat([any_not_in, any_in], dim = 1)

        return labels, any_in


# In[14]:


model = AbdominalClassifier()

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


# # Train

# In[15]:


model = AbdominalClassifier()
model.to(DEVICE)


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


# In[16]:


def normalize_to_one(tensor):
    norm = torch.sum(tensor, 1)
    for i in range(0, tensor.shape[1]):
        tensor[:,i]/=norm
    return tensor

def apply_softmax_to_labels(X_out):
    softmax = nn.Softmax(dim=1)

    X_out[:,:2]    = normalize_to_one(softmax(X_out[:,:2]))
    X_out[:,2:4]   = normalize_to_one(softmax(X_out[:,2:4]))
    X_out[:,4:7]   = normalize_to_one(softmax(X_out[:,4:7]))
    X_out[:,7:10]  = normalize_to_one(softmax(X_out[:,7:10]))
    X_out[:,10:13] = normalize_to_one(softmax(X_out[:,10:13]))

    return X_out

def calculate_score(X_outs, ys, step = 'train'):
    X_outs = X_outs.astype(np.float64)
    ys     = ys.astype(np.float64)

    bowel_weights  =  ys[:,0] + 2*ys[:,1]
    extrav_weights = ys[:,2] + 6*ys[:,3]
    kidney_weights = ys[:,4] + 2*ys[:,5] + 4*ys[:,6]
    liver_weights  = ys[:,7] + 2*ys[:,8] + 4*ys[:,9]
    spleen_weights = ys[:,10] + 2*ys[:,11] + 4*ys[:,12]
    any_in_weights = ys[:,13] + 6*ys[:,14]
    

    bowel_loss  = sklearn.metrics.log_loss(ys[:,:2], X_outs[:,:2], sample_weight = bowel_weights)
    extrav_loss = sklearn.metrics.log_loss(ys[:,2:4], X_outs[:,2:4], sample_weight = extrav_weights)
    kidney_loss = sklearn.metrics.log_loss(ys[:,4:7], X_outs[:,4:7], sample_weight = kidney_weights)
    liver_loss  = sklearn.metrics.log_loss(ys[:,7:10], X_outs[:,7:10], sample_weight = liver_weights)
    spleen_loss = sklearn.metrics.log_loss(ys[:,10:13], X_outs[:,10:13], sample_weight = spleen_weights)
    any_in_loss = sklearn.metrics.log_loss(ys[:,13:15], X_outs[:,13:15], sample_weight =  any_in_weights)
    
    avg_loss = (bowel_loss + extrav_loss + kidney_loss + liver_loss + spleen_loss + any_in_loss)/6

    losses= {f'{step}_bowel_metric': bowel_loss, f'{step}_extrav_metric': extrav_loss, f'{step}_kidney_metric': kidney_loss,
             f'{step}_liver_metric': liver_loss, f'{step}_spleen_metric': spleen_loss, f'{step}_any_in_metric': any_in_loss,
             f'{step}_avg_metric': avg_loss}

    wandb.log(losses)
    return avg_loss

def calculate_loss(X_out, X_any, y):
    batch_size = X_out.shape[0]
    bowel_loss  = crit_bowel(X_out[:,:2], y[:,:2])
    extrav_loss = crit_extrav(X_out[:,2:4], y[:,2:4])
    kidney_loss = crit_kidney(X_out[:,4:7], y[:,4:7])
    liver_loss  = crit_liver(X_out[:,7:10], y[:,7:10])
    spleen_loss = crit_spleen(X_out[:,10:13], y[:,10:13])
    any_in_loss = crit_any(X_any,  torch.cat([torch.ones(batch_size, 1).to(DEVICE)- y[:,13:14],y[:,13:14]], dim = 1))
    
    avg_loss = (bowel_loss + extrav_loss + kidney_loss + liver_loss + spleen_loss + any_in_loss)/6
    return bowel_loss, extrav_loss, kidney_loss, liver_loss, spleen_loss, any_in_loss, avg_loss


# In[17]:


def mixup(inputs, truth, clip=[0, 1]):
    indices = torch.randperm(inputs.size(0))
    shuffled_input = inputs[indices]
    shuffled_labels = truth[indices]

    lam = np.random.uniform(clip[0], clip[1])
    inputs = inputs * lam + shuffled_input * (1 - lam)
    return inputs, truth, shuffled_labels, lam

transforms_train = transforms.Compose([
    transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
    transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    transforms.RandAffined(keys=["image"], translate_range=[int(x*y) for x, y in zip([RESOL, RESOL, RESOL], [0.3, 0.3, 0.3])], padding_mode='zeros', prob=0.7),
    transforms.RandGridDistortiond(keys=("image"), prob=0.5, distort_limit=(-0.01, 0.01), mode="nearest"),    
    #monai.transforms.RandGibbsNoise(prob=0.1, alpha=(0.0, 1.0)),
])

transforms_valid = transforms.Compose([
])


# In[18]:


wandb.init(
    config = wandb_config,
    project= PROJECT_NAME,
    group  = GROUP_NAME,
    name   = RUN_NAME,
    dir    = BASE_PATH)

backbone = backbone.replace('/', '_')

if __name__ == '__main__':
    train_dataset = AbdominalCTDataset(train_meta_df[train_meta_df['fold']!=0], is_train = True, transform_set  = transforms_train)
    valid_dataset = AbdominalCTDataset(train_meta_df[train_meta_df['fold']==0], is_train = False, transform_set = transforms_valid)

        
    
    train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size = BATCH_SIZE, pin_memory = False, 
                            num_workers = N_WORKERS, drop_last = False)

    valid_loader = DataLoader(dataset = valid_dataset, shuffle = False, batch_size = BATCH_SIZE, pin_memory = False, 
                            num_workers = N_WORKERS//2, drop_last = False)     
    
    ttl_iters = N_EPOCHS * len(train_loader)
    
    #gradient accumulation for stability of the training
    accum_len = int(np.ceil(len(train_loader)/ACCUM_STEPS)+0.001)
    accum_points = np.zeros(accum_len, int)
    accum_scale  = np.zeros(accum_len, int)
    
    prev_step = -1
    for i in range(0, accum_len):
        accum_points[i] = min(prev_step+ACCUM_STEPS, len(train_loader)-1)
        accum_scale[i]  = accum_points[i] - prev_step
        prev_step = accum_points[i]
    
    print(accum_points)
    print(accum_scale)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    n_batch_iters = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, 
                                                    steps_per_epoch= n_batch_iters, epochs = N_EPOCHS)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    gc.collect()

    val_metrics = np.ones(N_EPOCHS)*100

    for epoch in tqdm(range(0, N_EPOCHS), leave = False):     

        train_meters = {'loss': AverageMeter()}
        val_meters   = {'loss': AverageMeter()}
        
        model.train()
        #pbar = tqdm(train_loader, leave=False)  

        X_outs=[]
        ys=[]
        accum_counter = 0
        counter = 0
        last_count_on = False
        for X, y in train_loader:
            current_lr = float(scheduler.get_last_lr()[0])
            wandb.log({'lr': current_lr})
            
            batch_size = X.shape[0]
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):  
                X_out, X_any  = model(X)
                do_mixup = False
                if np.random.random() < p_mixup:
                    do_mixup = True
                    X, y, labels_shuffled, lam = mixup(X, y)                
                
                bowel_loss, extrav_loss, kidney_loss, liver_loss, spleen_loss, any_in_loss, avg_loss = calculate_loss(X_out, X_any, y)
                if do_mixup:
                    bowel_loss2, extrav_loss2, kidney_loss2, liver_loss2, spleen_loss2, any_in_loss2, avg_loss2 = calculate_loss(X_out, X_any, labels_shuffled)
                    bowel_loss  = bowel_loss * lam  + bowel_loss2 * (1 - lam)
                    extrav_loss = extrav_loss * lam  + extrav_loss2 * (1 - lam)
                    kidney_loss = kidney_loss * lam  + kidney_loss2 * (1 - lam)         
                    liver_loss  = liver_loss * lam  + liver_loss2 * (1 - lam) 
                    spleen_loss = spleen_loss * lam  + spleen_loss2 * (1 - lam) 
                    any_in_loss = any_in_loss * lam  + any_in_loss2 * (1 - lam) 
                    avg_loss = avg_loss * lam  + avg_loss2 * (1 - lam)       
                    
                step = 'train'
                wandb.log({f'{step}_bowel_loss': bowel_loss.item(),
                           f'{step}_extrav_loss': extrav_loss.item(),
                           f'{step}_kidney_loss': kidney_loss.item(),
                           f'{step}_liver_loss': liver_loss.item(),
                           f'{step}_spleen_loss': spleen_loss.item(),
                           f'{step}_any_loss': any_in_loss.item(),
                           f'{step}_avg_loss': avg_loss.item()
                           })
                
                scaler.scale(avg_loss/accum_scale[accum_counter]).backward()
                if(counter==accum_points[accum_counter]):
                    scaler.step(optimizer)
                    scheduler.step()
                    scaler.update()    
                    accum_counter+=1
                
            counter+=1                   

            #Metric calculation
            y_any = torch.cat([torch.ones(batch_size, 1).to(DEVICE)- y[:,13:14],y[:,13:14]], dim = 1)    
            X_out = apply_softmax_to_labels(X_out).detach().to('cpu').numpy()
            X_any = X_any.detach().to('cpu').numpy()
            X_out = np.hstack([X_out, X_any])
            X_outs.append(X_out)

            y     = y.to('cpu').numpy()[:,:-1]
            y_any = y_any.to('cpu').numpy()
            y     = np.hstack([y, y_any])
            ys.append(y)

            trn_loss = avg_loss.item()      
            train_meters['loss'].update(trn_loss, n=X.size(0))     
            #pbar.set_description(f'Train loss: {trn_loss}')   
            
            
        print('Epoch {:d} / trn/loss={:.4f}'.format(epoch+1, train_meters['loss'].avg))    

        X_outs = np.vstack(X_outs) 
        ys     = np.vstack(ys)
        metric = calculate_score(X_outs, ys, 'train')                 
        print('Epoch {:d} / train/metric={:.4f}'.format(epoch+1, metric))   

        del X, X_outs, y, ys, X_any
        gc.collect()
        torch.cuda.empty_cache()

        X_outs=[]
        ys=[]
        model.eval()
        for X, y in valid_loader:        
            batch_size = X.shape[0]        
            X, y = X.to(DEVICE), y.to(DEVICE)
                 
            with torch.cuda.amp.autocast(enabled=True):                
                with torch.no_grad():                 
                    X_out, X_any = model(X)                                           
                    y_any = torch.cat([torch.ones(batch_size, 1).to(DEVICE)- y[:,13:14],y[:,13:14]], dim = 1)              
                              
                    X_out = apply_softmax_to_labels(X_out).to('cpu').numpy()

                    X_any = X_any.to('cpu').numpy()
                    X_out = np.hstack([X_out, X_any])
                    X_outs.append(X_out)

                    y     = y.to('cpu').numpy()[:,:-1]
                    y_any = y_any.to('cpu').numpy()
                    y     = np.hstack([y, y_any])
                    ys.append(y)

        X_outs = np.vstack(X_outs) 
        ys     = np.vstack(ys)
        metric = calculate_score(X_outs, ys, 'valid')                
        print('Epoch {:d} / val/metric={:.4f}'.format(epoch+1, metric))           
        
        del X, X_outs, y, ys, X_any
        gc.collect()        
        torch.cuda.empty_cache()
        
        #Save the best model    
        if(metric < np.min(val_metrics)):
            try:
                os.makedirs(f'{BASE_PATH}/weights')
            except:
                a = 1
            best_metric = metric
            print(f'Best val_metric {best_metric} at epoch {epoch+1}!')
            torch.save(model, f'{BASE_PATH}/weights/{backbone}_lr{LR}_epochs_{N_EPOCHS}_resol{UP_RESOL}_batch{BATCH_SIZE}.pt')    
            not_improve_counter=0
            val_metrics[epoch] = metric
            continue            
        
        val_metrics[epoch] = metric                        
        not_improve_counter+=1
        if(not_improve_counter == EARLY_STOP_COUNT):
            print(f'Not improved for {not_improve_counter} epochs, terminate the train')
            break
wandb.log({'best_total_log_loss': best_metric})
wandb.finish()


# In[ ]:


import wandb
try:
    wandb.log({'best_total_log_loss': best_metric})
    wandb.finish()
    
except:
    print('Wandb is already finished!')


# In[ ]:




