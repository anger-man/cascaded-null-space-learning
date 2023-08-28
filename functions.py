#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 12:38:38 2022

@author: c
"""

#%%

#load packages

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import nibabel as nib
import scipy
import matplotlib.pyplot as plt
import gc
import torchvision

    
#%%
def undersample(d3=320,acc=4):
    m=np.zeros(d3)
    center=(12-acc)/100
    m[d3//2-int(d3*center/2):d3//2+int(d3*center/2)]=1
    rest=np.arange(d3)[m==0]
    while(d3/np.sum(m)>acc):
        m[np.random.choice(rest,1)]=1
        rest=np.arange(d3)[m==0]
    u = np.ones((d3,d3))*m
    return(u)

def fourier(data):
    y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(data)))
    return y

def inv_fourier(data):
    y = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(data)))
    return y

def channelize(compl_data):
    return np.stack([compl_data.real,compl_data.imag], axis=0).astype(np.float32)

#%%

#metrics

def psnr(y_true,y_pred):
    res = 20*np.log10(1)-10*np.log10(np.mean(np.square(y_true-y_pred)))
    return res

def torch_psnr(y_true, y_pred):
    res = 20*torch.log10(torch.max(y_true))-10*torch.log10(torch.mean(torch.square(y_true-y_pred)))
    return res

def nmse(y_true,y_pred):
    return (np.sum(np.square(y_true-y_pred))/np.sum(np.square(y_true)))**.5

class MSE(nn.Module):
    def __init__(self, eps = 1e-6):
        self.eps = eps
        super(MSE, self).__init__()

    def forward(self, y_true, y_pred):
        result = torch.square(y_true-y_pred)
        return torch.mean(result)
    
class uncMAE(nn.Module):
    def __init__(self,use_unc=False, eps = 1e-6):
        self.use_unc = use_unc
        self.eps = eps
        super(uncMAE, self).__init__()

    def forward(self, y_true, y_pred, unc):
        if self.use_unc:
            unc = torch.clip(unc,self.eps,1e7)
            tmp = torch.abs(y_true-y_pred)/unc
            tmp = torch.where(tmp<=2,tmp,2+torch.log(torch.clip(tmp-2+1,self.eps,1e7)))
            return torch.mean(tmp + torch.log(1.+2*unc))
            print('yeah')
        else:
            result = torch.abs(y_true-y_pred)
            return torch.mean(result)


class uncMSE(nn.Module):
    def __init__(self,use_unc=False, eps = 1e-6):
        self.use_unc = use_unc
        self.eps = eps
        super(uncMSE, self).__init__()

    def forward(self, y_true, y_pred, unc):
        if self.use_unc:
            unc = torch.clip(unc,self.eps,1e7)
            tmp = torch.square(y_true-y_pred)/(2*torch.square(unc))
            tmp = torch.where(tmp<=2,tmp,2+torch.log(torch.clip(tmp-2+1,self.eps,1e7)))
            return torch.mean(tmp + .5*torch.log(1.+2*unc*np.pi))
            print('yeah')
        else:
            result = torch.square(y_true-y_pred)
            return torch.mean(result)

#%%

class CtDataGenerator(Dataset):
    
    
    def __init__(
            self,
            path: str = None,
            img_ids1: np.array = None,
            reduce_dim: bool = True,
            affine_matrix = False,
            test=False
        ):
        #######################################################################
            self.img_folder1  = f"{path}/train"
            if test:
                self.img_folder1  = f"{path}/evaluation"
            self.img_ids1 = img_ids1
            self.img_ids2 = np.random.permutation(img_ids1)
            self.reduce_dim = reduce_dim
            self.affine_matrix = affine_matrix
            
         ######################################################################   
            
            
            
    def __getitem__(self,idx):
        
        #generate the image path
        image_name = self.img_ids1[idx]
        image_path = os.path.join(self.img_folder1 , image_name)
        data = np.load(image_path)
        return np.expand_dims(data.astype(np.float32),0)
    
    def __len__(self):
        return np.min([len(self.img_ids1),len(self.img_ids1)])

#%%

class DataGenerator(Dataset):
    
    
    def __init__(
            self,
            datatype: str = 'train',
            task: str = 'fastmri',
            path: str = None,
            img_ids1: np.array = None,
            reduce_dim: bool = True,
            disturb_input: bool = True,
            affine_matrix = False,
            test=False
        ):
        #######################################################################
            self.img_folder1  = f"{path}/train"
            if test:
                self.img_folder1  = f"{path}/evaluation"
            self.img_ids1 = img_ids1
            self.img_ids2 = np.random.permutation(img_ids1)
            self.reduce_dim = reduce_dim
            self.disturb_input = disturb_input
            self.affine_matrix = affine_matrix
            self.datatype = datatype
            self.task = task

         ######################################################################   
            
            
            
    def __getitem__(self,idx):
        
        #generate the image path
        image_name = self.img_ids1[idx]
        image_path = os.path.join(self.img_folder1 , image_name)
        U = undersample(320)
        foo = np.load(image_path)
        x = inv_fourier(foo)
        # if np.random.uniform()<0.2:
        #     foo = inv_fourier(foo).real
        #     randn = np.random.randint(0,len(self.array_paths))
        #     shape = np.load(os.path.join('arrays',self.array_paths[randn]))
        #     shape = np.max(shape, axis=0)
        #     shape = scipy.ndimage.zoom(shape,(320/256,320/256),order=0)
        #     foo = np.where(shape==1,.2*foo,foo)
        #     foo = fourier(foo)
            
        y_u = foo*U
        rec = inv_fourier(y_u)
        
        recon = channelize(rec)
        # data = channelize(y_u)
        data=y_u
        full = channelize(x)
        
        image_name = self.img_ids2[idx]
        image_path = os.path.join(self.img_folder1 , image_name)
        foo = np.load(image_path)
        x = inv_fourier(foo)
        full2 = channelize(x)
        
        return U,recon,full,full2
    
    def __len__(self):
        return np.min([len(self.img_ids1),len(self.img_ids1)])



#%%
# define the DiceLoss Function

class MAE(nn.Module):
    def __init__(self,):
        super(MAE, self).__init__()

    def forward(self, targets, preds):
        result = torch.abs(preds-targets)
        return torch.mean(result)
    
class KLD(nn.Module):
    def __init__(self):
        super(KLD,self).__init__()
        
    def forward(self,mean, var):
        res = torch.sum(torch.exp(var)+mean.pow(2)-1.-var)
        return .5*res

class MSE_KLD(nn.Module):
    def __init__(self,kld_weight:float = 0.1):
        super(MSE_KLD,self).__init__()
        self.l1 = MSE()
        self.l2 = KLD()
        self.kldw = kld_weight
        
    def forward(self,out_img,inp_img, mean,var):
        return self.l1(out_img,inp_img) + self.kldw*self.l2(mean,var)
    
class Wasserstein1(nn.Module):
    def __init__(self, eps = 1e-6):
        super(Wasserstein1, self).__init__()

    def forward(self, preds, targets):
        return torch.mean(torch.multiply(preds,targets))
    

#%%

#data loader

class BoneDataset(Dataset):
    
    def __init__(
            self,
            path: str = None,
            img_ids: np.array = None,
            reduce_dim: bool = True,
            latent_dimension:int = 10,
            return_affine: bool = False
        ):
        #######################################################################
            self.img_folder  = f"{path}/train"
            self.mask_folder = f"{path}/train_masks"
            self.img_ids = img_ids
            self.reduce_dim = reduce_dim
            self.ld = latent_dimension
            self.ra = return_affine
         ######################################################################   
            
    def __getitem__(self,idx):
        
        if self.reduce_dim:
            subs = 2
        else:
            subs = 1
 
        #generate the image path
        image_name = self.img_ids[idx]
        image_path = os.path.join(self.img_folder , image_name)
        mask_path = os.path.join(self.mask_folder, image_name)
        
        #load the nifti file (affine matrix is omitted here)
        nifti = nib.load(image_path)
        affine = nifti.affine
        data = nifti.get_fdata()[::subs,::subs,::subs]
        nifti = nib.load(mask_path)
        mask = nifti.get_fdata()[::subs,::subs,::subs].astype(np.uint8)  
        #calculate the cortical columetric bone mineral density
        ctvbmd = np.mean(data[mask==1])
        ctvbmd -= 1110; ctvbmd /= (2644-1110)
        
        data = data-np.quantile(data,0.01); data[data<0] = 0;
        data /= np.quantile(data,.99); data[data>1] = 1
        
        label_nd = ctvbmd*np.ones(data.shape)
        label_ld = ctvbmd
        
        if self.ra:
            return np.expand_dims(data,0), np.expand_dims(label_nd,0), label_ld, affine
        else:
            return np.expand_dims(data,0), np.expand_dims(label_nd,0), label_ld

    def __len__(self):
        return(len(self.img_ids))
    

#%%

def load_data(batch_size, latent_dim, ids, data_path, count, return_affine = False, subs=1):
        img_folder  = f"{data_path}/train"
        mask_folder = f"{data_path}/train_masks"
        DATA = []; LABEL_ND = []; LABEL_LD = []; AFFINE = []
        for t in range(batch_size):
            idx = (count+t)%len(ids)
            image_path = os.path.join(img_folder, ids[idx])
            mask_path = os.path.join(mask_folder, ids[idx])
        
            nifti = nib.load(image_path)
            affine = nifti.affine
            data = nifti.get_fdata()[::subs,::subs,::subs]
            nifti = nib.load(mask_path)
            mask = nifti.get_fdata()[::subs,::subs,::subs].astype(np.uint8)  
            
            #calculate the cortical columetric bone mineral density
            ctvbmd = np.mean(data[mask==1])
            ctvbmd -= 1227; ctvbmd /= (2658-1227)
            
            data = data+292; data[data<0]=0;
            data /= (2809+292); data[data>1]=1
            
            label_nd = ctvbmd*np.ones(data.shape)
            label_ld = ctvbmd
            
            DATA.append(np.expand_dims(data,0)); 
            LABEL_ND.append(np.expand_dims(label_nd,0)); 
            LABEL_LD.append(label_ld); 
            AFFINE.append(affine)
        
        
        if return_affine:
            return np.array(DATA), np.array(LABEL_ND), np.array(LABEL_LD), AFFINE
        else:
            return np.array(DATA), np.array(LABEL_ND), np.array(LABEL_LD)

#%%
def init_weights(m):
    if isinstance(m, nn.Conv3d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/(4**3 * m.in_channels)))
        if m.bias is not None: 
            nn.init.zeros_(m.bias)
            
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/ m.in_features))
        if m.bias is not None: 
            nn.init.zeros_(m.bias)

#%% 

def get_gradient_penalty(data,fake_data,cri, device, p=10):
    eps = torch.rand(data.size(0),1, device=device)
    mixed_images = data*eps.view((-1,1,1,1)) + fake_data*(1.-eps).view((-1,1,1,1))
    mixed_images = torch.autograd.Variable(mixed_images, requires_grad = True)
    mixed_scores = cri(mixed_images)
    gradient = torch.autograd.grad(
        inputs = mixed_images, outputs = mixed_scores, 
        grad_outputs =torch.ones(mixed_scores.size()).to(device),
        create_graph = True, retain_graph = True,only_inputs=True)[0]
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    return p*torch.mean(torch.square(gradient_norm - 1))

def get_drift_penalty(cri_real, cri_fake, scale = 0.005):
    return scale * torch.mean(torch.square(cri_real) + torch.square(cri_fake))

#%%

def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

#%%

def plot_images(recon,net_out,full,unc_map,index,step,unc=True):
    
    fig, ax = plt.subplots(3,5,figsize=(15,6)); 
    for j in range(3):
        im = ax[j,0].imshow(recon.cpu()[j,0], cmap='Greys_r')
        ax[j,0].axis('off')
        plt.colorbar(im,ax=ax[j,0])
        
        im = ax[j,1].imshow((recon+net_out).cpu()[j,0], cmap='Greys_r')
        ax[j,1].axis('off')
        plt.colorbar(im,ax=ax[j,1])
        
        im = ax[j,2].imshow(full.cpu()[j,0], cmap='Greys_r')
        ax[j,2].axis('off')
        plt.colorbar(im,ax=ax[j,2])
        
        im = ax[j,3].imshow(np.abs((recon+net_out-full).cpu()[j,0]),cmap='Reds')
        ax[j,3].axis('off')
        plt.colorbar(im,ax=ax[j,3])
        
        if unc:
            im = ax[j,4].imshow(unc_map.cpu()[j,0])
            ax[j,4].axis('off')
            plt.colorbar(im,ax=ax[j,4])
        
    fig.tight_layout(pad=.1)
    plt.savefig('results/%s/ep_%02d_test.png'%(index,step), dpi=100)
    
    gc.collect()
    
#%%

def torch_fourier(x):
    d = len(x.size()) 
    return torch.fft.fftshift(torch.fft.fft2(x),axis=(d-2,d-1))

def torch_inv_fourier(x):
    d = len(x.size()) 
    return torch.fft.ifft2(torch.fft.fftshift(x,axis=(d-2,d-1)))

#%%

class PE(nn.Module):
    def __init__(self, coord):
        self.coord = coord
        super(PE,self).__init__()
        
    def forward(self,inputs):
        ii = 0
        inputs = inputs.float()
        res = inputs[0:1]
        coord = self.coord
        dim1 = inputs.size(2)
        dim2 = inputs.size(3)
        
        while ii < len(coord):
            pwidth = coord[ii]
            i1 = coord[ii+1]; i2=coord[ii+2]
            j = int(ii/3)
            tmp = inputs[j:j+1,:,i1:i1+pwidth,i2:i2+pwidth]
            tmp = torchvision.transforms.Resize(size=[dim1,dim2])(tmp)
            res = torch.concat([res,tmp],axis=0)
            ii = ii+3
        return(res[1:])
