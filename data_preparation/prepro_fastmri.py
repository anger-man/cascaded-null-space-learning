#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:17:42 2021

@author: c
"""

#%%

import h5py
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave,imread
import numpy as np
import time
import pandas as pd
from scipy.ndimage import zoom
from scipy.ndimage import rotate
import imageio


#%%

path = 'fastmri/'
evaluationp = 'evaluation/'
split1 = 'split1/' 
split2 = 'split2/'

try:
    os.chdir(path)
except:
    print('Already here')

#%%

def crop_center(img,cropx,cropy):
    y,x = img.shape[1:]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[:,starty:starty+cropy,startx:startx+cropx]

path = '/media/c/41CDAE5D20F11833/kne/singlecoil_train_val'

files = np.sort(os.listdir(path))

eva = files[::20]
res = [f for f in files if f not in eva]
spl1 = res[::2]
spl2 = res[1::2]

scans=0
for f in files:
    data=h5py.File(os.path.join(path,f))
    acq = data.attrs['acquisition']
    if acq == 'CORPDFS_FBK':
        continue; #remove images with fat suppression
    # y=data['kspace'][:]
    x=data['reconstruction_rss'][:]
    
    # kspace = np.stack([tmp.real,tmp.imag], axis=-1).astype(np.complex64)
    count=0
    scans += 1
    while count<x.shape[0]:
        xx=x[count]/np.quantile(x[count],.99); xx[xx>1]=1; xx[xx<0]=0
        kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(xx))).astype(np.complex64)
        if f in spl1:
            np.save(split1+f+'_%02d'%count,kspace)
            count+=1
        elif f in spl2:
            np.save(split2+f+'_%02d'%count,kspace)
            count+=1
        elif f in eva:
            np.save(evaluationp+f+'_%02d'%count,kspace)
            count+=1
          
# 571 scans
#%%
