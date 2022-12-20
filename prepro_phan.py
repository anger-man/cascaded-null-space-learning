#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:17:38 2022

@author: c
"""

import odl
import scipy
import numpy as np
from skimage.transform import radon, iradon
import matplotlib.pyplot as plt
import os
import time
from phantoms_radon import gen_phantom
from my_phans import generate_random_mask

#%%
# Nal = 320
# thetas = np.linspace(0, 180*(1-1/Nal), Nal)

# lam = 0.01
# data_matching = 'exact'

# N = 320
# Nal   = int(180)
# shape = [N, N]
# Ni=2100

# # tmp = radon(np.zeros([N, N]), np.linspace(0, 180, Nal))
# recsp = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=shape)

# # G=np.zeros([Ni, tmp.shape[0], Nal, 1])
# F=np.zeros([Ni, shape[0], shape[0], 1])

# for i in np.arange(0,Ni):    
#     phantom = gen_phantom(recsp,p=0.5)
#     F[i,:,:,0] = phantom.asarray()
#     print(i)
#     time.sleep(.1)
#     # g = radon(phantom, np.linspace(0, 180, Nal))
#     # G[i,:,:,0]=np.array(g)
    
# # data_sino = G[:,:,:,0]
# data_phan = F[:,:,:,0]


#%%

shape = 320
Ni=2200

F=np.zeros([Ni, shape, shape])

for i in np.arange(0,Ni):    
    phantom = generate_random_mask(dim=shape)
    F[i,:,:] = phantom
    # print(i)

data_phan = F[:,:,:]

plt.imshow(F[0])
plt.imshow(F[4])


#%%

path = 'phantom/'

try:
    os.chdir(path)
except:
    print('Already here')

evaluationp = 'evaluation/'
split1 = 'train/' 

spl1 = np.arange(0,2000,1); eva = np.arange(2000,2200,1)
count=0
for f in range(len(data_phan)):
    x=data_phan[f]
    x=x/x.max(); x[x>1]=1; x[x<0]=0
    
    kspace = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(x))).astype(np.complex64)
    # kspace = np.stack([tmp.real,tmp.imag], axis=-1).astype(np.complex64)
    if f in spl1:
        np.save(split1+'kspace_%03d'%count,kspace)
        count+=1
    elif f in eva:
        np.save(evaluationp+'kspace_%03d'%count,kspace)
        count+=1
        