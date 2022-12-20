#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:17:44 2022

@author: c
"""

import numpy as np
from scipy.fftpack import fftfreq
import random
from scipy import ndimage
import os
import scipy


def create_ellipse(N):
    
    # Create boundary such that ellipses lie compeletly inside image
    check = np.zeros([N, N], dtype=bool)
    check[0:2,:] = True
    check[:,0:2] = True
    check[-2:,:] = True
    check[:,-2:] = True
    
    # Create random Number of ellipses per phantom
    num = random.randint(5, 20)
    # num = 1
    # num = 2
    phantom = np.zeros([N, N])
    for i in range(num):
        
        F = np.zeros([N, N])
        x = np.linspace(-N/2, N/2-1, N)
        X,Y = np.meshgrid(x,x)
        
        # Random denominators for ellipse equation
        a = random.randint(5, 10)
        b = random.randint(5, 30)
            
        # Random rotation of ellipse
        angle = np.random.randint(1, 360)
        X = ndimage.rotate(X, angle, reshape = False, mode = 'mirror')
        Y = ndimage.rotate(Y, angle, reshape = False, mode = 'mirror')
        
        # Random shifts along x and y axis
        X_shift = X - (0.5 - np.random.rand(1))*N/2
        Y_shift = Y - (0.5 - np.random.rand(1))*N/2
        
        # Ellipse
        idx = (X_shift**2/a**2 + Y_shift**2/b**2 ) <= 1
                      
        F[idx] = 1
        
        # Check if ellipse overlaps with boundary, if not add to phantom
        if np.intersect1d(np.where(idx.flatten()), np.where(check.flatten())).size != 0:
            F = create_ellipse(N)
            return F#*random.uniform(0.5, 1.5)
        
        phantom += F#*random.uniform(0.5, 1.5)


    return phantom

#%%

def generate_random_mask(
    dim: int=320, 
    propval: float=.75,
    shapes: list=['elliptical','cuboid','non-convex'],
    shape_prop: list=[.5,.5]):

    """
    dim: defines the spatial dimension of the mask
    propval: specifies the maximum relative amount of voxels occupied by the mask objects
    shapes: list of available shapes; default: ['elliptical','cuboid','non-convex']
    shape_prop: list defining the likelihood of each shape, sum should equal 1
    concave describes the non-convex objects by simon
    """
    
    #define the maximal diameter of a connected component relative to the mask dimension
    diameter = 0.4
    a = dim*diameter

    # prop=np.random.uniform(0.25,propval),
    prop = 1.
    f = np.zeros([dim, dim])
    x = np.linspace(-dim/2, dim/2-1, dim)
    X,Y = np.meshgrid(x,x)
    
    n_samples = np.random.choice(np.arange(10,21,1))
    count = 0
    while count< n_samples:
        
        shape = np.random.choice(['elliptical','cuboid'])
            
        if shape=='elliptical':
            aa = a*np.random.uniform(.1,1)/2
            bb = a*np.random.uniform(.1,1)/2
        
            X_shift = X - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
            Y_shift = Y - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1)
        
            idx = (X_shift**2/aa**2 + Y_shift**2/bb**2 ) <= 1
            idx = np.rot90(idx,k=np.random.choice([1,2,3]))
            
            if np.sum(idx[:2,:])+np.sum(idx[dim-2:,:])+np.sum(idx[:,:2])+np.sum(idx[:,dim-2:])!=0:
                continue;
            else:
                f[idx] += 1
                count+=1
                
            fflat = f.flatten()
            if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                break
            
       
            
        elif shape=='cuboid':

            aa = int(.5*a*np.random.uniform(.1,1))
            bb = int(.5*a*np.random.uniform(.1,1))
        
            x_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
            y_mid = int(dim/2 - np.sign(0.5 - np.random.rand(1))*dim/2*np.random.rand(1))
            idx = np.zeros_like(f,dtype=np.uint8)
            lb = np.max([x_mid-aa,2]);rb = np.min([x_mid+aa,dim-2]);
            ub = np.max([y_mid-bb,2]);dummy = np.min([y_mid+bb,dim-2])
            idx[lb:rb,ub:dummy] = 1
            
            if np.sum(idx[:2,:])+np.sum(idx[dim-2:,:])+np.sum(idx[:,:2])+np.sum(idx[:,dim-2:])!=0:
                continue;
            if np.sum(idx)==0:
                continue;
                
            else:
                f[idx==1] += 1
                count+=1
                
            fflat = f.flatten()
            
            if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                break
            
            
        elif shape=='non-convex':
            files = os.listdir('arrays')
            mask = np.load(os.path.join('arrays',np.random.choice(files)))
            mask = np.max(mask,axis=0)
            idx = scipy.ndimage.zoom(mask,(320/256,320/256), order=0)
        
            if np.sum(idx[:2,:])+np.sum(idx[dim-2:,:])+np.sum(idx[:,:2])+np.sum(idx[:,dim-2:])!=0:
                continue;
            else:
                f[idx==1] += 1
                count+=1
                
            fflat = f.flatten()
            
            if np.count_nonzero(fflat)/np.prod(f.shape) >= prop:
                break
            
        else:
            print('No valid shape'); pass;
            
    
    g= np.zeros_like(f)
    # X_shift = X 
    # Y_shift = Y 
    # aa = 160
    # bb = 160*np.random.uniform(.6,.8)
    # idx = (X_shift**2/aa**2 + Y_shift**2/bb**2 ) <= 1
    # idx = np.rot90(idx,k=np.random.choice([1,2]))
    # g[idx] += 1
    # f[g==0]=0
    return (f+g).astype(np.uint8)