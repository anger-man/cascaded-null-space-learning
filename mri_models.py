#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  18 15:53:13 2022

@author: c
"""

#%%

#load packages

import torch
import torch.nn as nn
import numpy as np
from functions import torch_fourier, torch_inv_fourier

#%%
"""
define a 2d Unet with 3 downsampling steps, skip connections
"""

#%%

# functions

def NORM(ch_size,normalization):
    normlayer = nn.ModuleDict([
        ['instance',nn.InstanceNorm2d(ch_size)],
        ['batch', nn.BatchNorm2d(ch_size)],
        ['none', nn.Identity()]
        ])
    return normlayer[normalization]


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/(9 * m.in_channels)))
        if m.bias is not None: 
            nn.init.zeros_(m.bias)
            
    if isinstance(m,nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(2/ m.in_features))
        if m.bias is not None: 
            nn.init.zeros_(m.bias)


#%%

#modules

class double_conv(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(double_conv, self).__init__()
        
        self.conv1  = nn.Sequential(
            nn.Conv2d(in_f, out_f,kernel_size=3,stride=1,padding='same'),           
            NORM(out_f, normalization),
            nn.SiLU())
        
        self.conv2  = nn.Sequential(
            nn.Conv2d(out_f, out_f,kernel_size=3,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.SiLU())
                 
    def forward(self, x):
        return self.conv2(self.conv1(x))
    
    
class downsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization = 'instance'):
        super(downsampling, self).__init__()
        
        self.down  = nn.Sequential(
            nn.Conv2d(in_f, out_f,kernel_size=4,stride=2,padding=(1,1)),   
            NORM(out_f,normalization),
            nn.SiLU())
                 
    def forward(self, x):
        return(self.down(x))
    
    
class upsampling(nn.Module):
    def __init__(self,in_f,out_f,normalization='instance',nearest=True):
        super(upsampling, self).__init__()
        
        if nearest:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode = 'nearest'),
                nn.Conv2d(in_f,out_f,4,stride=1,padding='same'),
                NORM(out_f, normalization),
                nn.SiLU())
        else:
            self.up = nn.ConvTranspose3d(in_f,out_f,kernel_size=4,stride=2,padding=(1,1))
            
        self.conv1  = nn.Sequential(
            nn.Conv2d(in_f, out_f,3,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.SiLU())
        
        self.conv2  = nn.Sequential(
            nn.Conv2d(out_f, out_f,3,stride=1,padding='same'),   
            NORM(out_f, normalization),
            nn.SiLU())
            
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x,skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

#%%

def P_ker(x,U):
    tmp = torch.complex(x[:,0],x[:,1])
    tmp = torch_fourier(tmp)
    tmp = tmp*U.float()
    tmp = torch_inv_fourier(tmp)
    rec = torch.stack([torch.real(tmp), torch.imag(tmp)],axis=1)
    return (x - rec)

#%%
#model 

class CascNullSpace(nn.Module):
    def __init__(self, n_channels, f_size, normalization='none',
                 out_acti = 'linear', out_channels=2, single_nsblock = False):
        super(CascNullSpace, self).__init__()
        
        self.single_nsblock = single_nsblock
        #unet block 1
        self.dc00 = double_conv(n_channels, f_size, normalization)
        self.ds00 = downsampling(f_size, f_size, normalization)
        self.dc01 = double_conv(f_size, 2*f_size, normalization)
        self.ds01 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc02 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds02 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc03 = double_conv(4*f_size, 8*f_size, normalization)
        
        self.up02 = upsampling(8*f_size, 4*f_size, normalization)
        self.up01 = upsampling(4*f_size, 2*f_size, normalization)
        self.up0_out = upsampling(2*f_size, f_size, normalization)
        self.out0 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
        
        #unet block 2
        self.dc10 = double_conv(n_channels+out_channels, f_size, normalization)
        self.ds10 = downsampling(f_size, f_size, normalization)
        self.dc11 = double_conv(f_size, 2*f_size, normalization)
        self.ds11 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc12 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds12 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc13 = double_conv(4*f_size, 8*f_size, normalization)
        self.ds13 = downsampling(8*f_size, 8*f_size, normalization)
        self.dc14 = double_conv(8*f_size, 16*f_size, normalization)
        
        self.up13 = upsampling(16*f_size, 8*f_size, normalization)
        self.up12 = upsampling(8*f_size, 4*f_size, normalization)
        self.up11 = upsampling(4*f_size, 2*f_size, normalization)
        self.up1_out = upsampling(2*f_size, f_size, normalization)
        
        # self.up_unc3 = upsampling(8*f_size, 4*f_size, normalization)
        self.up_unc2 = upsampling(4*f_size, 2*f_size, normalization)
        self.up_unc1 = upsampling(2*f_size, f_size, normalization)
        
        self.out1 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
        self.unc = nn.Sequential(
            nn.Conv2d(f_size, 1,3,stride=1,padding='same'),   
            nn.Softplus())
        
        
    def forward(self, x, U):
        x = x.float()
        U = U.float()
        
        ###############################
        skip00 = self.dc00(x)
        down01 = self.ds00(skip00)
        skip01 = self.dc01(down01)
        down02 = self.ds01(skip01)
        skip02 = self.dc02(down02)
        down03 = self.ds02(skip02)
        skip03 = self.dc03(down03)
        
        upsa02 = self.up02(skip03,skip02)
        upsa01 = self.up01(upsa02,skip01)
        obranch0 = self.up0_out(upsa01,skip00)
        img_out0 = self.out0(obranch0)
        
        inter = x + P_ker(img_out0, U)
        ###############################
        
        if self.single_nsblock:
            
            u = self.up_unc2(upsa02,skip01)
            u = self.up_unc1(u,skip00)
            unc_out = self.unc(u)
            
            return([inter,inter,unc_out])
        
        else:
            new_x = torch.concat([x,inter], axis=1)
            
            skip10 = self.dc10(new_x)
            down11 = self.ds10(skip10)
            skip11 = self.dc11(down11)
            down12 = self.ds11(skip11)
            skip12 = self.dc12(down12)
            down13 = self.ds12(skip12)
            skip13 = self.dc13(down13)
            down14 = self.ds13(skip13)
            skip14 = self.dc14(down14)
            
            upsa13 = self.up13(skip14,skip13)
            upsa12 = self.up12(upsa13,skip12)
            upsa11 = self.up11(upsa12,skip11)
            obranch1 = self.up1_out(upsa11,skip10)
            img_out1 = self.out1(obranch1)
            final = x + P_ker(img_out1, U)
            
            # u = self.up_unc3(upsa13,skip12)
            u = self.up_unc2(upsa12,skip11)
            u = self.up_unc1(u,skip10)
            unc_out = self.unc(u)
            ###############################
            return([inter,final,unc_out])
    
#%%


class regUnet(nn.Module):
    def __init__(self, n_channels, f_size, normalization='none',
                 out_acti = 'tanh', out_channels=2):
        super(regUnet, self).__init__()

        #unet block 2
        self.dc10 = double_conv(n_channels, f_size, normalization)
        self.ds10 = downsampling(f_size, f_size, normalization)
        self.dc11 = double_conv(f_size, 2*f_size, normalization)
        self.ds11 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc12 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds12 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc13 = double_conv(4*f_size, 8*f_size, normalization)
        self.ds13 = downsampling(8*f_size, 8*f_size, normalization)
        self.dc14 = double_conv(8*f_size, 16*f_size, normalization)
        
        self.up13 = upsampling(16*f_size, 8*f_size, normalization)
        self.up12 = upsampling(8*f_size, 4*f_size, normalization)
        self.up11 = upsampling(4*f_size, 2*f_size, normalization)
        self.up1_out = upsampling(2*f_size, f_size, normalization)
        
        self.up_unc = upsampling(2*f_size, f_size, normalization)
        self.out1 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
      
        
        
    def forward(self, x):
        x = x.float()
        
        skip10 = self.dc10(x)
        down11 = self.ds10(skip10)
        skip11 = self.dc11(down11)
        down12 = self.ds11(skip11)
        skip12 = self.dc12(down12)
        down13 = self.ds12(skip12)
        skip13 = self.dc13(down13)
        down14 = self.ds13(skip13)
        skip14 = self.dc14(down14)
        
        upsa13 = self.up13(skip14,skip13)
        upsa12 = self.up12(upsa13,skip12)
        upsa11 = self.up11(upsa12,skip11)
        obranch1 = self.up1_out(upsa11,skip10)
        img_out1 = self.out1(obranch1)
        final = img_out1
        
        ###############################
        return(final)
    
#%%

class Unet(nn.Module):
    def __init__(self, n_channels, f_size, normalization='none',
                 out_acti = 'tanh', out_channels=2):
        super(Unet, self).__init__()

        #unet block 2
        self.dc10 = double_conv(n_channels, f_size, normalization)
        self.ds10 = downsampling(f_size, f_size, normalization)
        self.dc11 = double_conv(f_size, 2*f_size, normalization)
        self.ds11 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc12 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds12 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc13 = double_conv(4*f_size, 8*f_size, normalization)
        self.ds13 = downsampling(8*f_size, 8*f_size, normalization)
        self.dc14 = double_conv(8*f_size, 16*f_size, normalization)
        
        self.up13 = upsampling(16*f_size, 8*f_size, normalization)
        self.up12 = upsampling(8*f_size, 4*f_size, normalization)
        self.up11 = upsampling(4*f_size, 2*f_size, normalization)
        self.up1_out = upsampling(2*f_size, f_size, normalization)
        
        self.out1 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
        
        
    def forward(self, x):
        x = x.float()
        
        skip10 = self.dc10(x)
        down11 = self.ds10(skip10)
        skip11 = self.dc11(down11)
        down12 = self.ds11(skip11)
        skip12 = self.dc12(down12)
        down13 = self.ds12(skip12)
        skip13 = self.dc13(down13)
        down14 = self.ds13(skip13)
        skip14 = self.dc14(down14)
        
        upsa13 = self.up13(skip14,skip13)
        upsa12 = self.up12(upsa13,skip12)
        upsa11 = self.up11(upsa12,skip11)
        obranch1 = self.up1_out(upsa11,skip10)
        img_out1 = self.out1(obranch1)
        final = x + img_out1
        
        ###############################
        return([final,final])
    
    
#%%

class CascadedUnet(nn.Module):
    def __init__(self, n_channels, f_size, normalization='none',
                 out_acti = 'tanh', out_channels=2):
        super(CascadedUnet, self).__init__()
        
        #unet block 1
        self.dc00 = double_conv(n_channels, f_size, normalization)
        self.ds00 = downsampling(f_size, f_size, normalization)
        self.dc01 = double_conv(f_size, 2*f_size, normalization)
        self.ds01 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc02 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds02 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc03 = double_conv(4*f_size, 8*f_size, normalization)
        
        self.up02 = upsampling(8*f_size, 4*f_size, normalization)
        self.up01 = upsampling(4*f_size, 2*f_size, normalization)
        self.up0_out = upsampling(2*f_size, f_size, normalization)
        self.out0 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
        
        #unet block 2
        self.dc10 = double_conv(n_channels+out_channels, f_size, normalization)
        self.ds10 = downsampling(f_size, f_size, normalization)
        self.dc11 = double_conv(f_size, 2*f_size, normalization)
        self.ds11 = downsampling(2*f_size, 2*f_size, normalization)
        self.dc12 = double_conv(2*f_size, 4*f_size, normalization)
        self.ds12 = downsampling(4*f_size, 4*f_size, normalization)
        self.dc13 = double_conv(4*f_size, 8*f_size, normalization)
        self.ds13 = downsampling(8*f_size, 8*f_size, normalization)
        self.dc14 = double_conv(8*f_size, 16*f_size, normalization)
        
        self.up13 = upsampling(16*f_size, 8*f_size, normalization)
        self.up12 = upsampling(8*f_size, 4*f_size, normalization)
        self.up11 = upsampling(4*f_size, 2*f_size, normalization)
        self.up1_out = upsampling(2*f_size, f_size, normalization)
        
        self.out1 = nn.Sequential(
            nn.Conv2d(f_size, out_channels,3,stride=1,padding='same'),   
            nn.ModuleDict([['relu',nn.ReLU()],['tanh',nn.Tanh()],['linear',nn.Identity()]]
                          )[out_acti])
       
        
        
    def forward(self, x):
        x = x.float()
        
        ###############################
        skip00 = self.dc00(x)
        down01 = self.ds00(skip00)
        skip01 = self.dc01(down01)
        down02 = self.ds01(skip01)
        skip02 = self.dc02(down02)
        down03 = self.ds02(skip02)
        skip03 = self.dc03(down03)
        
        upsa02 = self.up02(skip03,skip02)
        upsa01 = self.up01(upsa02,skip01)
        obranch0 = self.up0_out(upsa01,skip00)
        img_out0 = self.out0(obranch0)
        
        ###############################
        inter = x + img_out0
        new_x = torch.concat([x,inter], axis=1)
        ###############################
        
        skip10 = self.dc10(new_x)
        down11 = self.ds10(skip10)
        skip11 = self.dc11(down11)
        down12 = self.ds11(skip11)
        skip12 = self.dc12(down12)
        down13 = self.ds12(skip12)
        skip13 = self.dc13(down13)
        down14 = self.ds13(skip13)
        skip14 = self.dc14(down14)
        
        upsa13 = self.up13(skip14,skip13)
        upsa12 = self.up12(upsa13,skip12)
        upsa11 = self.up11(upsa12,skip11)
        obranch1 = self.up1_out(upsa11,skip10)
        img_out1 = self.out1(obranch1)
        final = x + img_out1
        
        ###############################
        return([inter,final])
    
#%%

class critic(nn.Module):
    def __init__(self,n_channels,ch_size,normalization = 'instance'):
        super(critic,self).__init__()
        
        self.ds0 = downsampling(n_channels, ch_size, normalization)
        self.ds1 = downsampling(ch_size, 2*ch_size, normalization)
        self.ds2 = downsampling(2*ch_size, 4*ch_size, normalization)
        self.ds3 = downsampling(4*ch_size, 8*ch_size, normalization)
        self.ds4 = downsampling(8*ch_size, 8*ch_size, normalization)
        self.out = nn.Conv2d(8*ch_size,1,stride=1,kernel_size=(4,4),padding='valid')
        
        
    def forward(self,x):
        x = x.float();
        e = self.ds0(x)
        e = self.ds1(e)
        e = self.ds2(e)
        e = self.ds3(e)
        e = self.ds4(e)
        out = self.out(e)
        
        return(out)
    
def calculate_respective_field(S,F):
    r=1
    for l in range(1,len(S)+1,1):
        s=1
        i=1
        while i<l:
            s*=S[i-1]
            i+=1
        r+=(F[l-1]-1)*s
    return r

calculate_respective_field([2,2,2,2,2,1],[4,4,4,4,4,4])
