#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 11:41:42 2022

@author: c
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:30:11 2022

@author: c
"""

#%%

#load packages

import torch
torch.cuda.get_device_name(0)
from torchinfo import summary
from torch.utils.data import  DataLoader
import gc, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
import time
from functions import undersample, fourier, inv_fourier, channelize, MSE, MAE,uncMSE
from functions import DataGenerator, psnr
import optparse
import pandas as pd
from models import regUnet, init_weights
from skimage.metrics import structural_similarity as ssim
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)
#%%

parser = optparse.OptionParser()
parser.add_option('--lambda', action="store", type= float,dest="lambda",default=10)
parser.add_option('--wait', action="store", type=int, dest="wait",default=0)
parser.add_option('--lr', action='store', type=float, dest='lr', default=2e-4)
parser.add_option('--method', action='store',type=str,dest='meth', default='nett')
parser.add_option('--task', action='store', type=str, dest='task', default='phantom')
parser.add_option('--bs', action = 'store', type=float, dest='bs', default = 6)
parser.add_option('--epochs', action = 'store', type=float, dest='epochs', default = 60)

options,args = parser.parse_args()

time.sleep(options.wait*61)


#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()

index = '%s_%s'%(options.task, options.meth)

#%%

regularizer = regUnet(n_channels =2,f_size=32,out_channels=2)
regularizer.apply(init_weights)
regularizer.to(device)
print(summary(regularizer,[8,2,320,320]))

optim = torch.optim.Adam(regularizer.parameters(), lr=options.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.25, patience=3, cooldown=1)

criterion = MAE()

#%%
data_path = os.path.join(os.getcwd(),options.task)

spl1_ids = np.sort(os.listdir(os.path.join(data_path,'train')))
pat = np.unique([f[:15] for f in spl1_ids])
vali_pat = pat[::10]

vali1 = [f for f in spl1_ids if f[:15] in vali_pat]
train1 = [f for f in spl1_ids if f not in vali1]

train_dataset = DataGenerator(
    path = data_path,
    task = options.task,
    img_ids1 = train1)

valid_dataset = DataGenerator(
    path = data_path,
    task = options.task,
    img_ids1 = vali1)


#%%

#training pipeline

torch.cuda.empty_cache()
gc.collect()
batch_size = options.bs
epochs = options.epochs

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)


#%%

train_loss_list = []
valid_loss_list =[]
lr_rate_list = []
valid_loss_min = 1e5
# try:
#     os.mkdir( str('plots/%s'%(index)))
# except:
#     print( str('plots/%s  already exists'%(index)))
    

for epoch in range(1, epochs+1):
    train_loss = 0.0
    valid_loss = 0.0
    
    ###################
    # train the model #
    ###################
    regularizer.train()
    bar = tq(train_loader, postfix={"train_loss":0.0})
    
    for U,recon,full,full2 in bar:
        
        # clear the gradients of all optimized variables
        del U
        optim.zero_grad()
        
        # reconstructed input
        recon  = recon.to(device); full = full.to(device)
        pred1 = regularizer(recon)
        loss1 = criterion(full-recon,pred1)
        
        # real input
        full2  = full2.to(device); 
        pred2 = regularizer(full2)
        loss2 = criterion(torch.zeros_like(pred2),pred2)
        del full2
        
        if options.meth in ['nett']:
            loss = .5*(loss1+loss2)
            
        elif options.meth == 'nettScaled':
            # optimize on traces
            eps = torch.rand(recon.size(0),1, device=device)
            mixed = recon*eps.view((-1,1,1,1)) + full*(1.-eps).view((-1,1,1,1))
            pred3 = regularizer(mixed)
            loss3 = criterion(full-mixed,pred3)
            loss = 1/3*(loss1+loss2+loss3)
        
        loss.backward()
        optim.step()
        
        # perform a single optimization step (parameter update)
        train_loss+=loss*full.size(0)
        bar.set_postfix(ordered_dict={"train_loss":loss.item()})
        bar.update(n=1)
        
    ###################
    # validate the model #
    ###################
    regularizer.eval()
    
    with torch.no_grad():
        bar = tq(valid_loader, postfix={"validation_loss":0.0})
        
        for data, recon, full, full2 in bar:
            del data
            # reconstructed input
            recon  = recon.to(device); full = full.to(device)
            pred1 = regularizer(recon)
            loss1 = criterion(full-recon,pred1)
            
            # real input
            full2  = full2.to(device)
            pred2 = regularizer(full2)
            loss2 = criterion(torch.zeros_like(pred2),pred2)
            del full2
            
            if options.meth=='nett':
                loss = .5*(loss1+loss2)
                
            elif options.meth == 'nettScaled':
                # optimize on traces
                eps = torch.rand(recon.size(0),1, device=device)
                mixed = recon*eps.view((-1,1,1,1)) + full*(1.-eps).view((-1,1,1,1))
                pred3,unc3 = regularizer(mixed)
                loss3 = criterion(full-mixed,pred3)
                loss = 1/3*(loss1+loss2+loss3)
           
            
            # perform a single optimization step (parameter update)
            valid_loss+=loss*full.size(0)
            bar.set_postfix(ordered_dict={"validation_loss":loss.item()})
            bar.update(n=1)
                    
        
    train_loss = train_loss.item()/len(train_loader.dataset)
    valid_loss = valid_loss.item()/len(valid_loader.dataset)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    lr_rate_list.append([param_group['lr'] for param_group in optim.param_groups])
    
    scheduler.step(valid_loss)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121)
    ax.plot([i[0] for i in lr_rate_list])
    plt.ylabel('learing rate during training', fontsize=22)
    
    ax = fig.add_subplot(122)
    ax.plot(train_loss_list,  marker='x', label="Training Loss")
    ax.plot(valid_loss_list,  marker='x', label="Validation Loss")
    plt.ylabel('loss', fontsize=22)
    plt.legend()
    plt.savefig('plots/%s/loss.pdf'%index)
    
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss), flush=True)
        torch.save(regularizer.state_dict(), '%s.pt' %index)
        valid_loss_min = valid_loss

#%%

indices = np.sort([os.path.join(data_path,'evaluation',f) 
                   for f in os.listdir('%s/evaluation'%options.task)])
Y = []; X =[]; 
U = np.load('U_%s.npy'%options.task); count=0
# U=[]
for i in indices:
    # u = undersample(320)
    # U.append(u.astype(np.uint8))
    u = U[count]
    foo = np.load(i)
    Y.append(foo*u)
    X.append(inv_fourier(foo))
    count+=1
    # time.sleep(.5)
    
# np.save('U_%s.npy'%options.task,U)
    
#%%

K=np.linspace(0,len(indices)-1,10).astype(np.uint16)
LAM = np.linspace(.1,2,10)
stepsize = np.linspace(0.1,1,5)
TABLE = pd.DataFrame(columns=['lam','stepsize','psnr','ssim'])
regularizer.to(device)
regularizer.load_state_dict(torch.load('%s.pt'%index))
regularizer.eval()

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
###############################

for ss in stepsize:
    for lam in LAM:
        psnr_list = []; ssim_list = []
        for k in K:
            u = U[k]; y = Y[k]; x0 = inv_fourier(y); gt = X[k]; metric =[]
            maxiter = 100
            s = np.repeat(ss, maxiter); 
            img_fidelity=[]; img_reg=[]
            x = x0
            for i in range(maxiter):
                
                #data fidelity
                dif = inv_fourier((fourier(x)*u-y)*u)
                xnew = x-s[i]*dif
                img_fidelity.append(xnew.real)
                
                #artifact removal
                tmp = torch.Tensor(np.expand_dims(
                    channelize(xnew),0)).to(device)
                tmp.requires_grad = True
                
                                              
                # plt.imshow(regularizer(tmp).cpu().detach().numpy()[0,0]); plt.colorbar()
                pred,unc =regularizer(tmp)
                reg_out = torch.sum(torch.square(pred))
                grad = torch.autograd.grad(
                    inputs = tmp, outputs = reg_out, 
                    create_graph = True, retain_graph = True,only_inputs=True)[0]
                grad = grad[0].cpu().detach().numpy()    
                grad_reg = np.zeros_like(x)
                grad_reg.real = grad[0]
                grad_reg.imag = grad[1]
                xnew = xnew-s[i]*lam*grad_reg
                x = xnew
                img_reg.append(x.real)
                
                metric.append([psnr(gt.real,np.clip(x.real,0,1)),
                               100*ssim(gt.real,np.clip(x.real,0,1),data_range=1),
                               torch.mean(unc).item()])
                
               
            I = np.linspace(0,maxiter-1,5).astype(np.uint16)
            metric=np.array(metric)
            psnr_list.append(metric[-1,0]); ssim_list.append(metric[-1,1])
               
            fig, ax = plt.subplots(2,2,figsize=(9,5))
            pl = ax[0,0].plot(metric[:,0])
            pl = ax[0,1].plot(metric[:,1],color='orange')
            pl = ax[1,0].plot(metric[:,2],color='black')
            plt.suptitle('lam: %.2f, psnr: %.2f, sim: %.2f'%(
                lam,metric[-1,0],metric[-1,1])); 
            plt.show()
                
           
            
            fig, ax = plt.subplots(3,5,figsize=(18,9)); 
            for j in range(5):
                im = ax[0,j].imshow(img_fidelity[I[j]], cmap='Greys_r', vmin=0, vmax=1)
                ax[0,j].axis('off')
                plt.colorbar(im,ax=ax[0,j])
                
                im = ax[1,j].imshow(img_reg[I[j]], cmap='Greys_r',vmin=0,vmax=1)
                ax[1,j].axis('off') 
                plt.colorbar(im,ax=ax[1,j])
            im = ax[2,3].imshow(np.real(x0), cmap='Greys_r',vmin=0,vmax=1)
            ax[2,3].axis('off')
            plt.colorbar(im,ax=ax[2,3])
            im = ax[2,4].imshow(gt.real, cmap='Greys_r')
            ax[2,4].axis('off')
            plt.colorbar(im,ax=ax[2,4])
            fig.tight_layout(pad=.1)
            plt.show()
                
            
            
            
        
    
        table = pd.DataFrame(np.array([lam,ss,np.mean(psnr_list),np.mean(ssim_list)]).reshape(1,4),
                             columns=['lam','stepsize','psnr','ssim'])
    
        TABLE=pd.concat([TABLE,table],ignore_index=True)
        TABLE.to_csv('results_%s.csv'%index)
        
        gc.collect()
