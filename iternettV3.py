#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:41:21 2022

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
from functions import uncMAE, MAE
from functions import DataGenerator, torch_fourier, torch_inv_fourier
from functions import torch_psnr, plot_images, PE, psnr
from torchmetrics import StructuralSimilarityIndexMeasure as torch_ssim
import optparse
from models import UNet, init_weights
import pytorch_ssim
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)

#%%

parser = optparse.OptionParser()
parser.add_option('--lambda', action="store", type= float,dest="lambda",default=10)
parser.add_option('--wait', action="store", type=int, dest="wait",default=0)
parser.add_option('--lr', action='store', type=float, dest='lr', default=2e-4)
parser.add_option('--method', action='store',type=str,dest='meth', default='nett_unc')
parser.add_option('--task', action='store', type=str, dest='task', default='fastmri')

options,args = parser.parse_args()

time.sleep(options.wait*61)


#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()

index = '%s_%s'%(options.meth, options.task)

#%%

# regularizer = UNet(n_channels =2,f_size=2,out_channels=2, out_acti='tanh')
# regularizer.apply(init_weights)
# regularizer.to(device)

net = UNet(n_channels =2,f_size=64,out_channels=2, out_acti='linear')
net.apply(init_weights)
net.to(device)
summary(net,[[8,2,320,320],[8,320,320]],depth=4, col_names=(['input_size','output_size']),verbose=1)

# optim_reg = torch.optim.Adam(regularizer.parameters(), lr=options.lr)
optim_net = torch.optim.Adam(net.parameters(), lr=options.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim_net, factor=0.25, patience=3, cooldown=1,mode='max')

use_unc = True if options.meth=='nett_unc' else False 
criterion = uncMAE(use_unc = use_unc)

#%%
data_path = os.path.join(os.getcwd(),options.task)

spl1_ids = np.sort(os.listdir(os.path.join(data_path,'split1')))[::2]
test_ids = np.sort(os.listdir(os.path.join(data_path,'evaluation')))
pat = np.unique([f[:15] for f in spl1_ids])
vali_pat = pat[::10]

# spl2_ids = np.sort(os.listdir(os.path.join(data_path,'split2')))
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

test_dataset = DataGenerator(
    path = data_path,
    task = options.task,
    test = True,
    img_ids1 = test_ids)


#%%

#training pipeline

torch.cuda.empty_cache()
gc.collect()
batch_size = 8
epochs = 100
nc = 1
train_resnet=epochs

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)
valid_loader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True,
    num_workers = 6)


#%%




#%%

# train the residual network

step = 0; gloss= torch.Tensor([0]); loss=torch.Tensor([0])
resnet_list = []; resnet_valid_list = []; psnr_list =[]; ssim_list = []
reg_score_list=[]; reg_valid_list = []; lr_rate_list = []
ssim_max = 0.
resnet_iter = epochs +1 if train_resnet else 0
w1 = .5; w2 = .5
    
for epoch in range(1, resnet_iter):
    resnet_score = 0.0
    resnet_valid = 0.0
    psnr_metric = 0.0
    ssim_metric = 0.0
    reg_score = torch.Tensor([0.0])
    reg_valid = torch.Tensor([0.0])
        
    ###################
    # train the model #
    ###################
    
    bar = tq(train_loader, postfix={"net_loss":0.0,"reg_loss":0.0})
   
    for U,recon,full,full2 in bar:
        recon = recon.to(device); full= full.to(device)
        U = U.to(device)
        
        # coord=[]
        # for dummy in range(batch_size):
        #     pwidth= int(np.random.uniform(.5,.6)*320)
        #     coord.append(pwidth)
        #     coord.append(np.random.choice(range(320-pwidth)))
        #     coord.append(np.random.choice(range(320-pwidth)))
        
        # # if (step+1)%2 == 1:
        # regularizer.train(); net.eval()
        # optim_reg.zero_grad()
        # inter = recon + net(recon)[0]
        # pred1,unc1 = regularizer(PE(coord)(inter))
        # loss1 = criterion(PE(coord)(full-inter),pred1,unc1)
        
        # # real input
        # full2  = full2.to(device); 
        # pred2,unc2 = regularizer(PE(coord)(full2))
        # loss2 = criterion(torch.zeros_like(pred2),pred2,unc2)
        # del full2
        
        # if options.meth in ['nett','nett_unc']:
        #     loss = .5*(loss1+loss2)
                  
        # loss.backward()
        # optim_reg.step()
        # reg_score+=loss*full.size(0)

        if (step+1)%nc == 0:
            net.train(); #regularizer.eval()
            optim_net.zero_grad()
            inter,net_out,unc = net(recon,U)
                
            
            # tmp = torch_inv_fourier(tmp)
            # rec = torch.stack([torch.real(tmp), torch.imag(tmp)],axis=1)
            loss_image = .5*criterion(full,net_out,unc)+.5*MAE()(full,inter)
            loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
            # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
            
            gloss = w1 * loss_image + w2*loss_ssim
            # w1 = .95*w1 + 0.05*(loss_ssim/(loss_ssim+loss_image)).item()
            # w2 = .95*w2 + 0.05*(loss_image/(loss_ssim+loss_image)).item()
            gloss.backward()
            optim_net.step()
            resnet_score += gloss*recon.size(0)
                
        bar.set_postfix(ordered_dict={"net_loss":gloss.item(), "reg_loss":loss.item()})
        bar.update(n=1)
        step += 1
        # print('loss_image: %.3f, loss_ssim: %.3f' %(loss_image,loss_ssim))
      
    ######################
    # validate the model #
    ######################
     
    net.eval(); #regularizer.eval()
    save_images = 1
    with torch.no_grad():
        bar = tq(valid_loader, postfix={"net_vali":0.0,"reg_vali":0.0})
        for U,recon,full,full2 in bar:
            recon = recon.to(device); full= full.to(device)
            U = U.to(device)
            
            # coord=[]
            # for dummy in range(batch_size):
            #     pwidth= int(np.random.uniform(.5,.6)*320)
            #     coord.append(pwidth)
            #     coord.append(np.random.choice(range(320-pwidth)))
            #     coord.append(np.random.choice(range(320-pwidth)))
        
            # #####################################
            # inter = recon + net(recon)[0]
            # pred1,unc1 = regularizer(PE(coord)(inter))
            # loss1 = criterion(PE(coord)(full-inter),pred1,unc1)
            # full2  = full2.to(device); 
            # pred2,unc2 = regularizer(PE(coord)(full2))
            # loss2 = criterion(torch.zeros_like(pred2),pred2,unc2)
            # del full2
            # if options.meth in ['nett','nett_unc']:
            #     loss = .5*(loss1+loss2)
            # reg_valid+=loss*full.size(0)
            #####################################
 
            inter,net_out,unc = net(recon,U)
            
            # tmp = net_out
            # tmp = torch.complex(tmp[:,0],tmp[:,1]).to(device)
            # tmp = torch_fourier(tmp)
            # tmp = tmp*U
            # tmp = torch_inv_fourier(tmp)
            # rec = torch.stack([torch.real(tmp), torch.imag(tmp)],axis=1)
            
            loss_image = .5*criterion(full,net_out,unc)+.5*MAE()(full,inter)
            loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
            # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
            
            gloss = (w1/(w1+w2)) * loss_image + (w2/(w1+w2))*loss_ssim
            resnet_valid += gloss*recon.size(0)
                
            gt = full[:,0:1]; pred = (net_out)[:,0:1]
            psnr_metric += torch_psnr(gt, torch.clip(pred,0.,1.))*recon.size(0)
            
            ssim_metric += 100*torch_ssim(data_range=1.)(
                torch.clip(pred,0.,1.).to('cpu'),gt.to('cpu'))*recon.size(0)
            
            bar.set_postfix(ordered_dict={"net_loss":gloss.item(), "reg_loss":loss.item()})
            bar.update(n=1)
            #####################################
            if save_images:
                plot_images(recon,net_out-recon,full, unc, index, epoch)
                save_images = 0
           
    # plt.imshow((recon+net_out)[0,0].cpu());plt.colorbar()
    # plt.imshow((net_out)[0,0].cpu());plt.colorbar()
    # plt.imshow(full[0,0].cpu()); plt.colorbar()
    # plt.imshow(full[0,1].cpu()); plt.colorbar()
            
        
    
    resnet_list.append(resnet_score.item()/len(train_loader.dataset)*nc)
    reg_score_list.append(reg_score.item()/len(train_loader.dataset))
    
    resnet_valid_list.append(resnet_valid.item()/len(valid_loader.dataset))
    reg_valid_list.append(reg_valid.item()/len(valid_loader.dataset))
    lr_rate_list.append([param_group['lr'] for param_group in optim_net.param_groups])

    psnr_list.append(psnr_metric.item()/len(valid_loader.dataset))
    ssim_list.append(ssim_metric.item()/len(valid_loader.dataset))
    sm = ssim_list[-1]
    
    if sm > ssim_max:
        print('\nSSIM increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        ssim_max,
        sm), flush=True)
        torch.save(net.state_dict(), 'resnet_%s.pt' %index)
        # torch.save(regularizer.state_dict(), 'regularizer_%s.pt' %index)
        ssim_max = sm
    
    
    #############################
    # plot intermediate results #
    #############################
    
    fig = plt.figure(figsize=(15,7))
    
    ax = fig.add_subplot(221)
    ax.plot(resnet_list, label='train')
    ax.plot(resnet_valid_list, label='val')
    plt.legend()
    plt.title('resnet_loss', fontsize=15)
    
    ax = fig.add_subplot(222)
    ax.plot(psnr_list, color='red',marker='x')
    ax.plot(ssim_list, color='black',marker='x')
    try:
        plt.title('PSNR %.2f, SSIM %.2f' %(np.max(psnr_list), np.max(ssim_list)))    
    except:
        pass;
    
    ax = fig.add_subplot(223)
    ax.plot(reg_score_list,  marker='x', label="Training Loss")
    ax.plot(reg_valid_list,  marker='x', label="Validation Loss")
    plt.ylabel('regularizer', fontsize=15)
    plt.legend()
    
    ax = fig.add_subplot(224)
    ax.plot([i[0] for i in lr_rate_list])
    plt.ylabel('learing rate during training', fontsize=15)
    
    plt.tight_layout(pad=2)
    plt.savefig('plots/%s/loss.pdf'%(index))
    plt.show()

    #################################
    

    scheduler.step(sm)
    
#%%

w = []
for param in net.parameters():
    w.append(param.cpu().detach().numpy())
    
# from torchviz import make_dot
# make_dot(net_out, params=dict(list(net.named_parameters()))).render("torchviz", format="png")
#%%

##########################
# test the trained model #
##########################

net.load_state_dict(torch.load('4_resnet_%s.pt'%index))
net.to(device)
net.eval(); 

preds = []; gts = []; uncs = []; inps = []
psnr_metric = 0.0
ssim_metric = 0.0

with torch.no_grad():
    bar = tq(test_loader, postfix={"net_vali":0.0,"reg_vali":0.0})
    for U,recon,full,full2 in bar:
        recon = recon.to(device); full= full.to(device)
        U = U.to(device)
        inter,net_out,unc = net(recon,U)
        
        loss_image = .5*criterion(full,net_out,unc)+.5*MAE()(full,inter)
        loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
        # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
        
            
        gt = full[:,0:1]; pred = (net_out)[:,0:1]
        psnr_metric += torch_psnr(gt, torch.clip(pred,0.,1.))*recon.size(0)
        
        ssim_metric += 100*torch_ssim(data_range=1.)(
            torch.clip(pred,0.,1.).to('cpu'),gt.to('cpu'))*recon.size(0)
        
        bar.set_postfix(ordered_dict={"net_loss":gloss.item(), "reg_loss":loss.item()})
        bar.update(n=1)
        
        inps.append(recon[:,0].cpu())
        preds.append(net_out[:,0].cpu())
        gts.append(full[:,0].cpu())
        uncs.append(unc[:,0].cpu())
    preds = np.concatenate(preds,0)
    inps  = np.concatenate(inps,0)
    gts   = np.concatenate(gts,0)
    uncs  = np.concatenate(uncs,0)
        



fig = plt.figure(figsize=(5,6.5))
for i in range(5):
    ax=fig.add_subplot(6, 5, 1 + i)
    plt.axis('off')
    ax.imshow(inps[i])
for i in range(5):
    ax=fig.add_subplot(6, 5, 5+1 + i)
    plt.axis('off')
    ax.imshow(preds[i],cmap='Greys_r')
for i in range(5):
    ax=fig.add_subplot(6, 5, 2*5+1 + i)
    plt.axis('off')
    ax.imshow(gts[i],cmap='Greys_r')
for i in range(5):
    ax=fig.add_subplot(6, 5, 3*5+1 + i)
    plt.axis('off')
    ax.imshow(np.abs(gts-preds)[i],cmap='Reds')
for i in range(5):
    ax=fig.add_subplot(6, 5, 4*5+1 + i)
    plt.axis('off')
    ax.imshow(uncs[i],cmap='Reds',vmin=0,vmax=.4)
plt.show()

print('PSNR: %.3f' %(psnr_metric.item()/len(test_loader.dataset)))
print('SSIMx100: %.3f' %(ssim_metric.item()/len(test_loader.dataset)))



#     q = np.quantile(uncer,0.95,axis=(1,2))
#     ubool=[]
#     for i in range(X1.shape[0]):
#         ubool.append((uncer[i]<q[i]))
#     ubool = np.array(ubool)
#     dif1 = np.mean(np.abs(X2[...,0]-preds_B[...,0]))
#     udif = np.mean(np.abs(X2[...,0]-preds_B[...,0])[ubool])
#     fig.suptitle('DIF1: %.3f\n uncer DIF1 %.3f\n' %(dif1,udif))
    
# fig.tight_layout(pad=.1)
# plt.show()

# blc1 = -np.sort(-np.reshape(X2,(len(X2),320**2)));
# blc2 = -np.sort(-np.reshape(preds_B,(len(preds_B),320**2)));
# dif1 = -np.mean(np.abs(blc1-blc2))/0.11078#np.mean(np.abs(blc1-np.mean(blc1,axis=0)))
# print(dif1)
# MADtrain 0.1108
# MADtrain unfiltered: 0.13524

plt.plot(np.mean(uncs,axis=(1,2)),np.mean(np.abs(gts-preds),axis=(1,2)),'o')
plt.show()
x = pd.Series(np.mean(np.abs(gts-preds),axis=(1,2)))
y = pd.Series(np.mean(uncs,axis=(1,2)))
df = pd.DataFrame(columns=['x_coord','y_coord'])
df['y_coord']=x
df['x_coord']=y
df.to_csv('xy-coordinates.csv')


