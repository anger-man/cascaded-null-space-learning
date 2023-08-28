#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:41:21 2022

@author: c
"""

#%%

#load packages

import torch
from torchinfo import summary
from torch.utils.data import  DataLoader
import gc, os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm as tq
import time
from functions import uncMAE, MAE
from functions import DataGenerator 
from functions import torch_psnr, plot_images, PE, psnr
import optparse
from mri_models import init_weights, CascNullSpace
import pytorch_ssim
from skimage.metrics import structural_similarity as ssim
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)

#%%

parser = optparse.OptionParser()
parser.add_option('--lambda', action="store", type= float,dest="lambda",default=10)
parser.add_option('--wait', action="store", type=int, dest="wait",default=0)
parser.add_option('--lr', action='store', type=float, dest='lr', default=5e-5)
parser.add_option('--method', action='store',type=str,dest='meth', default='nullspace')
parser.add_option('--architecture', action='store', type=str, dest='arch', default = 'casnet')
parser.add_option('--task', action='store', type=str, dest='task', default='fastmri')
parser.add_option('--bs', action = 'store', type=float, dest='bs', default = 6)
parser.add_option('--epochs', action = 'store', type=float, dest='epochs', default =40)

options,args = parser.parse_args()

time.sleep(options.wait*61)


#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()

index = '%s_%s_%s'%(options.task, options.arch, options.meth)

#%%

# regularizer = UNet(n_channels =2,f_size=2,out_channels=2, out_acti='tanh')
# regularizer.apply(init_weights)
# regularizer.to(device)

if options.arch == 'unet':
    net = CascNullSpace(n_channels =2,f_size=32,out_channels=2, single_nsblock=True)
if options.arch == 'casnet':
    net = CascNullSpace(n_channels =2,f_size=32,out_channels=2, single_nsblock=False)
    
net.apply(init_weights)
net.to(device)
summary(net,[[8,2,320,320],[8,320,320]],depth=4, col_names=(['input_size','output_size']),verbose=1)

# optim_reg = torch.optim.Adam(regularizer.parameters(), lr=options.lr)
optim_net = torch.optim.Adam(net.parameters(), lr=options.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim_net, factor=0.25, patience=2, cooldown=1,mode='max')

use_unc = True if options.meth=='nullspaceUnc' else False 
adapt_w = True if options.meth=='nullspaceUnc' else False 

criterion = uncMAE(use_unc = use_unc)

#%%
data_path = os.path.join(os.getcwd(),options.task)

spl1_ids = np.sort(os.listdir(os.path.join(data_path,'train')))
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
batch_size = int(options.bs)
epochs = int(options.epochs)
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
w1 = .9; w2 = .1
    
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
            loss_image = .8*criterion(full,net_out,unc)+.2*MAE()(full,inter)
            loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
            # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
            
            gloss = w1 * loss_image + w2*loss_ssim
            if adapt_w:
                w1 = .999*w1 + 0.001*(loss_ssim/(loss_ssim+loss_image)).item()
                w2 = .999*w2 + 0.001*(loss_image/(loss_ssim+loss_image)).item()
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
    print('w1 %.4f, w2 %.4f' %(w1,w2))
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
            
            loss_image = .8*criterion(full,net_out,unc)+.2*MAE()(full,inter)
            loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
            # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
            
            gloss = (w1/(w1+w2)) * loss_image + (w2/(w1+w2))*loss_ssim
            resnet_valid += gloss*recon.size(0)
                
            gt = full[:,0:1].cpu().numpy(); pred = (net_out)[:,0:1].cpu().numpy()
            
            psnr_metric += psnr(gt, np.clip(pred,0,1))*recon.size(0)
            ssim_metric += 100*ssim(gt[:,0],np.clip(pred,0.,1.)[:,0], data_range=1,
                                    channel_axis=0)*recon.size(0)
            
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
        torch.save(net.state_dict(), '%s.pt' %index)
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
    plt.savefig('results/%s/loss.pdf'%(index))
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

net.load_state_dict(torch.load('%s.pt'%index, map_location = device))
net.to(device)
net.eval(); 

preds = []; gts = []; uncs = []; inps = []; Us = []
psnr_metric = 0.0
ssim_metric = 0.0

with torch.no_grad():
    bar = tq(test_loader)
    for U,recon,full,full2 in bar:
        recon = recon.to(device); full= full.to(device)
        U = U.to(device)
        del full2
        inter,net_out,unc = net(recon,U)
        
        loss_image = .8*criterion(full,net_out,unc)+.2*MAE()(full,inter)
        loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],net_out[:,0:1])
        # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
        
            
        gt = full[:,0:1].cpu().numpy(); pred = (net_out)[:,0:1].cpu().numpy()
        
        psnr_metric += psnr(gt, np.clip(pred,0,1))*recon.size(0)
        ssim_metric += 100*ssim(gt[:,0],np.clip(pred,0.,1.)[:,0], data_range=1,
                                channel_axis=0)*recon.size(0)
        
        # bar.set_postfix(ordered_dict={"net_loss":gloss.item(), "reg_loss":loss.item()})
        bar.update(n=1)
        
        inps.append(recon.cpu())
        preds.append(net_out[:,0].cpu())
        gts.append(full[:,0].cpu())
        uncs.append(unc[:,0].cpu())
        Us.append(U.cpu())
    preds = np.concatenate(preds,0)
    inps  = np.concatenate(inps,0)
    gts   = np.concatenate(gts,0)
    uncs  = np.concatenate(uncs,0)
    Us = np.concatenate(Us,0)
        



fig = plt.figure(figsize=(5,6.5))
for i in range(5):
    ax=fig.add_subplot(6, 5, 1 + i)
    plt.axis('off')
    ax.imshow(inps[i,0])
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

print('PSNR: %.5f' %(psnr_metric.item()/len(test_loader.dataset)))
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

#%%

# perturbations

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{siunitx}'] 
import scipy

targets = []; preds =[]; uncs=[]
for k in range(100):
    file = gts[k]
    U = Us[k]
    data =np.fft.fftshift(np.fft.fft2(file))*U
    data_clean = data.copy()

    mask = np.random.uniform(size=data.shape)
    data_ip05 = 11*np.random.normal(size=data.shape)+data
    data_ip2 =  20*np.random.normal(size=data.shape)+data
    
    w1 = []; w2 = []
    for d in [data_clean, data_ip05, data_ip2]:
        file =np.fft.ifft2(np.fft.fftshift(d))
        file = np.stack([np.real(file),np.imag(file)], axis=0)
    
        data = torch.Tensor(file).to(device)
        U = torch.Tensor(U).to(device)
        with torch.no_grad():
            inter,net_out,unc = net(data.unsqueeze(0), U.unsqueeze(0))
        w1.append(net_out[0,0].cpu())
        w2.append(unc[0,0].cpu())
        
    preds.append(np.stack(w1,axis=0))
    uncs.append(np.stack(w2,axis=0))
    targets.append(gts[k])
    
    
preds = np.array(preds); uncs = np.array(uncs); targets = np.array(targets)
target_names = np.repeat([1,2,3],repeats = len(preds))
p = np.concatenate([preds[:,0],preds[:,1],preds[:,2]], axis=0)
t = np.concatenate([targets,targets,targets], axis=0)
u = np.concatenate([uncs[:,0],uncs[:,1],uncs[:,2]], axis=0)

x = np.mean(np.abs(p-t), axis=(1,2))
y = np.mean(u, axis=(1,2))

#%%

import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

fig, ax = plt.subplots()
k=0
for label in [r"noise-reduced",
              r"noise $|| \epsilon ||_2 \leq 5\times 10^3$",
              r"noise $|| \epsilon ||_2 \leq 9\times 10^3$"]:
    color = ['tab:blue', 'tab:orange', 'tab:green'][k]
    
    xx = x[k*100:(k+1)*100]; yy=y[k*100:(k+1)*100]
    print(xx[0])
    ax.scatter(yy, xx, c=color, label=label,
               alpha=0.3, edgecolors='none')
    k+=1
ax.set_xlabel('$Mean \ \ Uncertainty$', labelpad=2)
ax.set_ylabel('$Mean \ \ Abs \ \ Residual$',  labelpad=2)
plt.xlim(0.03,0.19)
ax.legend()
ax.grid(True)
plt.savefig('%s_unc.pdf'%(options.task), dpi=300)
plt.show()

    
#%%


# insert alpha shapes

import scipy

cor_inps =[]; cor_uncs = []; cor_preds =  []
array_paths = os.listdir('arrays')
for k in range(5):
    file = gts[k]
    U = Us[k]
    randn = np.random.randint(0,len(array_paths))
    shape = np.load(os.path.join('arrays',array_paths[randn]))
    shape = np.max(shape, axis=0)
    shape = scipy.ndimage.zoom(shape,(180/256,180/256),order=0)
    shape = np.pad(shape,[[70,70],[70,70]])
    file = np.where(shape==1,file+.5*np.random.normal(size=file.shape),file)
    # file = np.where(shape==1,0,file)
    file =np.fft.fftshift(np.fft.fft2(file))*U
    file =np.fft.ifft2(np.fft.fftshift(file))
    file = np.stack([np.real(file),np.imag(file)], axis=0)

    
    data = torch.Tensor(file).to(device)
    U = torch.Tensor(U).to(device)
    with torch.no_grad():
        inter,net_out,unc = net(data.unsqueeze(0), U.unsqueeze(0))
    
    cor_inps.append(data[0:1].cpu())
    cor_preds.append(net_out[:,0].cpu())
    cor_uncs.append(unc[:,0].cpu())
    
cor_inps = np.concatenate(cor_inps, 0)
cor_preds = np.concatenate(cor_preds, 0)
cor_uncs = np.concatenate(cor_uncs, 0)

#%%
fig = plt.figure(figsize=(5,6.5))
for i in range(5):
    ax=fig.add_subplot(6, 5, 1 + i)
    plt.axis('off')
    ax.imshow(cor_inps[i])
for i in range(5):
    ax=fig.add_subplot(6, 5, 5+1 + i)
    plt.axis('off')
    ax.imshow(cor_preds[i],cmap='Greys_r',vmin=0,vmax=1)
for i in range(5):
    ax=fig.add_subplot(6, 5, 2*5+1 + i)
    plt.axis('off')
    ax.imshow(gts[i],cmap='Greys_r')
for i in range(5):
    ax=fig.add_subplot(6, 5, 3*5+1 + i)
    plt.axis('off')
    ax.imshow(np.abs(gts[i]-cor_preds[i]),cmap='Reds')
for i in range(5):
    ax=fig.add_subplot(6, 5, 4*5+1 + i)
    plt.axis('off')
    ax.imshow(np.square(cor_uncs[i]),cmap='Reds',vmax=.1)
fig.tight_layout(pad=.1)
plt.show()

#%%

# np.save('fastmri_cor_inps.npy',cor_inps)
# np.save('fastmri_cor_preds.npy',cor_preds)
# np.save('fastmri_cor_uncs.npy',cor_uncs)
# import matplotlib
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# indices = [2,4]
# ind = [90,150,265,205,
#        128,192,192,128]

# fig, ax = plt.subplots(3,2,figsize=(8.5,12)); 
# for j in range(2):
    
#     im = ax[0,j].imshow(cor_inps[indices[j]], cmap='Greys_r')
#     ax[0,j].axis('off')
#     ax[0,j].set_title(r"$\mathcal{A}^+(y_\mathrm{cor})$",fontsize=14,y=0.989)
    
#     for kk in range(2):
#         if kk == 0:
#             r=cor_preds[indices[j]]; cmap = 'Greys_r';vmax=1; vmin=0
#             title = r"${\mathrm{NS}^\mathrm{I}}$"
#         else:
#             r=np.square(cor_uncs[indices[j]]); cmap = 'gist_heat';vmax=.1;vmin=-0.01
#             title = r"${\mathrm{NS}^\mathrm{\sigma}}$"
            
#         im = ax[kk+1,j].imshow(r, cmap=cmap,vmin=vmin,vmax=vmax)
#         ax[kk+1,j].set_title(title,fontsize=12, y=0.98)
#         ax[kk+1,j].axis('off')
#         axins = zoomed_inset_axes(ax[kk+1,j],1.9,loc='upper right')
#         axins.imshow(r, cmap=cmap,vmin=vmin,vmax=vmax)
#         axins.set_xlim(ind[j*4], ind[j*4+1])
#         axins.set_ylim(ind[j*4+2], ind[j*4+3])
#         plt.xticks(visible=False)
#         plt.yticks(visible=False)
#         _patch,pp1,pp2 = mark_inset(ax[kk+1,j] ,axins, loc1=1, loc2=3, fc="none", ec="1")
#         pp1.loc1, pp1.loc2 = 2, 3
#         pp2.loc1, pp2.loc2 = 4, 1
#         plt.draw()
        
# fig.tight_layout(h_pad=.5,w_pad = .5)
# plt.savefig('%s.pdf'%(options.task), dpi=300)


#%%
from functions import inv_fourier, fourier, undersample

    
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
    # time.sleep(.2)
    
# np.save('U_%s.npy'%options.task,U)


K=np.linspace(0,len(indices)-1,np.min([len(indices),400])).astype(np.uint16)
stepsize = np.linspace(0.1,1,5)
TABLE = pd.DataFrame(columns=['stepsize','psnr','ssim'])
net.to(device)
# net.load_state_dict(torch.load('%s.pt' %(index)))
net.load_state_dict(torch.load('%s.pt' %(index)))
net.eval()

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
ss=0
psnr_list = []; ssim_list = []
for k in K:
    u = U[k]; y = Y[k]; x0 = inv_fourier(y); gt = X[k]; metric =[]
    maxiter = 1
    s = np.repeat(ss, maxiter); 
    img_fidelity=[]; 
    init = torch.Tensor(np.expand_dims(np.stack(
        [x0.real, x0.imag],axis=0),0)).to(device)
    with torch.no_grad():
        init =  net(init, torch.Tensor(np.expand_dims(u,0)).to(device))[1]
    res = np.zeros_like(x0)
    res.real = init[0,0].cpu(); res.imag = init[0,1].cpu()
    x = res
    for i in range(maxiter):
        
        #data fidelity
        dif = inv_fourier((fourier(x)*u-y)*u)
        xnew = x-s[i]*dif
        x = xnew
        img_fidelity.append(xnew.real)
                          
        metric.append([psnr(gt.real,np.clip(x.real,0,1)),
                       100*ssim(gt.real,np.clip(x.real,0,1),data_range=1)
                       ])
        
       
    I = np.linspace(0,maxiter-1,5).astype(np.uint16)
    metric=np.array(metric)
    psnr_list.append(metric[-1,0]); ssim_list.append(metric[-1,1])
   
fig, ax = plt.subplots(1,2,figsize=(9,3))
pl = ax[0].plot(metric[:,0])
pl = ax[1].plot(metric[:,1],color='orange')
plt.suptitle('psnr: %.2f, sim: %.2f'%(
    metric[-1,0],metric[-1,1])); 
plt.show()
    
   

fig, ax = plt.subplots(2,5,figsize=(14,4)); 
for j in range(5):
    im = ax[0,j].imshow(img_fidelity[I[j]], cmap='Greys_r', vmin=0, vmax=1)
    ax[0,j].axis('off')
    plt.colorbar(im,ax=ax[0,j])
    

im = ax[1,3].imshow(np.real(x0), cmap='Greys_r',vmin=0,vmax=1)
ax[1,3].axis('off')
plt.colorbar(im,ax=ax[1,3])
im = ax[1,4].imshow(gt.real, cmap='Greys_r')
ax[1,4].axis('off')
plt.colorbar(im,ax=ax[1,4])
fig.tight_layout(pad=.3)
plt.show()

        
table = pd.DataFrame(np.array([ss,np.mean(psnr_list),np.mean(ssim_list)]).reshape(1,3),
                     columns=['stepsize','psnr','ssim'])

TABLE=pd.concat([TABLE,table],ignore_index=True)
TABLE.to_csv('table_%s.csv'%index)

gc.collect()

#%%
    
#analysis

#x=gt
#x = x[::8,::8].real
#i = np.eye(40)
#fi = (np.fft.fft(i))
#fi = np.kron(fi,fi)
#plt.imshow(fi.imag); plt.colorbar()

#fx = np.matmul(fi, np.reshape(x,[1600,1])).reshape(40,40)
#plt.imshow(fx.real); plt.colorbar()
#plt.imshow(np.fft.ifft2(np.fft.fft2(x)).real); plt.colorbar()

##%%

#def DFT_matrix_2d(N):
    #i, j = np.meshgrid(np.arange(N), np.arange(N))
    #A=np.multiply.outer(i.flatten(), i.flatten())
    #B=np.multiply.outer(j.flatten(), j.flatten())
    #omega = np.exp(-2*np.pi*1J/N)
    #W = np.power(omega, A+B)
    #return W

#dftm = DFT_matrix_2d(40)
#fx = np.dot(dftm,np.reshape(x,[1600,1])).reshape(40,40)
#plt.imshow(fx.real); plt.colorbar()
#plt.imshow(dftm.real); plt.colorbar()

#import scipy.linalg
#m = scipy.linalg.dft(40)


