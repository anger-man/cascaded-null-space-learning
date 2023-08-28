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
from functions import CtDataGenerator
from functions import plot_images, psnr
import optparse
from ct_models import init_weights, CascNullSpace
import pytorch_ssim
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from scipy.sparse import load_npz

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device,flush=True)

#%%

parser = optparse.OptionParser()
parser.add_option('--wait', action="store", type=int, dest="wait",default=0)
parser.add_option('--lr', action='store', type=float, dest='lr', default=2e-4)
parser.add_option('--method', action='store',type=str,dest='meth', default='nullspaceUnc')
parser.add_option('--task', action='store', type=str, dest='task', default='radon')
parser.add_option('--rec_method', action='store', type=str, dest='rec_method', default = 'pseudoinverse')
parser.add_option('--architecture', action = 'store', type = str, dest = 'arch', default = 'casnet')
parser.add_option('--bs', action = 'store', type=float, dest='bs', default = 4)
parser.add_option('--epochs', action = 'store', type=float, dest='epochs', default = 30)

options,args = parser.parse_args()

time.sleep(options.wait*61)


#%%

#empty cuda cache and check, if GPU is available

torch.cuda.empty_cache()
gc.collect()
train_on_gpu = torch.cuda.is_available()

index = '%s_%s_%s'%(options.rec_method,options.arch, options.meth)

#%%
NP = 90; M = 192; N1= 192
r = load_npz('fpbessel.npz')#/N1
rplus= np.load('fbpbessel.npy')#*N1

values = r.data
indices = np.vstack((r.row, r.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape=r.shape
R = torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)



r_sparse = r
rt_sparse= r.T;




Rplus = torch.Tensor(rplus).to(device)
del rplus


def FP(x):
    y = []
    for k in range(x.size(0)):
        rx = torch.matmul(R,x[k].reshape([-1,1])); rx = rx.reshape(NP,M)
        y.append(rx)
    return torch.stack(y, axis=0)

    
if options.rec_method == 'pseudoinverse':
    def rec_method(y):
        rec = []
        for k in range(y.size(0)):
            rx = torch.matmul(Rplus,y[k].reshape([-1,1])); rx = rx.reshape([1,N1,N1])
            rec.append(rx)
        return torch.stack(rec, axis=0)
    
if options.rec_method == 'landweber':
    Rt = torch.sparse_coo_tensor(torch.LongTensor(np.vstack((r.col,r.row))),
                                 torch.FloatTensor(r.data.T),torch.Size((r.T).shape),
                                 device=device)
    def rec_method(y):
        rec = []
        for k in range(y.size(0)):
            xk = torch.zeros([1,N1,N1]).to(y.get_device())
            for ii in range(25):
                a = torch.matmul(R,xk.reshape([-1,1])).reshape(NP,M)
                b = a-y[k]
                c = torch.matmul(Rt,b.reshape([-1,1])).reshape([1,N1,N1])
                xk = xk - 0.012*c
            rec.append(xk)
        return torch.stack(rec, axis=0)
    
        


#%%

# regularizer = UNet(n_channels =2,f_size=2,out_channels=2, out_acti='tanh')
# regularizer.apply(init_weights)
# regularizer.to(device)

if options.arch == 'unet':
    net = CascNullSpace(n_channels =1,f_size=32,out_channels=1,
                        FP = FP, rec_method = rec_method, single_nsblock = True)
if options.arch == 'casnet':
    net = CascNullSpace(n_channels =1,f_size=32,out_channels=1,
                        FP = FP, rec_method = rec_method, single_nsblock = False)
    
net.apply(init_weights)
net.to(device)
summary(net,[1,1,192,192],depth=4, col_names=(['input_size','output_size']),verbose=1)

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

train_dataset = CtDataGenerator(
    path = data_path,
    img_ids1 = train1)

valid_dataset = CtDataGenerator(
    path = data_path,
    img_ids1 = vali1)

test_dataset = CtDataGenerator(
    path = data_path,
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
if adapt_w:
    w1=.5; w2=.5
    
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
   
    for full in bar:
        full = full.to(device)
        ydata = FP(full)
        recon = rec_method(ydata)
        recon = recon.to(device);
    
        net.train(); #regularizer.eval()
        optim_net.zero_grad()
        inter,net_out,unc = net(recon)
            
        
        loss_image = .7*criterion(full,net_out,unc)+.3*MAE()(full,inter)
        loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],(net_out)[:,0:1])
        
        gloss = w1 * loss_image + w2*loss_ssim
        # if adapt_w:
        #     w1 = .999*w1 + 0.001*(loss_ssim/(loss_ssim+loss_image)).item()
        #     w2 = .999*w2 + 0.001*(loss_image/(loss_ssim+loss_image)).item()
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
        for full in bar:
            full = full.to(device)
            ydata = FP(full)
            recon = rec_method(ydata)
            recon = recon.to(device);
            
          
            inter,net_out,unc = net(recon)
            
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
                plot_images(recon,net_out-recon,full, unc, index, epoch,unc = use_unc)
                save_images = 0
           
    # plt.imshow((recon+net_out)[0,0].cpu());plt.colorbar()
    # plt.imshow((net_out)[0,0].cpu());plt.colorbar()
    # plt.imshow(full[0,0].cpu()); plt.colorbar()
    # plt.imshow(full[0,1].cpu()); plt.colorbar()
            
        
    
    resnet_list.append(resnet_score.item()/len(train_loader.dataset))
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
        torch.save(net.state_dict(), 'weights/%s.pt' %index)
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

net.load_state_dict(torch.load('weights/%s.pt'%index))
net.to(device)
net.eval(); 

preds = []; gts = []; uncs = []; inps = []
psnr_metric = 0.0
ssim_metric = 0.0

with torch.no_grad():
    bar = tq(test_loader, postfix={"net_vali":0.0,"reg_vali":0.0})
    for full in bar:
        full = full.to(device)
        ydata = FP(full)
        recon = rec_method(ydata)
        recon = recon.to(device);
        
        inter,net_out,unc = net(recon)
        
        loss_image = .8*criterion(full,net_out,unc)+.2*MAE()(full,inter)
        loss_ssim = 1-pytorch_ssim.SSIM()(full[:,0:1],net_out[:,0:1])
        # loss_reg = torch.mean(torch.abs(regularizer(PE(coord)(recon+net_out))[0]))
        
            
        gt = full[:,0:1].cpu().numpy(); pred = (net_out)[:,0:1].cpu().numpy()
        
        psnr_metric += psnr(gt, np.clip(pred,0,1))*recon.size(0)
        ssim_metric += 100*ssim(gt[:,0],np.clip(pred,0.,1.)[:,0], data_range=1,
                                channel_axis=0)*recon.size(0)
        
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

# investigate uncertainty estimate


if options.meth == 'nullspaceUnc':
    plt.plot(np.mean(uncs,axis=(1,2)),np.mean(np.abs(gts-preds),axis=(1,2)),'o')
    plt.show()
    x = pd.Series(np.mean(np.abs(gts-preds),axis=(1,2)))
    y = pd.Series(np.mean(uncs,axis=(1,2)))
    df = pd.DataFrame(columns=['x_coord','y_coord'])
    df['y_coord']=x
    df['x_coord']=y
    df.to_csv('xy-coordinates.csv')



#%%

    
indices = np.sort([os.path.join(data_path,'evaluation',f) 
               for f in os.listdir('%s/evaluation'%options.task)])
Y = []; X =[]; 
# U=[]
for i in indices:
    foo = np.load(i)
    X.append(foo)
    # time.sleep(.2)
    
# np.save('U_%s.npy'%options.task,U)


K=np.linspace(0,len(indices)-1,np.min([len(indices),400])).astype(np.uint16)
stepsize = np.linspace(0.1,1,5)
TABLE = pd.DataFrame(columns=['stepsize','psnr','ssim'])
net.to(device)
net.load_state_dict(torch.load('weights/%s.pt' %(index)))
net.eval()

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
ss=0
psnr_list = []; ssim_list = []
for k in K:
    gt = X[k]; metric =[]
    ydata = torch.Tensor(gt).to(device).unsqueeze(0);  
    ydata = FP(ydata)
    recon = rec_method(ydata)
    y = ydata.cpu()[0].numpy()
    x0 = recon.cpu()[0,0].numpy()
    
    maxiter = 1
    s = np.repeat(ss, maxiter); 
    img_fidelity=[]; 
    
    with torch.no_grad():
        init =  net(recon)[1]
    x = init[0,0].cpu().numpy()
    for i in range(maxiter):
        
        #data fidelity
        tmp = (r_sparse.dot(x.reshape([-1,1])).reshape([NP,M])-y)
        dif = rt_sparse.dot(tmp.reshape([-1,1])).reshape([N1,N1])
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
TABLE.to_csv('results/table_%s.csv'%index)

gc.collect()

