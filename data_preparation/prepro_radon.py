import numpy as np
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from scipy import ndimage, misc, sparse
import random
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import scipy
from skimage.transform import radon,iradon
import h5py


#%%

# phantoms schwab antholzer 128x128

# data = (h5py.File('randshepp.mat')['data'][:]).transpose([-1,0,1])
# phantom = data[0,:,:]


# for k in range(data.shape[0]):
#     name = 'phantom_%04d' %k
#     phan = data[k]; phan = phan/np.max(phan)
    
#     if k<2000:
#         mode = 'train'
#     else:
#         mode = 'evaluation'
        
#     np.save('radon/%s/'%mode+ name, phan)
    
#%%

from phantoms_radon import gen_phantom
import odl
import skimage



data_matching = 'exact'

N = 188
shape = [N, N]
Ni=2200

recsp = odl.uniform_discr(min_pt=[-1, -1], max_pt=[1, 1], shape=shape)

F=np.zeros([Ni, shape[0], shape[0]])

for i in np.arange(0,Ni):    
    phantom = gen_phantom(recsp,p=0)
    tmp = phantom.asarray()
    tmp = skimage.transform.rotate(tmp,np.random.choice(range(45)),preserve_range=True)
    F[i]=tmp
    print(i)

data = F

for k in range(data.shape[0]):
    name = 'phantom_%04d' %k
    phan = data[k].astype(np.float32); 
    phan = np.pad(phan,[[2,2],[2,2]])
    phan[phan<0]=0
    phan = phan/np.max(phan)
    
    if k<2000:
        mode = 'train'
    else:
        # plt.imshow(phan);plt.colorbar();plt.show()
        mode = 'evaluation'
        
    np.save('radon/%s/'%mode+ name, phan)


#%%
# helper functions for radon matrix bessel construction
ALPHA = 7
KB_RADIUS = 0.055

def kaiser_bessel(x, alpha, a):
    r = np.linalg.norm(x)
    if r<=a:
        return np.i0(alpha*np.sqrt(1-(r/a)*(r/a)))/np.i0(alpha)
    else:
        return 0

def rt_kb_v(s):
    """vectorized rt_kb"""
    i = 2*KB_RADIUS/(ALPHA*np.i0(ALPHA))
    z = np.zeros(len(s))
    y = i*np.sinh(ALPHA*np.sqrt(1-(s/KB_RADIUS)*(s/KB_RADIUS)))
    x = np.where(np.abs(s)<=KB_RADIUS, y, z)
    return x

def rt_shift(theta, s, a):
    tw = np.array([np.cos(theta), np.sin(theta)])
    return rt_kb_v(s-np.dot(tw,a))

def get_A(N_PHI,N_S,N):
    T = np.linspace(0, np.pi-np.pi/N_PHI, N_PHI) # angles
    S = np.linspace(-1, 1, N_S)
    n = int(np.sqrt(N))
    grid = np.meshgrid(np.linspace(-1,1,n), np.linspace(-1,1,n))
    coord = np.concatenate((grid[0].reshape((-1,1)), grid[1].reshape((-1,1))), axis=1) # N \times 2
    A = np.zeros((N_S*N_PHI, N))
    for n in range(N):
        if n%512 == 0 and n>0:
            print(n, end="\r", flush=True)
        x = np.zeros(N_S*N_PHI)
        for k in range(N_PHI):
            A[k*N_S:(k+1)*N_S,n] = rt_shift(T[k], S, coord[n])
        if n%100 == 0:
            print(str(n) + '/' + str(N))
    return A
    

def get_disk_radius(x):
    n = x.shape[0]
    mid = x.shape[0]//2
    disc = np.zeros((n,n))
    rad = 0.4
    dif = np.where(x>disc,1,0)
    while np.sum(dif)>0:
        rad += 2/n
        for i in np.linspace(-1,1,n):
            for j in np.linspace(-1,1,n):
                if (i**2+j**2)<= rad**2:
                    i1 = int(mid+i*n/2)
                    j1 = int(mid+j*n/2)
                    disc[i1,j1] = 1
        dif = np.where(x>disc,1,0)
    return rad, disc
#%%

# create radon matrix R

x=np.load('radon/train/phantom_1003.npy')
plt.imshow(x);plt.colorbar();plt.show()
N = x.shape[0]**2
N1 = x.shape[0]; N2=N1
NP = 90

N = N1*N1
N_S = 192
al = np.linspace(0, np.pi*(1-1/180), NP)

#%%

# kaiser bessel approach

# R = get_A(NP,N1,N)
# np.save('fpbessel.npy',R)


#%%

# simon approach
def my_radon(f, Nal, Ns, Nx, x, al, s, t):
    
    dt = 3/t.size                                   
    S, T = np.meshgrid(s, t)
    Rf = np.zeros([Ns, Nal])
    
    Nx = f.shape[0]
    
    for i in range(Nal):
        Trot = (S*np.cos(al[i]) - T*np.sin(al[i]) + 1)*Nx/2
        Srot = (S*np.sin(al[i]) + T*np.cos(al[i]) + 1)*Nx/2
        frot = ndimage.map_coordinates(f, [Trot,Srot], order=3)
        fsum = np.sum(frot, axis=0)        
        Rf[:,i] = fsum*dt
        
    return Rf

x  = np.linspace(-1, 1, N1)
s  = np.linspace(-1.5, 1.5, N_S)
t  = np.linspace(-1.5, 1.5, N_S)
R = np.zeros([N_S*NP, N])



for k in range(NP):
    for i in range(N):
        ei = np.zeros((N1,N1)); 
        i1 = int(i/N1); i2 = i- i1*N1;
        ei[i1,i2] = 1;
        ri = my_radon(ei,1,N_S,N1,x,al[k:(k+1)],s,t);
        R[k*N_S:(k+1)*N_S,i:i+1] = ri.astype(np.float32);
    print(k)
    
np.save('fp.npy', R)


#%%

# scikit-image approach

# theta = np.linspace(0, 180*(1-1/NP), NP)
# e1 = np.zeros((x.shape[0],x.shape[1])); e1[0,0] = 1;
# r1 = radon(e1,theta=theta[0:1]);

# R  = np.zeros((N1*NP,N),dtype=np.float32);


# for k in range(NP):
#     for i in range(N):
#         ei = np.zeros((N1,N2)); 
#         i1 = int(i/N1); i2 = i- i1*N1;
#         ei[i1,i2] = 1;
#         ri = radon(ei,theta=theta[k:(k+1)],preserve_range=True);
#         R[k*N1:(k+1)*N1,i:i+1] = ri.astype(np.float32);
#     print(k)

#%%
x=np.load('radon/train/phantom_1003.npy')

R = np.load('fp.npy')

# rad, disc = get_disk_radius(x)
# for k in range(10):
#     try:
#         xx=np.load('radon/train/phantom_%04d.npy'%k)
#         disc += get_disk_radius(xx)[1]
#     except:
#         print(k)
#         continue;
    
# disc[disc>1]=1
# plt.imshow(disc); plt.show()

# m =disc.flatten().astype(np.uint8)
# M=scipy.sparse.coo_matrix(np.diag(m))
# R = R @ M

# R = R.toarray()
theta=al
for k in range(len(theta)):
    if np.abs(theta[k])>2/3*np.pi:
        R[k*N_S:(k+1)*N_S]=0

R = R.astype(np.float32)
R[np.abs(R)<1e-9] = 0
scipy.sparse.save_npz('fp.npz',scipy.sparse.coo_matrix(R))




#%%

import torch
import jax
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


R = scipy.sparse.load_npz('fpbessel.npz')
test = jax.numpy.linalg.svd(R.toarray(), full_matrices=True)

# U, S, V = torch.linalg.svd(torch.Tensor(R.toarray()).to(device), full_matrices=True)
# U = U.cpu().numpy(); S = S.cpu().numpy(); V = V.cpu().numpy()

U,S,V=test

np.save('Ubessel.npy',U); np.save('Sbessel.npy',S); np.save('Vbessel.npy',V)


#%%
U = np.load('U.npy'); S = np.load('S.npy'); V=np.load('V.npy')
alpha = 1e-5 #cut off for bessel
#cut off for fp.npz equals 1e-10
plt.plot(S[S<alpha])
if U.shape[0]<V.shape[0]:
    zz, Z = np.zeros(S.shape, dtype='float32'), np.zeros((S.shape[0],V.shape[0]-S.shape[0]), dtype='float32')
    S_alpha = np.concatenate((np.diag(np.where(np.square(S)>alpha, 1/S, zz)), Z), axis=1)
else:
    zz, Z = np.zeros(S.shape, dtype='float32'), np.zeros((U.shape[0]-S.shape[0],S.shape[0]), dtype='float32')
    S_alpha = np.concatenate((np.diag(np.where(np.square(S)>alpha, 1/S, zz)), Z), axis=0)

VV = V.T
S_alpha = S_alpha.T
U = U.T

# self.R = torch.Tensor(R).to(config.device)
Ralpha = (VV @ S_alpha) @ U

# scipy.sparse.save_npz('fbp.npz',scipy.sparse.coo_matrix(Ralpha))
np.save('fbpbessel.npy', Ralpha)#cheaper


#%%

# load final matrices

R = scipy.sparse.load_npz('fpbessel.npz')
Ralpha = np.load('fbpbessel.npy')

#%%

# check sinogram and properties of a right inverse

x=np.load('radon/train/phantom_%04d.npy'%np.random.choice(range(2000)))
rx = R @ x.flatten(); rx = rx.reshape(NP,N_S)
# rx = rx*limited_mask.cpu().numpy()[0]
plt.imshow(rx); plt.colorbar()
rec = Ralpha @ (rx).flatten();  rec = rec.reshape([N1,N1])
# rec = np.clip(rec,0,1)
# rec=np.clip(rec,0,1)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(x);
ax = fig.add_subplot(122)
ax.imshow(rec,vmin=0,vmax=1);
plt.show()

g = R @ rec.flatten(); g = g.reshape([NP,N_S])
print(np.max(np.abs(rx-g)))
# plt.imshow(np.abs(rx-g));plt.colorbar()



#%%




#%%

    
# disc = plt.imread('Examples/htc2022_solid_disc_full_recon_fbp_seg.png')
# m =disc[::8,::8].flatten()
# M=scipy.sparse.coo_matrix(np.diag(m))
# R = R @ M
# R = R/np.sqrt(NP)
# scipy.sparse.save_npz('fp180.npz',scipy.sparse.coo_matrix(R))

 
# #%%


# from scipy.spatial import Delaunay

# def in_alpha_shape(p, dt, is_in_alpha):
#     simplex_ids = dt.find_simplex(p)
#     res = np.full(p.shape[0], False)
#     res[simplex_ids >= 0] = is_in_alpha[simplex_ids[simplex_ids >= 0]]  # simplex should be in dt _and_ in alpha
#     return res

# def circ_radius(p0,p1,p2):
#     """
#     Vectorized computation of triangle circumscribing radii.
#     See for example https://www.cuemath.com/jee/circumcircle-formulae-trigonometry/
#     """
#     a = p1-p0
#     b = p2-p0

#     norm_a = np.linalg.norm(a, axis=1)
#     norm_b = np.linalg.norm(b, axis=1)
#     norm_a_b = np.linalg.norm(a-b, axis=1)
#     cross_a_b = np.cross(a,b)  # 2 * area of triangles(
#     return (norm_a*norm_b*norm_a_b) / np.abs(2.0*cross_a_b)


# def alpha_shape_delaunay_mask(points, alpha):
#     """
#     Compute the alpha shape (concave hull) of a set of points and return the Delaunay triangulation and a boolean
#     mask for any triangle in the triangulation whether it belongs to the alpha shape.
#     :param points: np.array of shape (n,2) points.
#     :param alpha: alpha value.
#     :return: Delaunay triangulation dt and boolean array is_in_shape, so that dt.simplices[is_in_alpha] contains
#     only the triangles that belong to the alpha shape.
#     """
#     # Modified and vectorized from:
#     # https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points/50714300#50714300

#     assert points.shape[0] > 3, "Need at least four points"
#     dt = Delaunay(points)

#     p0 = points[dt.simplices[:,0],:]
#     p1 = points[dt.simplices[:,1],:]
#     p2 = points[dt.simplices[:,2],:]

#     rads = circ_radius(p0, p1, p2)

#     is_in_shape = (rads < alpha)

#     return dt, is_in_shape



# def get_points(x, y, N, NumPoints, dist):
#     points = []

#     x0 = np.random.choice(x)
#     y0 = np.random.choice(y)
    
#     disc = plt.imread('Examples/htc2022_solid_disc_full_recon_fbp_seg.png')
    
#     x0 = list(set(x[np.where(disc)[0]]))
#     y0 = list(set(x[np.where(disc)[1]]))
#     x0.remove(max(x0))
#     x0.remove(min(x0))
#     y0.remove(max(y0))
#     y0.remove(min(y0))
#     x0 = np.random.choice(x0)
#     y0 = np.random.choice(y0)
    
#     flag = True
#     while flag:
    
#         # xt = np.random.choice(np.linspace(-1/2, 1/2, N))
#         # yt = np.random.choice(np.linspace(-1/2, 1/2, N))
#         # print('here 1')
        
#         t1 = np.random.choice(np.linspace(0.1, dist, N))
#         t2 = np.random.choice(np.linspace(0.1, dist, N))
        
#         points.append([x0 + t1, y0 + t2])
        
#         # if np.sqrt((xt-x0)**2 + (yt-y0)**2) <= dist:
#         #     xi = x0 - xt
#         #     yi = y0 - yt      
#         #     if abs(xi) < x[450] and abs(yi) < y[450]:
#         #         points.append([xi, yi])
#         #         print('here 2')
#         #     else:
#         #         print('Point not contained inside disc')
#         if len(points) == NumPoints:
#             flag = False

#     points = np.array(points)
#     return points

# def filtering(g):
#     g = g.transpose()
    
#     # g = g.transpose()
#     filt_len = g.shape[0]

#     freqs = np.linspace(-1, 1, filt_len);
#     myFilter = np.abs(freqs);
    
#     Filter = np.zeros_like(g)
#     for i in range(g.shape[1]):
#         Filter[:,i] = myFilter
    
#     gF = np.fft.fftshift(np.fft.fft(g, axis=0), axes=0)
#     gfilt = np.multiply(gF, Filter)
#     gfilt = np.fft.ifftshift(gfilt, axes=0)
#     gfilt = np.fft.ifft(gfilt, axis=0)
#     gfilt = np.real(gfilt)
#     gfilt = gfilt.transpose()
    
#     return gfilt


# def filter(g,F):
#     tmp = np.fft.fftshift(np.fft.fft2((g)))*F
#     frx = np.real((np.fft.ifft2(np.fft.fftshift(tmp))))
#     return frx

# from numpy.fft import fftshift
# from numpy.fft import fft, ifft

# def arange2(start, stop=None, step=1):
#     """#Modified version of numpy.arange which corrects error associated with non-integer step size"""
#     if stop == None:
#         a = np.arange(start)
#     else: 
#         a = np.arange(start, stop, step)
#         if a[-1] > stop-step:   
#             a = np.delete(a, -1)
#     return a


# def projFilter(sino):
#     """filter projections. Normally a ramp filter multiplied by a window function is used in filtered
#     backprojection. The filter function here can be adjusted by a single parameter 'a' to either approximate
#     a pure ramp filter (a ~ 0)  or one that is multiplied by a sinc window with increasing cutoff frequency (a ~ 1).
#     Credit goes to Wakas Aqram. 
#     inputs: sino - [n x m] numpy array where n is the number of projections and m is the number of angles used.
#     outputs: filtSino - [n x m] filtered sinogram array"""
    
#     a = 0.001;
#     sino = sino.transpose()
#     projLen, numAngles = sino.shape
#     step = 2*np.pi/projLen
#     w = arange2(-np.pi, np.pi, step)
#     if len(w) < projLen:
#         w = np.concatenate([w, [w[-1]+step]]) #depending on image size, it might be that len(w) =  
#                                               #projLen - 1. Another element is added to w in this case
#     rn1 = abs(2/a*np.sin(a*w/2));  #approximation of ramp filter abs(w) with a funciton abs(sin(w))
#     rn2 = np.sin(a*w/2)/(a*w/2);   #sinc window with 'a' modifying the cutoff freqs
#     r = rn1*(rn2)**2;              #modulation of ramp filter with sinc window
    
#     filt = fftshift(r)/3
#     filtSino = np.zeros((projLen, numAngles))
#     for i in range(numAngles):
#         projfft = fft(sino[:,i])
#         filtProj = projfft*filt
#         filtSino[:,i] = np.real(ifft(filtProj))

#     return filtSino.transpose()


# def my_ramp(sino):
#     num_thetas,size=sino.shape
#     n = np.concatenate(
#         (
#             # increasing range from 1 to size/2, and again down to 1, step size 2
#             np.arange(size / 2-1 ,0, -2, dtype=int),
#             np.arange(1,size/2, 2, dtype=int),
#         )
#     )
#     f = np.zeros(size)
#     f[32] = 0.25
#     f[1::2] = -1 / (np.pi*n) ** 2
#     f = 2*np.real(fft(fftshift(f)))*2
#     # omega = np.pi * np.fft.fftfreq(size)[1:]
#     # f[1:] *= np.sin(omega) / omega
   
    
#     filter_sinogram=np.zeros((num_thetas,size))
#     for i in range(num_thetas):
#         proj_fft=fft(sino[i,:])
#         filter_proj=proj_fft*f
#         filter_sinogram[i,:]=np.real(ifft(filter_proj))
    
#     return filter_sinogram








# #%%


# #%%
# # scipy.sparse.save_npz('fbpbessel.npz',scipy.sparse.coo_matrix(Ralpha))
# # np.save('fbpbessel.npy', Ralpha)#cheaper

# # rec = Ralpha @ rx.flatten(); rec = rec.reshape([N1,N1])


# # t1=radon(iradon(radon(x,theta,preserve_range=True),theta,filter_name='ramp'),theta,preserve_range=True)
# # t2=radon(x,theta,preserve_range=True)
# #not exact pseudoinverse

# # np.save('ramp_filter.np',F)


# #%%

# # pseudoinverse via iradon

# # N = x.shape[0]**2
# # N1 = x.shape[0]; N2=N1
# # M=N1

# # NP=90
# # theta=np.linspace(0,180-180/NP,NP)
# # RT  = scipy.sparse.lil_matrix(np.zeros((N,M*NP)))
# # RT = np.zeros((N,M*NP),dtype=np.float32)
# # rx = radon(x,circle=True,theta=theta,preserve_range=True).T


# # for k in range(NP):
# #     i=0
# #     for i in range(M):
# #         ei = np.zeros((M,1)); 
# #         i1 = i
# #         ei[i1,0] = 1;
# #         ri = iradon(ei,theta=theta[k:k+1],filter_name = None, interpolation='linear').reshape([N1**2,1])
# #         iii= k*M+i
# #         RT[:,iii:iii+1] = ri/NP
# #     print(k)
    

# # np.savez('fbp0.npz',RT)
# # scipy.sparse.save_npz('fbp.npz',scipy.sparse.csr_matrix(RT))





# #%%

# def create_masks(NumMasks,mode,R):

#     disc = plt.imread('Examples/htc2022_solid_disc_full_recon_fbp_seg.png')
    
#     N = disc.shape[0]
#     x = np.linspace(-1, 1, N)
#     y = np.linspace(-1, 1, N)
#     X, Y = np.meshgrid(x, x)

#     # dist = 0.5
#     ones = np.ones([N, N])
#     inv_disc = ones - disc
#     NumPoints = random.randint(15, 50)
#     NumSingleShape = random.randint(4, 10)
#     num = 1
#     while num <= NumMasks:
#         MASK = disc.copy()
#         num_shape = 0
#         while num_shape < NumSingleShape:
            
#             dist = np.random.choice(np.linspace(0.2, 0.5, N))
#             points = get_points(x, y, N, NumPoints, dist)
#             # points = np.vstack([x, y]).T
            
        
#             alpha = random.randint(1, 15)
            
#             flag = False
#             try:
#                 dt, is_in_alpha = alpha_shape_delaunay_mask(points, alpha)
#                 p1 = np.stack((X.ravel(), Y.ravel())).T
#                 cond = in_alpha_shape(p1, dt, is_in_alpha)
#                 p2 = p1[cond,:]
            
#                 cond = np.reshape(cond, [N,N])
#                 # MASK = np.zeros([N, N])
#                 MASK[cond] = 0

#                 # WARUM FUNKTIONIERT DAS NICHT
#                 if len(np.intersect1d(np.where(cond.flatten()), np.where(inv_disc.flatten()))) != 0:
#                     num_shape -= 1
#                 else:
#                     flag = True
                    
#             except:
#                 print('Probably not enough points found')
#                 num_shape -= 1
                
#             num_shape += 1

#         if flag:
#             plt.figure()
#             plt.imshow(MASK)
        
#             name = 'mask_' + str(num)
#             plt.savefig('radon/masks_as_pdf/%s.pdf'%name)
#             plt.close()
            
#             m = MASK.astype(np.uint8)[::2,::2]
#             # m = R.dot(m.reshape([256**2,1])); m = m.reshape(90,256)
#             np.save('radon/%s/'%mode+ name, m)
#             plt.figure()
#             plt.imshow(m)
#             plt.colorbar();plt.show()

#             print(str(num) + '/' + str(NumMasks))
#             num += 1
        
#     return m
        


# R = scipy.sparse.load_npz('fp.npz')

# m = create_masks(1000,mode='train',R=R) 
# m = create_masks(100,mode='evaluation',R=R) 








# #%%

# # make sparse coo tensors

# # import scipy
# # import torch
# # import numpy as np


# # x=np.load('radon/train/mask_1000.npy')[::2,::2]
# # R = scipy.sparse.load_npz('fp.npz')
# # RT = scipy.sparse.load_npz('fbp.npz')

# # values = R.data
# # indices = np.vstack((R.row, R.col))
# # i = torch.LongTensor(indices)
# # v = torch.FloatTensor(values)
# # shape=R.shape
# # fp = torch.sparse_coo_tensor(i, v, torch.Size(shape), device='cuda')

# # values = RT.data
# # indices = RT.indices
# # i = torch.LongTensor(indices)
# # v = torch.FloatTensor(values)
# # shape=RT.shape
# # fbp = torch.sparse_coo_tensor(i, values=v,size= torch.Size(shape), device='cpu')

# # t=torch.matmul(fp,torch.Tensor(x.reshape([-1,1])).to('cuda')).reshape([90,256])
