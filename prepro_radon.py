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


#%%

from scipy.spatial import Delaunay

def in_alpha_shape(p, dt, is_in_alpha):
    simplex_ids = dt.find_simplex(p)
    res = np.full(p.shape[0], False)
    res[simplex_ids >= 0] = is_in_alpha[simplex_ids[simplex_ids >= 0]]  # simplex should be in dt _and_ in alpha
    return res

def circ_radius(p0,p1,p2):
    """
    Vectorized computation of triangle circumscribing radii.
    See for example https://www.cuemath.com/jee/circumcircle-formulae-trigonometry/
    """
    a = p1-p0
    b = p2-p0

    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    norm_a_b = np.linalg.norm(a-b, axis=1)
    cross_a_b = np.cross(a,b)  # 2 * area of triangles(
    return (norm_a*norm_b*norm_a_b) / np.abs(2.0*cross_a_b)


def alpha_shape_delaunay_mask(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points and return the Delaunay triangulation and a boolean
    mask for any triangle in the triangulation whether it belongs to the alpha shape.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :return: Delaunay triangulation dt and boolean array is_in_shape, so that dt.simplices[is_in_alpha] contains
    only the triangles that belong to the alpha shape.
    """
    # Modified and vectorized from:
    # https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points/50714300#50714300

    assert points.shape[0] > 3, "Need at least four points"
    dt = Delaunay(points)

    p0 = points[dt.simplices[:,0],:]
    p1 = points[dt.simplices[:,1],:]
    p2 = points[dt.simplices[:,2],:]

    rads = circ_radius(p0, p1, p2)

    is_in_shape = (rads < alpha)

    return dt, is_in_shape



def get_points(x, y, N, NumPoints, dist):
    points = []

    x0 = np.random.choice(x)
    y0 = np.random.choice(y)
    
    disc = plt.imread('Examples/htc2022_solid_disc_full_recon_fbp_seg.png')
    
    x0 = list(set(x[np.where(disc)[0]]))
    y0 = list(set(x[np.where(disc)[1]]))
    x0.remove(max(x0))
    x0.remove(min(x0))
    y0.remove(max(y0))
    y0.remove(min(y0))
    x0 = np.random.choice(x0)
    y0 = np.random.choice(y0)
    
    flag = True
    while flag:
    
        # xt = np.random.choice(np.linspace(-1/2, 1/2, N))
        # yt = np.random.choice(np.linspace(-1/2, 1/2, N))
        # print('here 1')
        
        t1 = np.random.choice(np.linspace(0.1, dist, N))
        t2 = np.random.choice(np.linspace(0.1, dist, N))
        
        points.append([x0 + t1, y0 + t2])
        
        # if np.sqrt((xt-x0)**2 + (yt-y0)**2) <= dist:
        #     xi = x0 - xt
        #     yi = y0 - yt      
        #     if abs(xi) < x[450] and abs(yi) < y[450]:
        #         points.append([xi, yi])
        #         print('here 2')
        #     else:
        #         print('Point not contained inside disc')
        if len(points) == NumPoints:
            flag = False

    points = np.array(points)
    return points


#%%

#create

x=np.load('radon/mask_example.npy')[::2,::2]
N = x.shape[0]**2
N1 = x.shape[0]; N2=N1

NP=90
theta=np.linspace(0,180-180/NP,NP)

e1 = np.zeros((x.shape[0],x.shape[1])); e1[0,0] = 1;
r1 = radon(e1,theta=theta[0:1]);
M  = np.shape(r1)[0]

# R  = np.zeros((N1*NP,N),dtype=np.float32);


# for k in range(NP):
#     for i in range(N):
#         ei = np.zeros((N1,N2)); 
#         i1 = int(i/N1); i2 = i- i1*N1;
#         ei[i1,i2] = 1;
#         ri = radon(ei,theta=theta[k:(k+1)]);
#         R[k*M:(k+1)*M,i:i+1] = ri.astype(np.float32);
#     print(k)
    
# scipy.sparse.save_npz('fp.npz',scipy.sparse.coo_matrix(R))

R = scipy.sparse.load_npz('fp.npz')
scipy.sparse.save_npz('bp.npz',scipy.sparse.coo_matrix(R.T/NP))


rx = R.dot(x.reshape([N1*N2,1])); rx = rx.reshape(NP,M)
f = np.abs(np.linspace(-1,1,M))
F = np.stack([f for j in range(NP)], axis=0)
np.save('ramp_filter.np',F)
plt.imshow(rx); plt.colorbar()
plt.imshow(radon(x,circle=True,theta=theta,preserve_range=True).T);plt.colorbar()

#%%

def create_masks(NumMasks,mode,R):

    disc = plt.imread('Examples/htc2022_solid_disc_full_recon_fbp_seg.png')
    
    N = disc.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # dist = 0.5
    ones = np.ones([N, N])
    inv_disc = ones - disc
    NumPoints = random.randint(15, 50)
    NumSingleShape = random.randint(4, 10)
    num = 1
    while num <= NumMasks:
        MASK = disc.copy()
        num_shape = 0
        while num_shape < NumSingleShape:
            
            dist = np.random.choice(np.linspace(0.2, 0.5, N))
            points = get_points(x, y, N, NumPoints, dist)
            # points = np.vstack([x, y]).T
            
        
            alpha = random.randint(1, 15)
            
            flag = False
            try:
                dt, is_in_alpha = alpha_shape_delaunay_mask(points, alpha)
                p1 = np.stack((X.ravel(), Y.ravel())).T
                cond = in_alpha_shape(p1, dt, is_in_alpha)
                p2 = p1[cond,:]
            
                cond = np.reshape(cond, [N,N])
                # MASK = np.zeros([N, N])
                MASK[cond] = 0

                # WARUM FUNKTIONIERT DAS NICHT
                if len(np.intersect1d(np.where(cond.flatten()), np.where(inv_disc.flatten()))) != 0:
                    num_shape -= 1
                else:
                    flag = True
                    
            except:
                print('Probably not enough points found')
                num_shape -= 1
                
            num_shape += 1

        if flag:
            plt.figure()
            plt.imshow(MASK)
        
            name = 'mask_' + str(num)
            plt.savefig('radon/masks_as_pdf/%s.pdf'%name)
            plt.close()
            
            m = MASK.astype(np.uint8)[::2,::2]
            # m = R.dot(m.reshape([256**2,1])); m = m.reshape(90,256)
            np.save('radon/%s/'%mode+ name, m)
            plt.figure()
            plt.imshow(m)
            plt.colorbar();plt.show()

            print(str(num) + '/' + str(NumMasks))
            num += 1
        
    return m
        


R = scipy.sparse.load_npz('fp.npz')

m = create_masks(1000,mode='train',R=R) 
m = create_masks(100,mode='evaluation',R=R) 


#%%

N = x.shape[0]**2
N1 = x.shape[0]; N2=N1
M=N1

NP=90
theta=np.linspace(0,180-180/NP,NP)
# RT  = scipy.sparse.lil_matrix(np.zeros((N,M*NP)))
RT = np.zeros((N,M*NP),dtype=np.float32)
rx = radon(x,circle=True,theta=theta,preserve_range=True).T


# for k in range(NP):
#     i=0
#     for i in range(M):
#         ei = np.zeros((M,1)); 
#         i1 = i
#         ei[i1,0] = 1;
#         ri = iradon(ei,theta=theta[k:k+1],filter_name = None, interpolation='linear').reshape([N1**2,1])
#         iii= k*M+i
#         RT[:,iii:iii+1] = ri/NP
#     print(k)
    

# np.savez('fbp0.npz',RT)
# scipy.sparse.save_npz('fbp.npz',scipy.sparse.csr_matrix(RT))
    
    
tmp = np.fft.fftshift(np.fft.fft2((rx)))*F
frx = np.real((np.fft.ifft2(np.fft.fftshift(tmp))))
fbp = iradon(rx.T,theta = theta, filter_name = 'ramp',circle=True,interpolation='linear')
plt.imshow(fbp,vmin=0,vmax=2); plt.colorbar()

fbp2 =np.transpose(R/NP).dot(frx.reshape([M*NP,1])).reshape([N2,N1])
plt.imshow(fbp2,vmin=0,vmax=2); plt.colorbar()





#%%

#analyse radon



#%%


#%%


import scipy
import torch
import numpy as np


x=np.load('radon/train/mask_1000.npy')[::2,::2]
R = scipy.sparse.load_npz('fp.npz')
RT = scipy.sparse.load_npz('fbp.npz')

values = R.data
indices = np.vstack((R.row, R.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape=R.shape
fp = torch.sparse_coo_tensor(i, v, torch.Size(shape), device='cuda')

values = RT.data
indices = RT.indices
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape=RT.shape
fbp = torch.sparse_coo_tensor(i, values=v,size= torch.Size(shape), device='cpu')

t=torch.matmul(fp,torch.Tensor(x.reshape([-1,1])).to('cuda')).reshape([90,256])
