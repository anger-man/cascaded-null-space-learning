# cascaded-null-space-learning
Accelerated MRI reconstruction using data-consistent and uncertainty-aware null space networks

## Data
Download here:
https://fileshare.uibk.ac.at/d/4777303fa7f94b17978d/

## Installation
```
#from github
git clone https://github.com/anger-man/cascaded-null-space-learning
cd cascaded-null-space-learning
conda env create --name your_env_name --file=env_cascaded_null_space_learning.yml
conda activate your_env_name
```

## Methods

### NETT (NETwork Tikhonov)
Journal paper: https://iopscience.iop.org/article/10.1088/1361-6420/ab6d57

**Training**
```
python iternett.py --bs 8 --epochs 60 --method 'nett' --task 'phantom'
```
Available methods: 
- *nett*
- *nettScaled*

Available tasks: 
- *phantom*
- *fastmri*


### Residual U-net
Journal paper: https://ieeexplore.ieee.org/document/7949028

**Training**
```
python residual_net.py --bs 8 --epochs 60 --method 'residual' --task 'phantom' --architecture 'unet'
```
Available methods: 
- *residual*
- *residualIter* (enforcing data consistency by subsequent Tikhonov iterations)

Available tasks: 
- *phantom*
- *fastmri*

Available architectures:
- *unet* (https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- *casnet* (https://www.sciencedirect.com/science/article/pii/S0895611119300990)


### Uncertainty-aware cascaded null space network (Ours)
**Training**
```
python casc_null_space.py --bs 8 --epochs 60 --method 'nullspaceUnc' --task 'phantom'
```
Available methods: 
- *nullspaceUnc* (joint reconstruction and uncertainty estimation)
- *nullspace* (w.o. uncertainty estimation)

Available tasks: 
- *phantom*
- *fastmri*
