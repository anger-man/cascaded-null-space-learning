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
