# cascaded-null-space-learning
Image reconstruction using data-consistent and uncertainty-aware null space networks

Applications:
- limited angle CT
- accelerated MRI

## Paper
**Uncertainty-Aware Null Space Networks for Data-Consistent Image Reconstruction**

https://arxiv.org/abs/2304.06955

## Data
Download here: https://fileshare.uibk.ac.at/d/07248c5a6fc442c1acaf/

## Installation
```
#from github
git clone https://github.com/anger-man/cascaded-null-space-learning
cd cascaded-null-space-learning
conda env create --name your_env_name --file=environment_pytorch.yml
conda activate your_env_name
```

## 4x accelerated MRI reconstruction on fastMRI data 
fastMRI project page: https://fastmri.org/

**Null space network**
```
python mri_casc_null_space.py --bs 8 --epochs 30 --method 'nullspace' --task 'fastmri' --architecture ARCHITECTURE
```

**Uncertainty-aware null space network**
```
python mri_casc_null_space.py --bs 8 --epochs 30 --method 'nullspaceUnc' --task 'fastmri' --architecture ARCHITECTURE
```

## limited angle CT reconstruction on phantom data 
Download Radon matrix and corresponding pseudoinverse based on Kaiser-Bessel functions: https://fileshare.uibk.ac.at/d/dc44b035a75a4b24945a/

**Null space network**
```
python ct_casc_null_space.py --bs 8 --epochs 30 --method 'nullspace' --task 'radon' --architecture ARCHITECTURE
```

**Uncertainty-aware null space network**
```
python ct_casc_null_space.py --bs 8 --epochs 30 --method 'nullspaceUnc' --task 'radon' --architecture ARCHITECTURE
```

## Available architectures: 
- 'unet' (https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- 'casnet' (https://www.sciencedirect.com/science/article/pii/S0895611119300990)
