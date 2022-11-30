# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 12:16:52 2022

@author: Ahsan
plotting the relationship plot for Christoph work
"""

#%%
'''
libraries
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import savefig
import seaborn as sns
from scipy.stats import pearsonr
import os
#%% loading the data

dtf = pd.read_csv("xy-coordinates.csv")
corr, _ = pearsonr(dtf['x_coord'], dtf['y_coord'])
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":18,"axes.labelsize":12})
sns.set(font_scale=1.5)
sns.set_style("ticks", {"font.family": "Times New Roman"})
#sns.set_theme(style="whitegrid")
plt.figure(figsize=(5,5))
pairplt = sns.regplot(dtf['x_coord'], dtf['y_coord'])
pairplt.set_xlabel('$Mean \ \ Uncertainty$', labelpad=2)
pairplt.set_ylabel('$Mean \ \ Abs \ \ Residual$',  labelpad=2)
textstr =  r'$PCC = %.2f$' % (corr)
props = dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.4)
pairplt.text(0.55, 0.09, textstr, transform=pairplt.transAxes, fontsize=18, bbox=props)
plt.title('$phantom$', pad=5)
figure = pairplt.get_figure()    
figure.savefig('phantom_relation.png', dpi=600)
plt.show()
#dtf.head()


