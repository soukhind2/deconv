#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:06:28 2020

@author: sdas
"""

#%%

from pathlib import Path
from brainiak.utils import fmrisim
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.ndimage as ndimage
import scipy.spatial.distance as sp_distance
import sklearn.manifold as manifold
import scipy.stats as stats
import sklearn.model_selection
import sklearn.svm
import statsmodels.stats.power as smodel
import math
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from pylab import meshgrid, cm, imshow, colorbar
from matplotlib import pyplot
import matplotlib as mpl
import scipy.io as sio
import seaborn as sns
from scipy.optimize import minimize
import scipy.stats as stats
from statsmodels.tsa.ar_model import AutoReg
from scipy.linalg import toeplitz

from avgHRF import avgHRF
from plotdata import plotdata
from _dghrf import _dghrf
#%%

home = str(Path.home())
nii = nibabel.load('/Users/sdas/Desktop/FMRIsim/Full_Data/Corr_MVPA_Data_dataspace/Participant_03_rest_run02.nii')
volume = nii.get_data()
#%%
dim = volume.shape  # What is the size of the volume
dimsize = nii.header.get_zooms()  # Get voxel dimensions from the nifti header
tr = dimsize[3]
if tr > 100:  # If high then these values are likely in ms
    tr /= 1000
print(dim)
#%%
plt.figure()
plt.imshow(volume[:, :, 15, 0], cmap=plt.cm.gray)
plt.axis('off')
f, ax = plt.subplots(1,1, figsize=(10,5))
ax.plot(volume[21,21,21,:])
ax.set_xlabel('TR')
ax.set_ylabel('Voxel Intensity');
mask, template = fmrisim.mask_brain(volume=volume, 
                                    mask_self=True,)
#%%
# Calculate the noise parameters from the data. Set it up to be matched.
noise_dict = {'voxel_size': [dimsize[0], dimsize[1], dimsize[2]], 'matched': 1}
noise_dict = fmrisim.calc_noise(volume=volume,
                                mask=mask,
                                template=template,
                                noise_dict=noise_dict,
                                )

# Calculate the noise given the parameters
noise = fmrisim.generate_noise(dimensions=dim[0:3],
                               tr_duration=int(tr),
                               stimfunction_tr=[0] * dim[3], 
                               mask=mask,
                               template=template,
                               noise_dict=noise_dict,
                               )
#%%
# Create the region of activity where signal will appear
coordinates = np.array([[21,21,21]])  # Where in the brain is the signal
feature_size = 3  # How big, in voxels, is the size of the ROI
signal_volume = fmrisim.generate_signal(dimensions=dim[0:3],
                                        feature_type=['cube'],
                                        feature_coordinates=coordinates,
                                        feature_size=[feature_size],
                                        signal_magnitude=[1],
                                        )
plt.figure()
plt.imshow(signal_volume[:,:,21] , cmap=plt.cm.gray)
plt.imshow(mask[:, :, 21], cmap=plt.cm.gray, alpha=.5)
plt.axis('off')
#%% 
#np.random.seed(1)
# Pull the conical voxel activity from a uniform distribution
pattern_A = np.random.rand(voxels).reshape((voxels, 1))  
pattern_B = np.random.rand(voxels).reshape((voxels, 1))
pattern_A = np.ones((27,1))
pattern_B = 0.4 * np.ones((27,1))
#%%
