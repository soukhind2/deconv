#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:22:01 2020

@author: sdas
"""
from brainiak.utils import fmrisim
import nibabel
import numpy as np
import matplotlib.pyplot as plt


class loadvolume:
    
    def __init__(self,path):
        self.path = path
    
    def loaddata(self):
        nii = nibabel.load(self.path) # '/Users/sdas/Desktop/FMRIsim/Full_Data/Corr_MVPA_Data_dataspace/Participant_03_rest_run02.nii'
        self.volume = nii.get_data()
        self.dim = self.volume.shape  # What is the size of the volume
        self.dimsize = nii.header.get_zooms()  # Get voxel dimensions from the nifti header
        self.tr = self.dimsize[3]
        if self.tr > 100:  # If high then these values are likely in ms
            self.tr /= 1000
    
    def loadmask(self):
        self.mask, self.template = fmrisim.mask_brain(volume=self.volume, 
                                    mask_self=True,)
    def generate_noise(self):
        noise_dict = {'voxel_size': [self.dimsize[0], self.dimsize[1], 
                                     self.dimsize[2]], 'matched': 1}
        self.noise_dict = fmrisim.calc_noise(volume=self.volume,
                                mask=self.mask,
                                template=self.template,
                                noise_dict=noise_dict,
                                )

        # Calculate the noise given the parameters
        self.noise = fmrisim.generate_noise(dimensions=self.dim[0:3],
                               tr_duration=int(self.tr),
                               stimfunction_tr=[0] * self.dim[3], 
                               mask=self.mask,
                               template=self.template,
                               noise_dict=self.noise_dict,
                               )
        
    def generate_region(self):
        # Create the region of activity where signal will appear
        self.coordinates = np.array([[21,21,21]])  # Where in the brain is the signal
        self.feature_size = 3  # How big, in voxels, is the size of the ROI
        self.signal_volume = fmrisim.generate_signal(dimensions=self.dim[0:3],
                                                feature_type=['cube'],
                                                feature_coordinates=self.coordinates,
                                                feature_size=[self.feature_size],
                                                signal_magnitude=[1],
                                                )
        plt.figure()
        plt.imshow(self.signal_volume[:,:,21] , cmap=plt.cm.gray)
        plt.imshow(self.mask[:, :, 21], cmap=plt.cm.gray, alpha=.5)
        plt.axis('off')
        self.voxels = self.feature_size ** 3
        

    