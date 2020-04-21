#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:16:41 2020

@author: sdas
"""

import numpy as np
from brainiak.utils import fmrisim
from tools._dghrf import _dghrf
import statsmodels.api as sm
from scipy.linalg import toeplitz


class expdesign:
    
    def __init__(self,cue_ratio,l,u,edur,tevents,signal_mag,loadvolume):
        self.lower_isi = l
        self.upper_isi = u
        self.total_events = tevents
        self.burn_in = 1
        self.onsets_A = np.empty((0,1))
        self.onsets_all = np.empty((0,1))
        self.cue_r = cue_ratio
        self.temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated?
        self.event_duration = edur  # How long is each event
        self.loadvolume = loadvolume
        self.signal_magnitude = signal_mag
    

    def tcourse(self):
        pattern_A = np.ones((27,1))
        
        time = self.burn_in
        nevents = 0
        total_time = int(self.loadvolume.dim[3] * self.loadvolume.tr) + self.burn_in  # How long is the total event time course
        while time <= (total_time - 5) :
        #while nevents <= self.total_events:

            self.onsets_A = np.append(self.onsets_A, time)
            self.onsets_all = np.append(self.onsets_all,time)
            time = time + self.event_duration + np.random.uniform(self.lower_isi, 
                                                             self.upper_isi)
            nevents = nevents + 1
 
        

        #total_time = time

        self.onsets_A = self.onsets_A[:-2].transpose()
        #self.onsets_B = np.sort(np.random.choice(self.onsets_B,
                                                 #int(len(self.onsets_B)*self.cue_r)
                                                 #,replace = False))
        stimfunc_A = np.empty((0,1))
        #stimfunc_B = np.empty((0,1))


        stimfunc_A = fmrisim.generate_stimfunction(onsets=self.onsets_A,
                                                   event_durations=[self.event_duration],
                                                   total_time=total_time,
                                                   temporal_resolution=self.temporal_res,
                                                   )

        
        

        # Multiply each pattern by each voxel time course
        weights_A = np.empty((0,1))

        weights_A = np.matlib.repmat(stimfunc_A, 1, self.loadvolume.voxels).transpose() * pattern_A

        # Sum these time courses together
        stimfunc_weighted = np.empty((0,1))
        stimfunc_weighted = weights_A
        stimfunc_weighted = stimfunc_weighted.transpose()
        
        signal_func = fmrisim.convolve_hrf(stimfunction=stimfunc_weighted,
                                           tr_duration=self.loadvolume.tr,
                                           temporal_resolution=self.temporal_res,
                                           scale_function=0,)
        
        # Specify the parameters for signal
        #signal_method = 'CNR_Amp/Noise-SD'
        signal_method = 'PSC' 
        # Where in the brain are there stimulus evoked voxels
        # The np.where is traversing through the self.signal_volume across every coordinate and looking where the signal is to be 
        # evoked as set before
        signal_idxs = np.where(self.loadvolume.signal_volume == 1)

        # Pull out the voxels corresponding to the noise volume
        noise_func = self.loadvolume.noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :]

        # Compute the signal appropriate scaled
        signal_func_scaled = fmrisim.compute_signal_change(signal_func,
                                                           noise_func,
                                                           self.loadvolume.noise_dict,
                                                           magnitude=self.signal_magnitude,
                                                           method=signal_method,)

        signal = fmrisim.apply_signal(signal_func_scaled,self.loadvolume.signal_volume,)

        self.brain = signal + self.loadvolume.noise
        
        return self.brain
    
class expanalyse:
     
    def __init__(self,data,contrast,expdesign):
         
        self.expdesign = expdesign

        HRF = _dghrf()
        self. boxcar_A = np.zeros(np.size(data,3))
        
        self.boxcar_A[(self.expdesign.onsets_A/self.expdesign.loadvolume.tr).astype('int')] = 1
        self.conv_boxcar_A = np.convolve(self.boxcar_A,HRF,'same')
       
        
        lb = (self.expdesign.loadvolume.coordinates - ((self.expdesign.loadvolume.feature_size - 1) / 2)).astype('int')[0]
        ub = (self.expdesign.loadvolume.coordinates + ((self.expdesign.loadvolume.feature_size - 1) / 2) + 1).astype('int')[0]
        
        roi_brain = data[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2], :]
        roi_brain = roi_brain.reshape((self.expdesign.loadvolume.voxels,data.shape[3]))
        self.roi = roi_brain[13,:].T
        
        self.X = np.zeros((self.conv_boxcar_A.shape[0],1))
        self.design = np.empty((self.boxcar_A.shape[0],1))
        self.X[:,0] = self.conv_boxcar_A
        self.design[:,0] = self.boxcar_A
        self.C = contrast
        
        
    def calc_Fd(self):

        temp = sm.OLS(self.roi,sm.add_constant(self.X)).fit()
        res_resid = sm.OLS(temp.resid[1:], temp.resid[:-1]).fit()
        rho = res_resid.params
        order = toeplitz(np.arange(len(self.roi)))
        sigma = rho**order
        gls_model = sm.GLS(self.roi, self.X, sigma=sigma)
        gls_results = gls_model.fit()
        med = gls_model.cholsigmainv * sigma * gls_model.cholsigmainv.T
        m = np.dot(np.dot(gls_model.pinv_wexog , med) , gls_model.pinv_wexog.T)
        o = self.C*m*self.C.T
        out = 1/np.matrix.trace(o)
        return out
    
    def calc_FIR(self,ncond,hrflen = 16):
        nCond = ncond
        hrflen = hrflen
        self.X_FIR = np.zeros((len(self.roi),hrflen*nCond))
        
        for iC in range(0,nCond):
            temp = np.nonzero(self.design[:,iC])
            onsets = []
            for item in temp:
                onsets.extend(item)
    
            col = iC + 1
            idxCols = np.arange(((col-1)*hrflen+1),(col*hrflen))
            for jO in range(0,len(onsets)-1):
                idxRows = np.arange(onsets[jO],(onsets[jO]+hrflen-1))
                for kR in range(0,len(idxRows)):
                    if idxRows[kR] < len(self.roi):
                        self.X_FIR[int(idxRows[kR]),int(idxCols[kR])] = 1
                    else:
                        break
        return self.X_FIR
                    
                    
    def calc_Fe(self,ncond,hrflen = 16):
        hrflen = hrflen
        X_FIR = self.calc_FIR(ncond)
        temp = sm.OLS(self.roi,sm.add_constant(X_FIR)).fit()
        res_resid = sm.OLS(temp.resid[1:], temp.resid[:-1]).fit()
        rho = res_resid.params
        order = toeplitz(np.arange(len(self.roi)))
        sigma = rho**order
        gls_model = sm.GLS(self.roi, X_FIR, sigma=sigma)
        gls_results = gls_model.fit()
        med = gls_model.cholsigmainv * sigma * gls_model.cholsigmainv.T
        m = np.dot(np.dot(gls_model.pinv_wexog , med) , gls_model.pinv_wexog.T)
        C = np.array(np.kron(self.C, np.eye(hrflen)))
        o = np.dot(np.dot(C,m),C.T)
        out = 1/np.matrix.trace(o)
        return out
    

        
         
         

