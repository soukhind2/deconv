#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:16:41 2020

@author: sdas
"""

import numpy as np
import fmrisim_modified as fmrisim
from tools._dghrf import _dghrf
import statsmodels.api as sm
from scipy.linalg import toeplitz


class expdesign:
    
    def __init__(self,l,u,edur,tevents,
                 signal_mag,loadvolume,
                 distribution,
                 dist_param = None,
                 cue_ratio = None, 
                 noise = False,
                 nonlinear = True, 
                 load = None):
        
        self.lower_isi = l
        self.upper_isi = u
        self.total_events = tevents
        self.burn_in = 5
        self.onsets_A = np.empty((0,1))
        self.onsets_B = np.empty((0,1))
        self.onsets_all = np.empty((0,1))
        self.cue_r = cue_ratio
        self.temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated?
        self.event_duration = edur  # How long is each event
        self.loadvolume = loadvolume
        self.signal_magnitude = signal_mag
        self.noise = noise
        self.nonlin = nonlinear
        self.load = load
        self.distribution = distribution
        self.exp = dist_param

    
    def transient(self,etrain,etrain2,l,u):
        '''
        
        Parameters
        ----------
        def transient : ndarray
        etrain = default weighted stimfunction array
        l = lower bound of isi
        u = upper bound of isi
        self.load = type of transient activity
            'attnmap'- uses the default attention maintenance map for determining
            the amount of maintenance
            'wmmap'- uses the default WM maintenance map for determining
            the amount of maintenance
            'wmflat'- special case when neurons fire at an uniform and even rate
            between two events  namely cue and target
        Returns: 
        Modified height adjusted stimweight array of target with transient properties
        -------
        None.
        '''
        tr = int(self.temporal_res)
        sub_imp = 0.8
        for i in np.arange(0,len(etrain)):
            if etrain[i] == 1:
                if self.load == 'attnmap':
                    
                    if l <= 7 and u <= 9:
                        etrain[ i + 1 : i + 1 + tr * 1 ] = sub_imp                    
                    elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
                          or (l == 8 and u >= 8 and u <= 9)
                          or (l == 9 and u == 9)):
                        etrain[ i + 1 : i + 1 + tr * 4 ] = sub_imp                    
                        
                    elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
                          or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
                          or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
                        etrain[ i + 1 + tr*int(l/2)  : i + 1 + tr*int(l/2) + tr*1 ] = sub_imp                    
        
                        
                    elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
                        or ((l == 10 or l == 11) and u >= 14 and u <= 20)
                        or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
                        or ((l == 14 or l == 15) and u >= 15 and u <= 20)
                        or (l >= 16 and u >= 15 and u <= 20)):
                        etrain[ i + 1 + tr*int(l/2)  : i + 1 + tr*int(l/2) + tr*4 ] = sub_imp
                        
                    return etrain
                elif self.load == 'wmmap':
                    idxA = np.where(etrain == 1)[0][:]
                    idxB = np.where(etrain2 == 1)[0][:]
                    minl = min(len(idxA),len(idxB))
                    for i in range(minl):
                        if idxA[i] > idxB[i]:
                            idxB[i] = 0  #This is used to treat for any error 
                            # sampling, or when the two events are not alternate
                    
                    if ((l <= 9 and u <= 13) or (l <= 5)):
                        for i in range(minl):
                            if idxB[i]!= 0:
                                etrain2[ idxA[i] + 1 : idxB[i] ] = sub_imp
                                
                    else:
                        for i in range(minl):
                            if idxB[i]!= 0:
                                etrain2[ idxA[i] + 1 + tr*int(l/2) : idxB[i] ] = sub_imp 
                    
                    return etrain2

                
                else:
                    raise ValueError('Invalid transient map arguement')

    def create_jitter(self):
        
        if self.distribution not in ['exp','uniform','stochastic_rapid','stochastic_interm','stochastic_slow']:
            raise ValueError('Invalid Distribution')
            
        if self.lower_isi != self.upper_isi and self.distribution != 'uniform':
            a = np.arange(self.lower_isi,self.upper_isi,0.1)

            if self.distribution == 'exp':
                ai = np.arange(a.size)        # an array of the index value for weighting

                if self.exp:
                    w = np.exp(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.exp(ai/30)            # higher weights for larger index values


            elif self.distribution == 'stochastic_rapid':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/50))  #50 is chosen for rapid cycles
                w = w + + abs(min(w))


            elif self.distribution == 'stochastic_interm':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/20))  #20 is chosen for interm cycles
                w = w + + abs(min(w))

            elif self.distribution == 'stochastic_slow':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/6))  #6 is chosen for one slow cycle
                w = w + + abs(min(w))
            
            w /= w.sum()                 # weight must be normalized
            
            self.w = w
            self.a = a
            return w
            
        elif self.distribution == 'uniform' and self.lower_isi != self.upper_isi:
            a = np.arange(self.lower_isi,self.upper_isi,0.1)
            w = np.ones(a.size)
            w/= w.sum()
            self.w = w
            self.a = a
            
            return w
        
        

    def tcourse(self):
        pattern_A = self.cue_r * np.ones((27,1))
        pattern_B = np.ones((27,1))
        time = self.burn_in
        f = 0 # Variable to switch between two conditions 
        nevents = 0

            
        total_time = int(self.loadvolume.dim[3] * self.loadvolume.tr)   # How long is the total event time course
        while time <= (total_time ) :
        #while nevents <= self.total_events:
            if self.lower_isi == self.upper_isi:
                
                if f == 0:
                    self.onsets_A = np.append(self.onsets_A, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + self.lower_isi
                    f = 1
                    nevents = nevents + 1
                else:
                    self.onsets_B = np.append(self.onsets_B, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + self.lower_isi
                    f = 0
                    nevents = nevents + 1
            else:
                        
                if f == 0:
                    self.onsets_A = np.append(self.onsets_A, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + np.random.choice(self.a, size=1, p=self.w)                    
                    
                    f = 1
                    nevents = nevents + 1
                    
                else:
                    self.onsets_B = np.append(self.onsets_B, time)
                    self.onsets_all = np.append(self.onsets_all,time)
                    time = time + self.event_duration + np.random.choice(self.a, size=1, p=self.w)
                   
                    f = 0
                    nevents = nevents + 1

        
        #total_time = time

        self.onsets_A = self.onsets_A.transpose()
        self.onsets_B = self.onsets_B.transpose()
        '''if self.cue_r:
            self.onsets_B = np.sort(np.random.choice(self.onsets_B,
                                                     int(len(self.onsets_B)*self.cue_r)
                                                     ,replace = False))'''
        stimfunc_A = np.empty((0,1))
        stimfunc_B = np.empty((0,1))


        stimfunc_A = fmrisim.generate_stimfunction(onsets=self.onsets_A,
                                                   event_durations=[self.event_duration],
                                                   total_time=total_time,
                                                   temporal_resolution=self.temporal_res,
                                                   )

        stimfunc_B = fmrisim.generate_stimfunction(onsets=self.onsets_B,
                                                   event_durations=[self.event_duration],
                                                   total_time=total_time,
                                                   temporal_resolution=self.temporal_res,
                                                   )
        
        #Transient activity introduced
        if self.load == 'wmmap':
            stimfunc_B = self.transient(stimfunc_A,stimfunc_B,self.lower_isi,self.upper_isi)
        elif self.load == 'attnmap':
            stimfunc_A = self.transient(stimfunc_A,stimfunc_B,self.lower_isi,self.upper_isi)

            # Multiply each pattern by each voxel time course
        weights_A = np.empty((0,1))
        weights_B = np.empty((0,1))
        weights_B1 = np.empty((0,1))

        weights_A = np.matlib.repmat(stimfunc_A, 1, self.loadvolume.voxels).transpose() * pattern_A
        weights_B = np.matlib.repmat(stimfunc_B, 1, self.loadvolume.voxels).transpose() * pattern_B
        weights_B1 = np.matlib.repmat(stimfunc_B, 1, self.loadvolume.voxels).transpose() * pattern_B


        
        # Sum these time courses together
        stimfunc_weighted = np.empty((0,1))

        stimfunc_weighted = weights_B1 + weights_A
        stimfunc_weighted = stimfunc_weighted.transpose()
        self.temp = stimfunc_weighted


        signal_func = fmrisim.convolve_hrf(stimfunction=stimfunc_weighted,
                                           tr_duration=self.loadvolume.tr,
                                           temporal_resolution=self.temporal_res,
                                           scale_function=0,nonlin = self.nonlin)

        # Specify the parameters for signal
        signal_method = 'PSC'

        # Where in the brain are there stimulus evoked voxels
        # The np.where is traversing through the self.signal_volume across every coordinate and looking where the signal is to be 
        # evoked as set before
        signal_idxs = np.where(self.loadvolume.signal_volume == 1)

        # Pull out the voxels corresponding to the noise volume
        noise_func = self.loadvolume.noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :]

        # Compute the signal appropriate scaled
        signal_func_scaled = fmrisim.compute_signal_change(signal_func,
                                                           noise_func.T,
                                                           self.loadvolume.noise_dict,
                                                           magnitude=self.signal_magnitude,
                                                           method=signal_method,)
        
        signal = fmrisim.apply_signal(signal_func_scaled,self.loadvolume.signal_volume,)
        self.temp = signal
        if self.noise:
            
            self.brain = signal + self.loadvolume.noise
        else:
            self.brain = signal
        
        return self.brain

    
class expanalyse:
     
    def __init__(self,data,contrast,expdesign):
         
        self.expdesign = expdesign

        HRF = _dghrf()
        self. boxcar_A = np.zeros(np.size(data,3))
        self. boxcar_B = np.zeros(np.size(data,3))
        
        self.boxcar_A[(self.expdesign.onsets_A/self.expdesign.loadvolume.tr).astype('int')] = 1
        self.conv_boxcar_A = np.convolve(self.boxcar_A,HRF,'same')
       
        self.boxcar_B[(self.expdesign.onsets_B/self.expdesign.loadvolume.tr).astype('int')] = 1
        self.conv_boxcar_B = np.convolve(self.boxcar_B,HRF,'same')
        
        lb = (self.expdesign.loadvolume.coordinates - ((self.expdesign.loadvolume.feature_size - 1) / 2)).astype('int')[0]
        ub = (self.expdesign.loadvolume.coordinates + ((self.expdesign.loadvolume.feature_size - 1) / 2) + 1).astype('int')[0]
        
        roi_brain = data[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2], :]
        roi_brain = roi_brain.reshape((self.expdesign.loadvolume.voxels,data.shape[3]))
        self.roi = roi_brain[13,:].T
        
        self.X = np.zeros((self.conv_boxcar_A.shape[0],2))
        self.design = np.empty((self.boxcar_A.shape[0],2))
        self.X[:,0] = self.conv_boxcar_A
        self.X[:,1] = self.conv_boxcar_B
        self.design[:,0] = self.boxcar_A
        self.design[:,1] = self.boxcar_B
        
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
    
    def calc_FIR(self,ncond,hrflen = 30):
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
                    
                    
    def calc_Fe(self,ncond,hrflen = 30):
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
    

        
         
         

