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
import importlib
importlib.reload(fmrisim)

class expdesign:
    
    def __init__(self,edur,tevents,
                 signal_mag,loadvolume,
                 distribution,
                 dist_param = None,
                 cue_ratio = None, 
                 null_ratio = None,
                 noise = False,
                 nonlinear = True, 
                 load = None,
                ):
        
        # self.lower_isi = l
        # self.upper_isi = u
        self.total_events = tevents
        self.burn_in = 5
        self.onsets_A = np.empty((0,1))
        self.onsets_B = np.empty((0,1))
        self.onsets_all = np.empty((0,1))
        self.cue_ratio = cue_ratio # Value between 0-1
        self.null_ratio = null_ratio # Value between 0-1
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

    def create_jitter(self, lower_isi, upper_isi, distribution):
        
        if distribution not in ['exp','uniform','stochastic_rapid','stochastic_interm','stochastic_slow']:
            raise ValueError('Invalid Distribution')
            
        if lower_isi != upper_isi and distribution != 'uniform':
            a = np.arange(lower_isi, upper_isi,0.1)
            if distribution == 'exp':
                ai = np.arange(a.size)        # an array of the index value for weighting

                if self.exp:
                    w = np.exp(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.exp(ai/ai[len(ai) - 1])            # higher weights for larger index values


            elif distribution == 'stochastic_rapid':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/50))  #50 is chosen for rapid cycles
                w = w + + abs(min(w))


            elif distribution == 'stochastic_interm':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/20))  #20 is chosen for interm cycles
                w = w + + abs(min(w))

            elif distribution == 'stochastic_slow':
                ai = np.arange(a.size)        # an array of the index value for weighting
                if self.exp:
                    w = np.cos(ai/self.exp)            # higher weights for larger index values
                else:
                    w = np.cos(ai/(a.size/6))  #6 is chosen for one slow cycle
                w = w + + abs(min(w))
            
            w /= w.sum()                 # weight must be normalized
            
            # self.w = w
            # self.a = a
            return (w, a)
            
        elif distribution == 'uniform' and lower_isi != self.upper_isi:
            a = np.arange(lower_isi,self.upper_isi,0.1)
            w = np.ones(a.size)
            w/= w.sum()
            # self.w = w
            # self.a = a
            
            return (w, a)
        
    def generate_tcourse(self, configurations, total_time, temporal_resolution):
        tcourse_groups = []
        paradigm_configs = []
        time = 0
        for config in configurations:
            current_event_duration = 0
            current_config_tcourse = []
            current_paradigm_configs = []
            for i in range(0, len(config)):
                event = config[i]
                if event["type"] == "event":
                    current_event_duration += event["event-duration"]
                    current_config_tcourse.append({ "duration": event["event-duration"], "type": "event", "name": event["name"] })
                elif event["type"] == "inter-events":
                    lsis = event["lsis"]
                    usis = event["usis"]
                    distribution = event["distribution"]
                    paradigm = event["paradigm"]
                    duration = 0
                    if lsis == usis:
                        duration = lsis
                        current_config_tcourse.append({ "duration": lsis, "type": "inter-events" })
                    else:
                        (w, a) = self.create_jitter(lsis, usis, distribution)
                        random_duration = int(np.random.choice(a, size=1, p=w)[0])
                        duration = random_duration
                        current_config_tcourse.append({ "duration": random_duration, "type": "inter-events" })
                        
                    if paradigm is not None:
                        previous_event_duration = current_config_tcourse[len(current_config_tcourse) - 2]["duration"]
                        previous_event = time + current_event_duration - previous_event_duration
                        current_paradigm_configs.append({ "paradigm": paradigm, "first_event_index": previous_event * temporal_resolution, "first_event_duration": previous_event_duration, "second_event_index": (time + current_event_duration + current_config_tcourse[len(current_config_tcourse) - 1]["duration"]) * temporal_resolution, "lsis": lsis, "usis": usis })
                    
                    current_event_duration += duration
                
            time += current_event_duration
            if time > total_time:
                break;
            else:
                tcourse_groups.append(current_config_tcourse)
                paradigm_configs.append(current_paradigm_configs)
                
        return (tcourse_groups, paradigm_configs)
    
    def convert_tcourse_info_array(self, tcourse_info_array):
        tcourse = []
        time_point = 0
        for tcourse_info in tcourse_info_array:
            if tcourse_info["type"] == "event":
                tcourse.append(time_point)
            time_point += tcourse_info["duration"]
            
        return tcourse
            
    
    def apply_transients(self, tcourse_groups, paradigm_configs, temporal_resolution, total_time, sub_imp = 0.8):
        tcourse = []
        event_durations = []
        for tcourse_info_array in tcourse_groups:
            tcourse.extend(self.convert_tcourse_info_array(tcourse_info_array))
            event_durations.extend(map(lambda info: info["duration"], filter(lambda info: info["type"] == "event", tcourse_info_array)))
            
        print((tcourse, event_durations))
            
        stimuli_function = fmrisim.generate_stimfunction(onsets = tcourse,
                                                   event_durations = event_durations,
                                                   total_time = total_time,
                                                   temporal_resolution = temporal_resolution,
                                                   )
        print(stimuli_function)
        for paradigm_config_array in paradigm_configs:
            for paradigm_config in paradigm_config_array:
                l = paradigm_config["lsis"] / 1000
                u = paradigm_config["usis"] / 1000
                paradigm = paradigm_config["paradigm"]
                i = paradigm_config["first_event_index"]
                j = paradigm_config["second_event_index"]
                tr = temporal_resolution
                if paradigm == 'attnmap':
                    if l <= 7 and u <= 9:
                        stimuli_function[ i + 1 : i + 1 + tr * 1 ] = sub_imp                    
                    elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
                          or (l == 8 and u >= 8 and u <= 9)
                          or (l == 9 and u == 9)):
                        stimuli_function[ i + 1 : i + 1 + tr * 4 ] = sub_imp                    

                    elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
                          or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
                          or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
                        stimuli_function[ i + 1 + tr * int(l/2)  : i + 1 + tr * int(l/2) + tr * 1 ] = sub_imp                    


                    elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
                        or ((l == 10 or l == 11) and u >= 14 and u <= 20)
                        or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
                        or ((l == 14 or l == 15) and u >= 15 and u <= 20)
                        or (l >= 16 and u >= 15 and u <= 20)):
                        stimuli_function[ i + 1 + tr * int(l/2)  : i + 1 + tr * int(l/2) + tr * 4 ] = sub_imp

                elif paradigm == 'wmmap':
                    # idxA = np.where(etrain == 1)[0][:]
                    # idxB = np.where(etrain2 == 1)[0][:]
                    # minl = min(len(idxA),len(idxB))
                    # for i in range(minl):
                    #     if idxA[i] > idxB[i]:
                    #         idxB[i] = 0  #This is used to treat for any error 
                    #         # sampling, or when the two events are not alternate

                    if ((l <= 9 and u <= 13) or (l <= 5)):
                        stimuli_function[ i + 1 : j ] = sub_imp

                    else:
                        stimuli_function[ i + 1 + tr * int(l/2) : j ] = sub_imp 

                else:
                    raise ValueError('Invalid transient map arguement')
        return stimuli_function
        

    
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
    
    

        
         
         

