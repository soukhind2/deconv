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
    
    def __init__(self, loadvolume,
                 signal_magnitude,
                 null_ratio = None,
                 noise = False,
                 nonlinear = True, 
                 dist_parameter = None
                ):
        
        self.null_ratio = null_ratio # Value between 0-1
        self.temporal_res = 10.0 # How many timepoints per second of the stim function are to be generated? ??
        
        self.loadvolume = loadvolume
        self.signal_magnitude = signal_magnitude
        self.noise = noise
        self.nonlinear = nonlinear
        self.exp = dist_parameter

    def __create_jitter(self, lower_isi, upper_isi, distribution):
        
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
        
    def generate_tcourse(self, configurations, total_time):
        print(configurations, total_time)
        tcourse_groups = []
        paradigm_configs = []
        time = 0
        # for config in configurations:
        while True:
            config = configurations[np.random.choice(np.arange(0, len(configurations)), size = 1)[0]]
            current_trial_event_duration = 0
            current_config_tcourse = []
            current_paradigm_configs = []
            for i in range(0, len(config)):
                event = config[i]
                if event["type"] == "event":
                    current_trial_event_duration += event["event-duration"]
                    current_config_tcourse.append({ "duration": event["event-duration"], "type": "event", "name": event["name"], "intensity": float(event["intensity"]) if event["intensity"] is not None else 1.0 })
                elif event["type"] == "inter-events":
                    lsis = event["lsis"]
                    usis = event["usis"]
                    distribution = event["distribution"]
                    paradigm = event["paradigm"]
                    duration = 0
                    if lsis == usis:
                        duration = lsis
                        current_config_tcourse.append({ "duration": duration, "type": "inter-events" })
                    else:
                        (w, a) = self.__create_jitter(lsis, usis, distribution)
                        random_duration = np.random.choice(a, size = 1, p = w).astype('float16')[0]
                        duration = random_duration
                        current_config_tcourse.append({ "duration": duration, "type": "inter-events" })
                        
                    if paradigm is not None:
                        previous_event_ending = time + current_trial_event_duration
                        current_paradigm_configs.append({ "paradigm": paradigm, "inter_event_starting": previous_event_ending, "inter_event_duration": duration, "inter_event_ending": (previous_event_ending + duration), "lsis": lsis, "usis": usis })
                    
                    current_trial_event_duration += duration
                
            time += current_trial_event_duration
            if time > total_time:
                break;
            else:
                tcourse_groups.append(current_config_tcourse)
                paradigm_configs.append(current_paradigm_configs)
                
        print((tcourse_groups, paradigm_configs))
        return (tcourse_groups, paradigm_configs)
    
    
    
    def __convert_tcourse_info_array(self, tcourse_info_array):
        tcourse = []
        time_point = 0
        for tcourse_info in tcourse_info_array:
            if tcourse_info["type"] == "event":
                tcourse.append({ "time_point": time_point, "event_name": tcourse_info["name"], "intensity": tcourse_info["intensity"] })
            time_point += tcourse_info["duration"]
            
        return (tcourse, time_point)
            
        
    def __fabricate_stimuli_function(self, tcourse_groups, temporal_resolution, total_time):
        tcourse = []
        event_durations = []
        event_onsets = []
        onset_intensities = []
        previous_trial_ending_time_point = 0;
        for tcourse_info_array in tcourse_groups:
            (current_event_onsets, next_time_point) = self.__convert_tcourse_info_array(tcourse_info_array)
            for temp_onset in current_event_onsets:
                temp_onset["time_point"] += previous_trial_ending_time_point
            event_onsets.extend(current_event_onsets)
            tcourse.extend(map(lambda t : t["time_point"], current_event_onsets))
            event_durations.extend(map(lambda info: info["duration"], filter(lambda info: info["type"] == "event", tcourse_info_array)))
            onset_intensities.extend(map(lambda info: info["intensity"], filter(lambda info: info["type"] == "event", tcourse_info_array)))
            previous_trial_ending_time_point += next_time_point
            
        print((tcourse, event_durations, onset_intensities))
        stimuli_function = fmrisim.generate_stimfunction(onsets = tcourse,
                                                   event_durations = event_durations,
                                                   total_time = total_time,
                                                   weights = onset_intensities,
                                                   temporal_resolution = temporal_resolution,
                                                   )
        
        return (stimuli_function, tcourse, event_onsets, onset_intensities)
    
    def apply_transients(self, tcourse_groups, paradigm_configs, temporal_resolution, total_time, sub_imp = 0.8):
        
        (stimuli_function, tcourse, event_onsets, onset_intensities) = self.__fabricate_stimuli_function(tcourse_groups, temporal_resolution, total_time)
        
        print((stimuli_function, tcourse, event_onsets, onset_intensities))
        
        tr = temporal_resolution
        
        for paradigm_config_array in paradigm_configs:
            for paradigm_config in paradigm_config_array:
                l = paradigm_config["lsis"]
                u = paradigm_config["usis"]
                paradigm = paradigm_config["paradigm"]
                i = int(paradigm_config["inter_event_starting"] * tr)
                j = int(paradigm_config["inter_event_ending"] * tr)
                if stimuli_function[i] != 0:
                    i += 1
                if paradigm == 'attnmap':
                    if l <= 7 and u <= 9:
                        stimuli_function[ i : i + tr * 1 ] = sub_imp                    
                    elif ((l <= 7 and u >= 10 and u <= 13) or (l <= 5 and u >= 14)
                          or (l == 8 and u >= 8 and u <= 9)
                          or (l == 9 and u == 9)):
                        stimuli_function[ i : i + tr * 4 ] = sub_imp                    

                    elif (((l == 6 or l == 7) and u >= 14 and u <= 20) 
                          or ((l >=8 and l <= 10) and u >= 10 and u <= 13)
                          or ((l >= 10 and l <= 13) and u >= 11 and u <= 13)):
                        stimuli_function[ i + tr * int(l/2)  : i + tr * int(l/2) + tr * 1 ] = sub_imp                    


                    elif (((l == 8 or l == 9) and u >= 14 and u <= 20) 
                        or ((l == 10 or l == 11) and u >= 14 and u <= 20)
                        or ((l >= 12 and l <= 14) and u >= 14 and u <= 20)
                        or ((l == 14 or l == 15) and u >= 15 and u <= 20)
                        or (l >= 16 and u >= 15 and u <= 20)):
                        stimuli_function[ i + tr * int(l/2)  : i + tr * int(l/2) + tr * 4 ] = sub_imp

                elif paradigm == 'wmmap':
                    # idxA = np.where(etrain == 1)[0][:]
                    # idxB = np.where(etrain2 == 1)[0][:]
                    # minl = min(len(idxA),len(idxB))
                    # for i in range(minl):
                    #     if idxA[i] > idxB[i]:
                    #         idxB[i] = 0  #This is used to treat for any error 
                    #         # sampling, or when the two events are not alternate

                    if ((l <= 9 and u <= 13) or (l <= 5)):
                        stimuli_function[ i : j ] = sub_imp

                    else:
                        stimuli_function[ i + tr * int(l/2) : j ] = sub_imp 

                else:
                    raise ValueError('Invalid transient map arguement')
                    
        print((stimuli_function, tcourse, event_onsets, onset_intensities))
        return (stimuli_function, tcourse, event_onsets, onset_intensities)
        

    def produce_signal(self, loadvolume, stimuli_function, temporal_resolution, is_non_linear, signal_magnitude, noiseAdded, hrf_type = 'double_gamma', cutome_hrf_params = None):
        expanded_stimfunc = np.matlib.repmat(stimuli_function, 1, loadvolume.voxels)
        

        signal_func = fmrisim.convolve_hrf(stimfunction = expanded_stimfunc,
                                           tr_duration = self.loadvolume.tr,hrf_type = hrf_type,params = cutome_hrf_params,
                                           temporal_resolution = temporal_resolution,
                                           scale_function = 0, nonlin = is_non_linear)


        # Where in the brain are there stimulus evoked voxels
        # The np.where is traversing through the self.signal_volume across every coordinate and looking where the signal is to be 
        # evoked as set before
        signal_idxs = np.where(loadvolume.signal_volume == 1)

        # Pull out the voxels corresponding to the noise volume
        noise_func = loadvolume.noise[signal_idxs[0], signal_idxs[1], signal_idxs[2], :]

        # Compute the signal appropriate scaled
        signal_func_scaled = fmrisim.compute_signal_change(signal_function = signal_func,
                                                           noise_function = noise_func.T,
                                                           noise_dict = loadvolume.noise_dict,
                                                           magnitude = signal_magnitude,
                                                           method = 'PSC')
        
        signal = fmrisim.apply_signal(signal_func_scaled, loadvolume.signal_volume,)
        
        if noiseAdded:
            
            return signal + loadvolume.noise
        else:
            return signal
        
    
class expanalyse:
     
    def __init__(self, data, event_onsets, contrast, expdesign):
         
        self.expdesign = expdesign

        HRF = _dghrf()
        
        events = event_onsets
        self.boxcars = {}
        self.conv_boxcars = {}
        for event in events:
            if event["event_name"] in self.boxcars:
                continue
            
            self.boxcars[event["event_name"]] = np.zeros(np.size(data,3))
            current_event_onsets = list(filter(lambda onset: onset["event_name"] == event["event_name"], event_onsets))
            print(current_event_onsets)
            current_onsets = map(lambda onset: onset["time_point"], current_event_onsets)
            for index, onset in enumerate(current_onsets):
                print(index, onset)
                self.boxcars[event["event_name"]][(onset / self.expdesign.loadvolume.tr).astype('int')] = current_event_onsets[index]["intensity"]
            
            self.conv_boxcars[event["event_name"]] = np.convolve(self.boxcars[event["event_name"]], HRF, 'same')
        
        
#         self. boxcar_A = np.zeros(np.size(data,3))
#         self. boxcar_B = np.zeros(np.size(data,3))
        
#         self.boxcar_A[(self.expdesign.onsets_A/self.expdesign.loadvolume.tr).astype('int')] = 1
#         self.conv_boxcar_A = np.convolve(self.boxcar_A,HRF,'same')
       
#         self.boxcar_B[(self.expdesign.onsets_B/self.expdesign.loadvolume.tr).astype('int')] = 1
#         self.conv_boxcar_B = np.convolve(self.boxcar_B,HRF,'same')
        
        lb = (self.expdesign.loadvolume.coordinates - ((self.expdesign.loadvolume.feature_size - 1) / 2)).astype('int')[0]
        ub = (self.expdesign.loadvolume.coordinates + ((self.expdesign.loadvolume.feature_size - 1) / 2) + 1).astype('int')[0]
        
        roi_brain = data[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2], :]
        roi_brain = roi_brain.reshape((self.expdesign.loadvolume.voxels,data.shape[3]))
        self.roi = roi_brain[13,:].T
        
        conv_boxcars_list = list(self.conv_boxcars.values())
        boxcars_list = list(self.boxcars.values())
        self.X = np.zeros((conv_boxcars_list[0].shape[0],len(self.conv_boxcars)))
        self.design = np.empty((boxcars_list[0].shape[0],len(self.boxcars)))
        
        for index, conv_boxcar in enumerate(conv_boxcars_list):
            self.X[:,index] = conv_boxcar
            
        for index, boxcar in enumerate(boxcars_list):
            self.design[:,index] = boxcar
        
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
    
    

        
         
         

