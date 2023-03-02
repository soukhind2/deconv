from importlib import reload
# design = reload(design)
from design import expdesign, expanalyse
import numpy as np
import loadvolume
import time
import functools



__all__ = [
    "run_experiment",
    "load_noise"
]

parameters = {
                "signal_mag": [2],
                # "exp" : 30,
                "null_ratio": 0,
                "noise" : False,
                "nonlinear" : True,
                "hrf_type": 'double_gamma',
                "cutome_hrf_params": None,
                "contrast": np.array([1, 0, 0, 0])
             }
def run_experiment(parameters, loadvolume, config_filename):
    
    with open(config_filename) as f:
        lines = [line.rstrip() for line in f]
    configs = [ parseExperimentConfigStr(line) for line in lines ]
    
    
    d = expdesign(loadvolume, signal_magnitude = parameters["signal_mag"], null_ratio = parameters["null_ratio"], noise = parameters["noise"], nonlinear = parameters["nonlinear"])
    
    total_time = int(loadvolume.dim[3] * loadvolume.tr)
    
    
    (tcourse, paradigm_configs) = d.generate_tcourse(configs, total_time = total_time, temporal_resolution = 10)
    
    (stimuli_function, tcourse, event_onsets, onset_intensities) = d.apply_transients(tcourse, paradigm_configs = paradigm_configs, temporal_resolution = 10, total_time = total_time, sub_imp = 0.8)
    
    signal = d.produce_signal(loadvolume = d.loadvolume, stimuli_function = stimuli_function, temporal_resolution = 10, is_non_linear = d.nonlinear, signal_magnitude = d.signal_magnitude, noiseAdded = d.noise, hrf_type = parameters["hrf_type"], cutome_hrf_params = parameters["cutome_hrf_params"])
    
    
    
#     e = expanalyse(signal, event_onsets = event_onsets , contrast = parameters['contrast'], expdesign = d)
#     p1 = e.calc_Fd()
#     p2 = e.calc_Fe(ncond =2)
    
    # result = {
    #             "e": e.roi,
    #             "t": np.sum(e.design, axis = 1),
    #         }
    
    return signal, event_onsets
    
    
def load_noise(noise_file_path):
    lv = loadvolume.loadvolume(noise_file_path)
    lv.loaddata()
    lv.loadmask()
    lv.generate_noise()
    lv.generate_region()
    return lv


def convertToExpermentConfig(configSubString):
    if ':' in configSubString:
        interEventsConfig = { "type": 'event', "name": configSubString.split(':')[0].strip(), "event-duration": int(configSubString.split(':')[1].strip()) / 1000, "intensity": float(configSubString.split(':')[2].strip()) if len(configSubString.split(':')) > 2 else None }
        if "intensity" in interEventsConfig and interEventsConfig["intensity"] is not None and (interEventsConfig["intensity"] <= 0 or interEventsConfig["intensity"] > 1):
            raise AttributeError("Intensity should be in the range of (0, 1]")
        return interEventsConfig
    elif '{' in configSubString:
        interEventsConfig = { "type": 'inter-events', "lsis": -1, "usis": -1, "distribution": None, "paradigm": None }
        processedString = configSubString[1 : -1]
        configStrings = list(map(lambda x: x.strip(), processedString.split(',')))
        
        for setting in configStrings:
            if '-' in setting:
                interEventsConfig['lsis'] = int(setting.split('-')[0].strip()) / 1000
                interEventsConfig['usis'] = int(setting.split('-')[1].strip()) / 1000
            elif setting.isnumeric():
                interEventsConfig['lsis'] = int(setting) / 1000
                interEventsConfig['usis'] = int(setting) / 1000
            elif setting in [ 'exp', 'uniform', 'stochastic_rapid', 'stochastic_interm', 'stochastic_slow' ]:
                interEventsConfig['distribution'] = setting
            elif setting in [ 'wmmap', 'attnmap' ]:
                interEventsConfig['paradigm'] = setting
            else:
                raise AttributeError(f"Unkown parameter: {setting}")
    
        return interEventsConfig
    

def parseExperimentConfigStr(configString):
    elements = functools.reduce(lambda a, b: a + b, list(map(lambda s: s.split('['), configString.split(']'))))
    return list(map(convertToExpermentConfig, list(filter(lambda e: e, elements))))