mimport design
import numpy as np
import loadvolume
import time
import functools

__all__ = [
    "run_experiment",
    "load_noise"
]

parameters = {
                "event_duration" : 0.1,
                "tevents" : 100,
                "signal_mag": [2],
                "distribution" : "uniform",
                "exp" : 30,
                "cue_ratio" : 1,
                "noise" : False,
                "nonlinear" : True,
                "paradigm" : None,
             }
def run_experiment(max_lisi, max_uisi, lv, parameters):
    p1 = np.zeros((max_lisi, max_uisi))
    p2 = np.zeros((max_lisi, max_uisi))
    result = {}

    k = 0
    start = time.time()
    
    if parameters["paradigm"]:
        arg_map = parameters["paradigm"] + 'map'
    else:
        arg_map = None
    
    d = design.expdesign(parameters["event_duration"], parameters["tevents"], parameters["signal_mag"], lv, parameters["distribution"], parameters["exp"], parameters["cue_ratio"], noise = parameters["noise"], nonlinear = parameters["nonlinear"], load = arg_map)
    # store = 1
    for lisi in np.arange(1, max_lisi + 1, 1):
        result[str(lisi)] = {}
        l = 0
        for uisi in np.arange(1, max_uisi + 1, 1):
            if lisi > uisi:
                l += 1
                continue
            
            #d = design.expdesign(lisi, uisi, parameters["event_duration"], parameters["tevents"], parameters["signal_mag"], lv, parameters["distribution"], parameters["exp"], parameters["cue_ratio"], noise = parameters["noise"], nonlinear = parameters["nonlinear"], load = arg_map)

            data = d.tcourse(lisi,uisi)
            e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
            p1[k,l] = e.calc_Fd()
            p2[k,l] = e.calc_Fe(ncond =2)

            result[str(lisi)][str(uisi)] = {
                "e": e.roi,
                "t": np.sum(e.design, axis = 1),
                "t1": e.design[:,0],
                "t2": e.design[:,1]
            }

            l += 1
        k += 1
    print(f'Time: {time.time() - start}')
    return p1, p2, result
    
    
def load_noise(noise_file_path):
    lv = loadvolume.loadvolume(noise_file_path)
    lv.loaddata()
    lv.loadmask()
    lv.generate_noise()
    lv.generate_region()
    return lv


def convertToExpermentConfig(configSubString):
    if '.' in configSubString:
        return { "type": 'event', "name": configSubString.split('.')[0], "event-duration": int(configSubString.split('.')[1]) }
    elif '{' in configSubString:
        interEventsConfig = { "type": 'inter-events', "lsis": -1, "usis": -1, "distribution": None, "paradigm": None }
        processedString = configSubString[1 : -1]
        configStrings = list(map(lambda x: x.strip(), processedString.split(',')))
        
        for setting in configStrings:
            if '-' in setting:
                interEventsConfig['lsis'] = int(setting.split('-')[0])
                interEventsConfig['usis'] = int(setting.split('-')[1])
            elif setting.isnumeric():
                interEventsConfig['lsis'] = int(setting)
                interEventsConfig['usis'] = int(setting)
            elif setting in [ 'exp', 'uniform', 'stochastic_rapid', 'stochastic_interm', 'stochastic_slow' ]:
                interEventsConfig['distribution'] = setting
            elif setting in [ 'wm', 'attn' ]:
                interEventsConfig['distribution'] = setting
    
        return interEventsConfig
    

def parseExperimentConfigStr(configString):
    elements = functools.reduce(lambda a, b: a + b, list(map(lambda s: s.split('['), expConfigStr.split(']'))))
    return list(map(convertToExpermentConfig, list(filter(lambda e: e, elements))))