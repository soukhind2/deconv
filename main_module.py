import design
import numpy as np
import loadvolume
import time

__all__ = [
    "test",
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
    d = design.expdesign(parameters["event_duration"], parameters["tevents"], parameters["signal_mag"], lv, parameters["distribution"], parameters["exp"], parameters["cue_ratio"], noise = parameters["noise"], nonlinear = parameters["nonlinear"], load = arg_map)
    # store = 1
    for lisi in np.arange(1, max_lisi + 1, 1):
        result[str(lisi)] = {}
        l = 0
        for uisi in np.arange(1, max_uisi + 1, 1):
            if lisi > uisi:
                l += 1
                continue
            if parameters["paradigm"]:
                arg_map = parameters["paradigm"] + 'map'
            else:
                arg_map = None

            #d = design.expdesign(lisi, uisi, parameters["event_duration"], parameters["tevents"], parameters["signal_mag"], lv, parameters["distribution"], parameters["exp"], parameters["cue_ratio"], noise = parameters["noise"], nonlinear = parameters["nonlinear"], load = arg_map)

            data = d.tcourse(lisi,uisi)
            e = design.expanalyse(data, np.array([1, 0]), expdesign = d)
            p1[k,l] = e.calc_Fd()
            p2[k,l] = e.calc_Fe(ncond =2)

            result[str(lisi)][str(uisi)] = {
                "e": e.roi,
                "t": e.design[:,0] + e.design[:,1],
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