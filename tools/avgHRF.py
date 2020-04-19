#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:56:46 2020

@author: sdas
"""

def avgHRF(onsets,brain):
    sum = 0
    res = 20
    for i in range(0,len(onsets)-1):
        start = (onsets[i] / tr).astype('int')
        series_A = np.linspace(start - 3,start + 16,res).astype('int')
        sum = sum + brain[series_A]
    mean_A = sum/(len(onsets)-1)
    return mean_A
# Taking mean of A trials from -4 onset to +16, looking at the averaged HRF
