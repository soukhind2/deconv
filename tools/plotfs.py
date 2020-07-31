#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:55:51 2020

@author: sdas
"""

import matplotlib.pyplot as plt
import numpy as np
def plotdata(data1,data2,max1 ,max2 ,title1 = "Detection Power",
             title2 = "Estimation Efficiency",normalize = False):
    if normalize:
        data1 = data1/np.amax(data1)
        data2 = data2/np.amax(data2)
        max1 = 1
        max2 = 1
    fig1= plt.figure(figsize = (5,5))
    ax = fig1.add_subplot(111)
    im = ax.imshow(data1,vmin = 0, vmax = max1,cmap = 'jet')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title1,size = 15)
    fig1.colorbar(im)
    labels = np.arange(1,len(data1)+1,2).astype(str)
    plt.xticks(np.arange(0,len(data1),2), labels)
    plt.yticks(np.arange(0,len(data1),2), labels)

    fig2= plt.figure(figsize = (5,5))

    ax = fig2.add_subplot(111)
    im2 = ax.imshow(data2,vmin = 0, vmax = max2,cmap = 'jet')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title2,size = 15)
    fig2.colorbar(im2)
    plt.xticks(np.arange(0,len(data2),2), labels)
    plt.yticks(np.arange(0,len(data2),2), labels)
    return fig1,fig2

def avgHRF(onsets,brain,tr):
    res = 20
    out = np.zeros((len(onsets),res))
    for i in range(0,len(onsets)):
        start = (onsets[i] / tr).astype('int')
        series_A = np.linspace(start - 3,start + 16,res).astype('int')
        if series_A[-1]  >= brain.shape[0]:
            break
        else:
            out[i,:] = brain[series_A]
        
    return out
