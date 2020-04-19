#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:55:51 2020

@author: sdas
"""

import matplotlib.pyplot as plt
import numpy as np
def plotdata(data1,data2,max1,max2):
    fig= plt.figure(figsize = (5,10))
    ax = fig.add_subplot(211)
    im = ax.imshow(data1,vmin = 0, vmax = max1,cmap = 'jet')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI")
    ax.set_ylabel("Lower Bound of ISI")
    ax.set_title('Detection Power')
    fig.colorbar(im)
    labels = np.arange(1,len(data1)+1,2).astype(str)
    plt.xticks(np.arange(0,len(data1),2), labels)
    plt.yticks(np.arange(0,len(data1),2), labels)


    ax = fig.add_subplot(212)
    im2 = ax.imshow(data2,vmin = 0, vmax = max2,cmap = 'jet')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI")
    ax.set_ylabel("Lower Bound of ISI")
    ax.set_title('HRF Estimation Efficiency')
    fig.colorbar(im2)
    plt.xticks(np.arange(0,len(data2),2), labels)
    plt.yticks(np.arange(0,len(data2),2), labels)
    return fig

