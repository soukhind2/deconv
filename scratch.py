#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 23:39:21 2020

@author: soukhind
"""
from tools.smooth import smooth
import matplotlib.pyplot as plt


fig = plt.figure(figsize = (15,10))
temp = np.diagonal(a1)
plt.plot(smooth(temp,9,window = 'hanning'),label = 'smooth')
plt.plot(np.diagonal(a1),label = 'raw') 
plt.xlim(-0.5,17)
plt.ylim(10,50)
plt.legend()
plt.title("Non linear effects",size = 20)
labels = np.arange(18).astype(str)
plt.xticks(np.arange(18), labels, size = 15)

