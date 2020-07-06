#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:03:23 2020

@author: soukhind
"""
# Cannot run without test.py and some minor edits in design


# For checking individual activations
import numpy as np
import matplotlib.pyplot as plt
from tools._dghrf import _dghrf as df

stim = np.array([0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0])
hrf = np.array(df())

def linear(f,g):
    f = np.concatenate((f,np.zeros(len(g))))
    g = np.concatenate((g,np.zeros(len(g))))
    r = np.zeros(len(f))
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p] * g[k-p])
    return r

def nonlinear(f,g):
    f = np.concatenate((f,np.zeros(len(g))))
    g = np.concatenate((g,np.zeros(len(g))))
    r = np.zeros(len(f))
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p]*f[p] * g[k-p]*g[k-p])
    return r

def nonlinear2(f,g):
    f = np.concatenate((f,np.zeros(len(g))))
    g = np.concatenate((g,np.zeros(len(g))))
    ff = np.zeros((len(f),len(f)))
    for i in range(len(f)):
        for j in range(len(f)):
            ff[i,j] = f[i]*f[j]
            
          
    r = np.zeros(len(f))
    for k in range(len(f)):
        for i in range(len(f)):
            for j in range(len(f)):
                r[k] = r[k] + ff[i,j]*g[k-i]*g[k-j]
    return r

out = linear(stim,hrf)
out2 = nonlinear(stim,hrf)
out3 = nonlinear2(stim,hrf)
base = np.convolve(stim,hrf)

fig = plt.figure()
ax = fig.add_subplot(511)
plt.plot(base/np.max(base))
plt.plot(stim)
ax.set_title("Base")


ax = fig.add_subplot(512)
plt.plot(out/np.max(out))
ax.set_title("Linear")

ax = fig.add_subplot(513)
plt.plot(out2/np.max(out2))
ax.set_title("Non Linear")

ax = fig.add_subplot(514)
plt.plot(out3/np.max(out3))
ax.set_title("Non Linear New")

ax = fig.add_subplot(515)
tot = out + out3/2
plt.plot(tot/np.max(tot))
ax.set_title("Non Linear Add")

#%%
temp = nonlinear2(hrf,stim)
plt.imshow(temp)