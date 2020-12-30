#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 01:03:23 2020
This is a scratch module. It contains different scratch scenarios.

---------------------------------
Following is to model different codes of non linear interaction between two 
stimuli very close to each other in a event sequence.
Primarily, 2nd order voltera kernel series has been used to model the non linearities.
H2 or the 2nd degree kernel is set to hrf*hrf as mentioned in Friston 1998
@author: soukhind
"""


# For checking individual activations
import numpy as np
import matplotlib.pyplot as plt
from tools._dghrf import _dghrf as df

stim = [1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1] * 2
hrf = np.array(df())


def linear(f,g):
    f = np.concatenate((f,np.zeros(len(g))))  # make sure that f and g are of equal length
    g = np.concatenate((g,np.zeros(len(f))))
    r = np.zeros(len(f))
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p] * g[k-p])
    return r


def nonlinear(f,g):
    f = np.concatenate((f,np.zeros(len(g))))
    g = np.concatenate((g,np.zeros(len(f))))
    r = np.zeros(len(f))
    for k in range(len(f)):
        for p in range(len(f)):
            r[k] = r[k] + (f[p]*f[p] * g[k-p]*g[k-p])
    return r

def nonlinear2(f,g):
    
    f = np.concatenate((f,np.zeros(len(g))))
    g = np.concatenate((g,np.zeros(len(f))))
    
    ff = np.zeros((len(f),len(f)))
    for i in range(len(f)):
        for j in range(len(f)):
            ff[i,j] = f[i]*f[j]
            
    
    r = np.zeros(len(f))
    for k in range(len(f)):
        for i in range(len(f)):
            for j in range(len(f)):
                r[k] = r[k] + f[i]*f[j]*g[k-i]*g[k-j]
     
    return r
    

out = linear(stim,hrf)
out2 = nonlinear(stim,hrf)
out3 = nonlinear2(stim,hrf)
base = np.convolve(stim,hrf)

fig = plt.figure()
# ax = fig.add_subplot(511)
# plt.plot(base/np.max(base))
# plt.plot(stim)
# ax.set_title("Base")


ax = fig.add_subplot(411)
plt.plot(stim)
ax.set_title("Stim Function")

ax = fig.add_subplot(412)
plt.plot(out)
ax.set_title("Linear")

# ax = fig.add_subplot(513)
# plt.plot(out2)
# ax.set_title("Non Linear")

ax = fig.add_subplot(413)
plt.plot(out3)
ax.set_title("Non Linear")

ax = fig.add_subplot(414)
tot = out + out2/2  
tot2 = out + 0.1*out3/2
#plt.plot(tot)
plt.plot(tot2)
#plt.plot(base)
ax.set_title("Non Linear Add")


def stim_convolve(a,b):
    '''
        
        Parameters
        ----------
        a = Stimulus function
        b = HRF response to a impulse
        len(a)>len(b)
        Returns: 
        The time course of the HRF convolved with the Stimulus Function
        -------
        None.
        '''
    lin = linear(a,b)
    nonlin = nonlinear(a,b)
    return lin + nonlin/2


#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 21:22:57 2020
Created to make the WM transient activity
@author: soukhind
"""
runcell(1, '/Users/soukhind/deconv/deconv/test.py')
x = d.temp[:,1]
y = d.temp2[:,1]
#%%
                
#%%
plt.figure()
plt.plot(x)  
plt.plot(y)
#%%
idxA = np.where(x == 1)[0][:]
idxB = np.where(y == 1)[0][:]
l = min(len(idxA),len(idxB))
for i in range(l):
    if idxA[i] > idxB[i]:
        idxB[i] = 0

for i in range(l):
    if idxB[i]!= 0:
        y[ idxA[i] + 1 : idxB[i] ] = 0.66
        
            