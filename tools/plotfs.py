#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:55:51 2020

@author: sdas
"""

import matplotlib.pyplot as plt
import numpy as np
def plotdata(data1,data2,max1 ,max2 ,title1 = "Detection Power",
             title2 = "Estimation Efficiency"):
# def plotdata(data1,data2,normalize = False,title1 = "Detection Power",
#              title2 = "Estimation Efficiency"):
#     if normalize:
#         data1 = data1/np.amax(data1)
#         data2 = data2/np.amax(data2)
#         max1 = 1
#         max2 = 1
#     else:
#         max1 = np.amax(data1)
#         max2 = np.amax(data2)
    fig1= plt.figure(figsize = (5,5))
    ax = fig1.add_subplot(111)
    im = ax.imshow(data1,vmin = 0, vmax = max1,cmap = 'viridis')
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
    im2 = ax.imshow(data2, vmin = 0, vmax = max2, cmap = 'viridis')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title2,size = 15)
    fig2.colorbar(im2)
    plt.xticks(np.arange(0,len(data2),2), labels)
    plt.yticks(np.arange(0,len(data2),2), labels)
    return fig1,fig2

def plot_result(data1, data2, weights = [ 0.5, 0.5 ], title = "Weighted Optimality"):
    data1 = data1 / np.amax(data1)
    data2 = data2 / np.amax(data2)
    
    data = weights[0] * data1 + weights[1] * data2
    
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, vmin = 0, vmax = 1, cmap = 'viridis')
    ax.invert_yaxis()
    ax.set_xlabel("Upper Bound of ISI",size = 12)
    ax.set_ylabel("Lower Bound of ISI",size = 12)
    ax.set_title(title,size = 15)
    fig.colorbar(im)
    labels = np.arange(1,len(data)+1,2).astype(str)
    plt.xticks(np.arange(0,len(data),2), labels)
    plt.yticks(np.arange(0,len(data),2), labels)

    return fig

def avg_response(onsets, brain, before, after):
    onset_indices = []
    for i in range(0, len(onsets)):
        if onsets[i] == 1:
            onset_indices = [ *onset_indices, i ]
    res = after + before + 1
    out = np.zeros((len(onset_indices), res))
    for i in range(0,len(onset_indices)):
        start = onset_indices[i]
        series = np.linspace(start - before, start + after, res).astype('int')
        if series[-1] >= brain.shape[0]:
            break
        else:
            out[i,:] = brain[series]
        
    out = np.mean(out, 0)
    return out

def plotjitterdist(w):
    
    fig1 = plt.figure(figsize = (5,5))
    ax = plt.plot(w)
    plt.yticks([0,max(w)],['0','1'])
    plt.ylim([0,max(w)*1.4])
    plt.xticks([0,len(w)],['Lower ISI','Upper ISI'])
    plt.xlabel('ISI')
    plt.ylabel('Probability Distribution')

def plot_avg_response(onsets, brain, before = 2, after = 10):
    out = avg_response(onsets,brain, before,after)
    out = out / np.max(out)
    plt.figure(figsize = (3,3))
    plt.plot(out)
    ooo = np.arange(-before,after+1,2)
    oo = np.arange(-before,after+1,1)
    plt.xticks(np.arange(0,len(out),2),labels = ooo.astype(str),fontsize = 12)
    plt.yticks([0,0.5,1],'')
    plt.ylim(-0.15,1.2)
    plt.xlim(0,len(out))
    plt.axhline(0, color='red',lw = 1)
    plt.axvline(np.where(oo == 0) , color='red',lw = 1)
    plt.title('Average evoked response',fontsize = 15)
    plt.xlabel('TR',fontsize = 15)
    plt.ylabel('Amplitude',fontsize = 15)


def remove_transient(x):
    for i in range(0,len(x)):
        if x[i] == 0.66:
            x[i] = 0
    return x

def remove_irrelevant(x, excluded_stimuli = 'A' ):
    order = 0;
    excluded_order = 0 if excluded_stimuli == 'A' else 1
    for i in range(0, len(x)):
        if x[i] == 1:
            if order == excluded_order:
                x[i] = 0
            order = order + 1
            order = order % 2
    return x

def graph_timecourses(result, stimuli_onsets = [ { "time_point": 0, "event_name": 'A' } ], xlim = 50, ylim = 1.3):
    e_t_array = [];
    
    e_t = copy.deepcopy(result);

    t = remove_transient(e_t["t"])
    
    e_t_array.append({
        "e": e_t["e"] / np.max(e_t["e"]),
        "t": t, 
    })
    
    fig = plt.figure(figsize = (20,10))
    
    num_figures = 1
    
    
    mainax = fig.add_subplot(111)
    mainax.spines['top'].set_color('none')
    mainax.spines['bottom'].set_color('none')
    mainax.spines['left'].set_color('none')
    mainax.spines['right'].set_color('none')
    mainax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    
    cmap = plt.cm.get_cmap('hsv', num_figures + 10)
    for i in range(0, num_figures):
        ax = fig.add_subplot(int(num_figures), 1, i + 1)
        e_t = e_t_array[i]
        plt.plot(e_t["e"], color = cmap(i))
        plt.xlim(0, xlim)
        plt.ylim(-0.1, ylim + 0.2)
        (markers, stemlines, baseline) = plt.stem(e_t["t"])
        plt.setp(stemlines, linestyle="-", color="black", linewidth=1 )
        plt.setp(markers, color="black")

        plt.xticks(np.linspace(0,xlim,11),fontsize = 12)
        plt.yticks([0,0.5,1],labels = [],fontsize = 12)

        #ax.set_title('Profile 5',fontsize = 15)
        f = 1
        event_index = 0
        for i in range(0,xlim):
            if e_t["t"][i] > 0:
                marker = stimuli_onsets[event_index]['event_name']
                ax.text(i - 0.1, 1.1, marker, fontsize = 12)
                event_index += 1;
                
    
    mainax.set_xlabel('TR',fontsize = 18)
    mainax.set_ylabel('Amplitude', fontsize = 18)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    #fig.savefig("timecourses",dpi = 600,bbox_inches = 'tight',pad_inches = 0.1)
    
    return fig

    