# -*- coding: utf-8 -*-
"""
NDT Dataset - Client 8 timeseries
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'xtick.labelsize':6, 'ytick.labelsize':6})

c = 'e45f01963c21'
st = 'd_throughput'
sites = ['gig01', 'gig02', 'gig03', 'gig04', 'gru02','gru03', 'gru05', 'rnp_rj', 'rnp_sp']


fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,4), layout='constrained')
for i,s in enumerate(sites):
    file = c + "_" + s + "_" + st + ".txt"
    y = np.loadtxt(f'../Dataset/ndt/{file}', 
                   usecols=1, delimiter=',')
    ax[i//3, i%3].set_title(s)
    ax[i//3, i%3].plot(y, linewidth=0.5)
    ax[i//3, i%3].set_yticks(np.arange(50,650,100))
    ax[i//3, i%3].set_ylim(50,550)
    
    ax[i//3, i%3].set_ylabel('Mbits/s', fontsize=6)
    ax[i//3, i%3].set_xlabel('sample (t)', fontsize=6, loc='center')
#ax[0,0].set_ylabel('Mbits/s', fontsize=6)
#ax[0,0].set_xlabel('sample (t)', fontsize=6, loc='center')
ax[0,1].set_xticks([0,500,1000])
ax[0,2].set_xticks([0,500,1000])


st = 'd_rttmean'
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,4), layout='constrained')
for i,s in enumerate(sites):
    file = c + "_" + s + "_" + st + ".txt"
    y = np.loadtxt(f'../Dataset/ndt/{file}', 
                   usecols=1, delimiter=',')
    ax[i//3, i%3].set_title(s)
    ax[i//3, i%3].plot(y, linewidth=0.5)
    
    ax[i//3, i%3].set_ylabel('ms', fontsize=6)
    ax[i//3, i%3].set_xlabel('sample (t)', fontsize=6, loc='center')

#ax[0,0].set_ylabel('ms', fontsize=6)
#ax[0,0].set_xlabel('sample (t)', fontsize=6, loc='center')

ax[0, 0].set_yticks(np.arange(15,30,5))
ax[0, 1].set_yticks(np.arange(0,15,5))
ax[0, 2].set_yticks(np.arange(0,150,50))
ax[1, 0].set_yticks(np.arange(0,10,2))
ax[1, 1].set_yticks(np.arange(0,200,50))
ax[2, 0].set_yticks(np.arange(10,25,5))
ax[2, 1].set_yticks(np.arange(0,150,50))
ax[2, 2].set_yticks(np.arange(10,60,10))
ax[0,1].set_xticks([0,500,1000])
ax[0,2].set_xticks([0,500,1000])

'''
st = 'u_throughput'
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,4), layout='constrained')
for i,s in enumerate(sites):
    file = c + "_" + s + "_" + st + ".txt"
    y = np.loadtxt(f'../Dataset/ndt/{file}', 
                   usecols=1, delimiter=',')
    ax[i//3, i%3].set_title(s)
    ax[i//3, i%3].plot(y, linewidth=0.5)  
    ax[i//3, i%3].set_yticks(np.arange(50,400,100))
    ax[i//3, i%3].set_ylim(top=350)

ax[0,0].set_ylabel('Mbits/s', fontsize=6)
ax[0,0].set_xlabel('sample (t)', fontsize=6, loc='center')
ax[0,1].set_xticks([0,500,1000])
ax[0,2].set_xticks([0,500,1000])


st = 'u_rttmean'
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(5,4), layout='constrained')
for i,s in enumerate(sites):
    file = c + "_" + s + "_" + st + ".txt"
    y = np.loadtxt(f'../Dataset/ndt/{file}', 
                   usecols=1, delimiter=',')
    ax[i//3, i%3].set_title(s)
    ax[i//3, i%3].plot(y, linewidth=0.5)


ax[0,0].set_ylabel('ms', fontsize=6)
ax[0,0].set_xlabel('sample (t)', fontsize=6, loc='center')

ax[0, 0].set_yticks(np.arange(15,30,5))
ax[0, 1].set_yticks(np.arange(0,15,5))
ax[0, 2].set_yticks(np.arange(0,150,50))
ax[1, 0].set_yticks(np.arange(0,10,2))
ax[1, 1].set_yticks(np.arange(0,200,50))
ax[2, 0].set_yticks(np.arange(10,25,5))
ax[2, 1].set_yticks(np.arange(0,150,50))
ax[2, 2].set_yticks(np.arange(10,60,10))
ax[0,1].set_xticks([0,500,1000])
ax[0,2].set_xticks([0,500,1000])
'''