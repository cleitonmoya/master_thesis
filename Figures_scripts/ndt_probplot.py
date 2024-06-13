# -*- coding: utf-8 -*-
"""
Create probplots for some NDT time-series with a change-point
Change-point identified with Pelt-NP
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot

plt.rcParams.update({'font.size': 6, 'axes.titlesize': 6, 'figure.titlesize':'large'})
rng = np.random.default_rng(seed=0)

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9c99', 'dca6326b9ca8',
       'dca6326b9ce4', 'e45f01359a20', 'e45f01963bb8', 'e45f01963c21',
       'e45f01ad569d']

dict_client = {c:n+1 for n,c in enumerate(clients)}

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

dict_series = {'d_throughput': 'download throughput', 
                    'd_rttmean': 'download rtt ', 
                    'u_throughput': 'upload throughput', 
                    'u_rttmean': 'upload RTT'}


# Read the dataset with the Pelt-NP change-points
df = pd.read_pickle('../Experiment/results_ndt/df_ndt_pelt_np.pkl')

# include the filename column
def file_name(r):
    file = f'{r.client}_{r.site}_{r.serie}.txt'
    return file

df['file'] = df.apply(file_name, axis=1)

# select only series with one ore zero change-points
df0 = df[df.num_cp == 0]
df1 = df[df.num_cp == 1]

# choose N series
N = 4
files = rng.choice(df1.file.values, size=N, replace=False)

# plot the series and change-point
fig = plt.figure(constrained_layout=True, figsize=(4.5,5))
subfigs = fig.subfigures(N, 1)

for i,f in enumerate(files):
    
    ax = subfigs[i].subplot_mosaic([[0,1,2]], gridspec_kw={'width_ratios':[1,0.5,0.5]}, sharey=True)
    
    y = np.loadtxt(f'../Dataset/ndt/{f}', usecols=1, delimiter=',')
    cp = df[df.file==f].CP.values[0][0]
    
    client = f[:12]
    if f[16] == '_':
        site = f[13:19]
        serie_type = f[20:-4]
    else:
        site = f[13:18]
        serie_type = f[19:-4]
    client_name = dict_client[client]
    serie_name = dict_series[serie_type]
    
    subfigs[i].suptitle(f'Client {client_name}, {site}, {serie_name}')
    ax[0].plot(y, linewidth=0.5)
    ax[0].axvline(cp, color='red', alpha=1, linewidth=1, label='Pelt-NP')
    
    y1 = y[:cp]
    _ = probplot(y1, plot=ax[1])
    ax[1].set_title('')
    ax[1].get_lines()[0].set(markersize=2, color='C0', alpha=0.5)
    ax[1].get_lines()[1].set(linewidth=0.5, alpha=1)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')
    ax[1].set_xticks([-2,-1,0,1,2])
    
    y2 = y[cp:]
    _ = probplot(y2, plot=ax[2])
    ax[2].set_title('')
    ax[2].get_lines()[0].set(markersize=2, color='C0', alpha=0.5)
    ax[2].get_lines()[1].set(linewidth=0.5, alpha=1)
    ax[2].set_xlabel('')
    ax[2].set_ylabel('')
    ax[2].set_xticks([-2,-1,0,1,2])

subfigs[0].get_axes()[0].set_xlabel('sample (t)')
subfigs[0].get_axes()[1].set_title('Segment 1')
_ = subfigs[0].get_axes()[1].set_xlabel('theoretical quantiles')
subfigs[0].get_axes()[2].set_title('Segment 2')
_ = subfigs[0].get_axes()[2].set_xlabel('theoretical quantiles')