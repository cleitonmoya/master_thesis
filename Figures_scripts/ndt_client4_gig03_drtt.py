# -*- coding: utf-8 -*-
"""
NDT Dataset - Changepoint Detection - Example 2
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9c99', 'dca6326b9ca8',
       'dca6326b9ce4', 'e45f01359a20', 'e45f01963bb8', 'e45f01963c21',
       'e45f01ad569d']

client_n = [f'Client {n}' for n in range(1,len(clients)+1)]
dict_client = {c:n+1 for n,c in enumerate(clients)}

series = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

methods = ['shewhart_ba', 'shewhart_ps', 'ewma_ba', 'ewma_ps',
           'cusum_2s_ba', 'cusum_2s_ps', 'cusum_wl_ba', 'cusum_wl_ps',
           'bocd_ba', 'bocd_ps', 'rrcf_ps', 'vwcd', 'pelt_np']

sites = ['gig01', 'gig02', 'gig03', 'gig04', 
         'gru02', 'gru03', 'gru05','rnp_rj', 'rnp_sp']

basic = ['shewhart_ba', 'ewma_ba', 'cusum_2s_ba', 'cusum_wl_ba', 'bocd_ba']
sequential_ps = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps']
others_ps = ['bocd_ps', 'rrcf_ps', 'vwcd']

units = {
    'd_throughput': 'Mbits/s',
    'u_throughput': 'Mbits/s',
    'd_rttmean': 'ms',
    'u_rttmean': 'ms'}


def method_name(m):
    if m[:8] == 'shewhart':
        return 'Shewhart'
    elif m[:4] == 'ewma':
        return 'EWMA'
    elif m[:8] == 'cusum_2s':
        return '2S-Cusum'
    elif m[:8] == 'cusum_wl':
        return 'WL-Cusum'
    elif m=='vwcd':
        return 'VWCD'
    elif m[:4]=='bocd':
        return 'BOCD'
    elif m=='rrcf_ps':
        return 'RRCF'
    elif m=='pelt_np':
        return 'Pelt-NP'


#file = typeA_rtt[0]
file = 'dca6326b9ca8_gig03_d_rttmean.txt'
client = file[:12]
if file[13]=='g':
    site = file[13:18]
    serie = file[19:-4]
else:
    site = file[13:19]
    serie = file[20:-4]


# Load the time-seies
y = np.loadtxt(f'../Dataset/ndt/{file}', usecols=1, delimiter=',')
print(f'Client {dict_client[client]}, {site}, {serie}')

# Load the dataframe with dtected change-points
df = pd.concat([pd.read_pickle(f'../Experiment/results_ndt/df_ndt_{m}.pkl') for m in methods], 
               ignore_index=True)

df['method_name'] = [method_name(m) for m in df['method']]
lw = 0.5 # linewidth to plot

# Formatters
markers_dict = {
    'Shewhart' :'x', 
    'EWMA': 's', 
    '2S-Cusum': '^', 
    'WL-Cusum': 'v',
    'VWCD': 'd',
    'BOCD': '>',
    'RRCF': '<' ,
    'Pelt-NP': '*'}

# Load the list of changepoints (not empty) for each method
df_ = df[(df.client==client) &
         (df.site==site) &
         (df.serie==serie) &
         (df.CP.str.len() != 0)]

# dict of methods chagepoint list
CP_dict = dict(df_[['method', 'CP']].to_dict(orient='split')['data'])

basic_list = df_[df_.method.isin(basic)].method.unique()
seq_ps_list = df_[df_.method.isin(sequential_ps)].method.unique()
others_ps_list = df_[df_.method.isin(others_ps)].method.unique()
methods_name_list = df_.method_name.unique()

fig = plt.figure(constrained_layout=True, figsize=(4.5,4))
ax = fig.subplot_mosaic([['legend'], [0], [1], [2]], sharex=True, sharey=True,
                          gridspec_kw={'height_ratios':[0.001, 1, 1,1 ]})


ax['legend'].axis('off')
ax[0].set_title('Basic methods (Classical and BOCD) and Pelt-NP')
ax[0].plot(y, linewidth=lw)
ax[0].grid(linestyle=':')
ax[0].set_ylabel(units[serie], fontsize=6)
ax[0].tick_params(axis='both', labelsize=6)


y0,y1 = ax[0].get_ylim()
#y2 = y0+(y1-y0)/2
y2 = 0.9*y1
y0 = y0+5
d = (y2-y0)/5

for i,m in enumerate(basic_list):
    CP = CP_dict[m]
    for cp in CP:
        ax[0].axvline(cp, color='r', alpha=0.5, linewidth=0.5)
        ax[0].plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')

if 'Pelt-NP' in methods_name_list:
    CP = np.array(CP_dict['pelt_np'])-1
    for cp in CP:
        ax[0].axvline(cp, color='g', alpha=0.5, linewidth=1)
        ax[0].plot(cp, y0+(i+1)*d, 
                   marker=markers_dict['Pelt-NP'], 
                   markersize=5, 
                   color='g')

        
ax[1].set_title('Classical methods (proposed)')
ax[1].plot(y, linewidth=lw)
ax[1].grid(linestyle=':')
ax[1].set_ylabel(units[serie], fontsize=6)
ax[1].tick_params(axis='both', labelsize=6)
y0 = y0+d
for i,m in enumerate(seq_ps_list):
    CP = CP_dict[m]
    for cp in CP:
        ax[1].axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
        ax[1].plot(cp, y0+i*d, 
                   marker=markers_dict[method_name(m)], 
                   markersize=3, 
                   color='r')

ax[2].set_title('BOCD, RRCF, VWCD (proposed)')
ax[2].plot(y, linewidth=lw)
ax[2].grid(linestyle=':')
ax[2].set_xlabel('sample (t)', fontsize=6)
ax[2].set_ylabel(units[serie], fontsize=6)
ax[2].tick_params(axis='both', labelsize=6)
y0 = y0+d
for i,m in enumerate(others_ps_list):
    CP = CP_dict[m]
    for cp in CP:
        
        if m != 'vwcd':
            ax[2].axvline(cp, color='r', linestyle='-', alpha=0.5, linewidth=0.5)
            ax[2].plot(cp, y0+i*d, 
                       marker=markers_dict[method_name(m)], 
                       markersize=3, 
                       color='r')
        else:
            ax[2].axvline(cp, color='tab:purple', linestyle='-', alpha=0.5, linewidth=0.5)
            ax[2].plot(cp, y0+i*d, 
                       marker=markers_dict[method_name(m)], 
                       markersize=3, 
                       color='tab:purple')


# draw the legend
lines_leg = [mlines.Line2D([], [], 
                           color='r', 
                           marker=markers_dict[m], 
                           linewidth=0, 
                           markersize=3, 
                           label=m) for m in methods_name_list
             if (m!='Pelt-NP' and m != 'VWCD')]

if 'VWCD' in methods_name_list:
    lines_leg = lines_leg + [mlines.Line2D([], [], 
                               color='tab:purple', 
                               marker=markers_dict['VWCD'], 
                               linewidth=0, 
                               markersize=3, 
                               label='VWCD')]


if 'Pelt-NP' in methods_name_list:
    lines_leg = lines_leg + [mlines.Line2D([], [], 
                               color='g', 
                               marker=markers_dict['Pelt-NP'], 
                               linewidth=0, 
                               markersize=4, 
                               label='Pelt-NP')]

_ = ax['legend'].legend(handles=lines_leg, 
                    loc='upper center',
                    ncol=4, 
                    fontsize=6,
                    handletextpad=0.01,
                    columnspacing=0.5)

xlim = (0,1100)
ylim = (0,300)
xran = np.arange(0,1200,100)
yran = np.arange(0,350,50)
for j in range(0,3):
    ax[j].xaxis.set_tick_params(labelbottom=True)
    ax[j].set_xticks(xran)
    ax[j].set_xlim(xlim)
    
    ax[j].set_yticks(yran)
    ax[j].set_yticklabels(yran)
    ax[j].set_ylim(ylim)
