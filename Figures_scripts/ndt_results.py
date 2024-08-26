# -*- coding: utf-8 -*-
"""
NDT Dataset - Results 
- Boxplot for the number of change-points and elapsed-time
- QoS application example - Download quality worsening
- 
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']
series_name = ['Download throughput', 'Download RTT ', 'Upload throughput', 'Upload RTT']

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9c99', 'dca6326b9ca8',
       'dca6326b9ce4', 'e45f01359a20', 'e45f01963bb8', 'e45f01963c21',
       'e45f01ad569d']

dict_client = {c:n+1 for n,c in enumerate(clients)}

sites = ['gig01', 'gig02', 'gig03', 'gig04', 
         'gru02', 'gru03', 'gru05','rnp_rj', 'rnp_sp']

methods_name = ['Shewhart', 'EWMA', '2S-CUSUM', 'WL-CUSUM', 'VWCD', 'Pelt-NP']
sequential_ps = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps']
methods_param = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps', 'vwcd', 'bocd_ps']
hatches = ['', '////']
methods = ['shewhart_ba', 'shewhart_ps', 'ewma_ba', 'ewma_ps',
           'cusum_2s_ba', 'cusum_2s_ps', 'cusum_wl_ba', 'cusum_wl_ps',
           'bocd_ba', 'bocd_ps', 'rrcf_ps', 'vwcd', 'pelt_np']

# Read the dataset
df = pd.concat([pd.read_pickle(f'../Experiment/results_ndt/df_ndt_{m}.pkl') for m in methods], 
               ignore_index=True)

def dev_abs(M, dev_abs, direction):
    M = np.array(M)
    n_mean = 0
    if len(M) > 1:
        dM = np.diff(M)
        
        if direction == 'dec':
            c2 = dM <= -dev_abs
        else:
            c2 = dM >= dev_abs
        n_mean = c2.sum()
    return n_mean

def method_type(m):
    if m[-2:] == 'ba':
        return 'Basic'
    elif m=='pelt_np':
        return 'Reference (off-line)'
    else:
        return 'Proposed'

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


def method_order(m):
    if m[:8] == 'shewhart':
        return 0
    elif m[:4] == 'ewma':
        return 1
    elif m[:8] == 'cusum_2s':
        return 2
    elif m[:8] == 'cusum_wl':
        return 3
    elif m[:4]=='bocd':
        return 4
    elif m=='rrcf_ps':
        return 5
    elif m=='vwcd':
        return 6
    elif m=='pelt_np':
        return 7


flierprops = dict( markersize=2)
hatches = ['', '///', '-', 'x', '\\', '*', 'o', 'O', '.']
C0 = np.array([142, 186, 217])/255 # blue
C1 = np.array([255, 190, 134])/255 # orange
C2 = np.array([149, 207, 149])/255 # green


def adjust_box(ax: plt.Axes):
    patches = ax.patches
    patches[0].set_y(-1/3)
    patches[5].set_y(0)
    patches[1].set_y(1-1/3)
    patches[6].set_y(1)
    patches[2].set_y(2-1/3)
    patches[7].set_y(2)
    patches[3].set_y(3-1/3)
    patches[8].set_y(3)
    patches[4].set_y(4-1/3)
    patches[9].set_y(4)
    patches[12].set_y(7-1/6)
    
# Number of change-points
fig = plt.figure(constrained_layout=True, figsize=(4,3))
ax = fig.subplot_mosaic([['legend', 'legend'],[0,1]], 
                          gridspec_kw={'height_ratios':[0.001, 1]})
ax['legend'].axis('off')

df_ = df.groupby(['method'], as_index=False, )['num_cp'].sum()
df_['method_type'] = [method_type(m) for m in df_['method']]
df_['method_name'] = [method_name(m) for m in df_['method']]
df_['method_order'] = [method_order(m) for m in df_['method']]
df_.sort_values(by='method_order', inplace=True)

_ = sns.barplot(data=df_, y='method_name', x='num_cp', hue='method_type',
                errorbar=None, palette=[C0, C1, C2], saturation=1, 
                ax=ax[0], zorder=2, width=1, hue_order=['Basic', 'Proposed' ,'Reference (off-line)'])

ax[0].set_xlabel("Number of changepoints")
ax[0].set_ylabel("")
ax[0].tick_params(axis='both', labelsize=8)
handles, labels = ax[0].get_legend_handles_labels()
ax[0].grid(axis='x', zorder=1, linestyle=':')
ax[0].set_xticks(np.arange(0,3500,500))
ax[0].set_xlim([0,3000])
ax[0].set_xticklabels(['0', '', '1000', '', '2000', '', '3000'])


handles, labels = ax[0].get_legend_handles_labels()
ax[0].get_legend().remove()
adjust_box(ax[0])


df_ = df.groupby(['method'], as_index=False)['elapsed_time'].sum()
df_['method_name'] = [method_name(m) for m in df_['method']]
df_['method_type'] = [method_type(m) for m in df_['method']]
df_['order'] = [method_order(m) for m in df_['method']]
df_.sort_values('order', inplace=True)

_ = sns.barplot(data=df_, y='method_name', x='elapsed_time', hue='method_type',
                errorbar=None, saturation=1, ax=ax[1], zorder=2, width=1,
                palette=[C0,C1,C2], legend=False)

ax[1].set_xlabel("Elapsed time (s)")
ax[1].set_xscale('log')
ax[1].set_ylabel("")
ax[1].tick_params(axis='x', labelsize=8)
ax[1].set_yticklabels('')
ax[1].grid(axis='x', zorder=1, linestyle=':')
ax[1].set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
adjust_box(ax[1])

_ = ax['legend'].legend(handles, labels, loc="upper center", 
                    ncol=3, fontsize=8, frameon=True, handletextpad=0.1,
                    columnspacing=1)


# Download quality worsening in the mean per client and serie - function of delta
markers = [',', 'o', 'x', 'v', 's', 'D']
methods_name = ['Shewhart', 'EWMA', '2S-CUSUM', 'WL-CUSUM', 'VWCD', 'BOCD']
k=0

fig = plt.figure(constrained_layout=True, figsize=(5,5))
ax = fig.subplot_mosaic([['legend', 'legend', 'legend'],[0,1,2], [3,4,5], [6,7,8]], 
                          gridspec_kw={'height_ratios':[0.001, 1, 1, 1]})

ax['legend'].axis('off')
xlim = 200
ylim = 14
x_thr = np.arange(1, xlim+1, 1)
for i,c in enumerate(clients):
    ax[i].set_title(f'Client {dict_client[c]}', fontsize=8)
    ax[i].grid(linestyle=':')
    
    for j,m in enumerate(methods_param):
        
        df_ = df[(df.client==c) & (df.serie==series_type[k]) & (df.method==m)]
        
        M0 = df_.M0.values.tolist()
        M = [sum([dev_abs(m0_list, p, 'dec') 
              for m0_list in M0]) for p in x_thr]
        ax[i].plot(x_thr, M, label=f'{methods_name[j]}')
        ax[i].set_yticks(range(0,16,2))
        ax[i].set_yticklabels(range(0,16,2))
        ax[i].set_xticks(range(0,250,50))
        ax[i].set_xticklabels(range(0,250,50))
        ax[i].set_ylim([0,ylim])
        ax[i].set_xlim([0,xlim])
        ax[i].tick_params(axis='both', which='major', labelsize=8)   
        
        #if i==0:
        ax[i].set_ylabel('Num. of changepoints', fontsize=8)
        ax[i].set_xlabel('Decrement (Mbits/s)', fontsize=8)
            
handles, labels = ax[0].get_legend_handles_labels()
_ = ax['legend'].legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
