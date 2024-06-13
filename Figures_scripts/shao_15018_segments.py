# -*- coding: utf-8 -*-
"""
Shao timeseries 15018 - Segments 2,5
@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})


basic = ['shewhart_ba', 'ewma_ba', 'cusum_2s_ba', 'cusum_wl_ba', 'bocd_ba']
sequential_ps = ['shewhart_ps', 'ewma_ps', 'cusum_2s_ps', 'cusum_wl_ps']
others_ps = ['bocd_ps', 'rrcf_ps', 'vwcd']
methods_ps = sequential_ps + others_ps + ['pelt_np']

methods = ['shewhart_ba', 'shewhart_ps', 'ewma_ba', 'ewma_ps',
           'cusum_2s_ba', 'cusum_2s_ps', 'cusum_wl_ba', 'cusum_wl_ps',
           'bocd_ba', 'bocd_ps', 'rrcf_ps', 'vwcd', 'pelt_np']

df = pd.concat([pd.read_pickle(f'../Experiment/results_shao/df_shao_{m}.pkl') for m in methods], 
               ignore_index=True)

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

Tmax = 1000
serie = '15018'
y = np.loadtxt(f'../Dataset/shao/{serie}.csv', 
               skiprows=1, usecols=1, delimiter=';')

CP_label = np.loadtxt(f'../Dataset/shao/{serie}.txt')


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


df_ = df[df.serie==f'{serie}.'].copy()

# Segments
t2i = int(CP_label[0])
t2f = int(CP_label[1])-1
t2 = np.arange(t2i,t2f)
y2 = y[t2i:t2f]

t5i = int(CP_label[3])
t5f = int(CP_label[4])
t5 = np.arange(t5i,t5f)
y5 = y[t5i:t5f]

# dict of methods chagepoint list
CP_dict = dict(df_[['method', 'CP_pred']].to_dict(orient='split')['data'])

CP_dict2 = {}
for m,cp_list in CP_dict.items():
    
    CP_list2 = [cp for cp in cp_list if cp>t2i and cp<t2f]
    if len(CP_list2) > 0 and m in methods_ps:
        CP_dict2[m] = CP_list2

CP_dict5 = {}
for m,cp_list in CP_dict.items():
    CP_list5 = [cp for cp in cp_list if cp>t5i and cp<t5f]
    if len(CP_list5) > 0 and m in methods_ps:
        CP_dict5[m] = CP_list5

methods2 = set(CP_dict2.keys())
methods5 = set(CP_dict5.keys())
methods_list = list(methods2.union(methods5))


fig = plt.figure(constrained_layout=True, figsize=(4,3))
ax = fig.subplot_mosaic([['legend'],[0],[1]],
                          gridspec_kw={'height_ratios':[0.001,1,1]})
ax['legend'].axis('off')

ax[0].set_title('Segment 2')
ax[0].plot(t2,y2, linewidth=0.5, marker='.', markersize=2)
ax[0].grid(linestyle=':')
ax[0].set_ylabel('ms', fontsize=6)
ax[0].tick_params(axis='both', labelsize=6)
ax[0].set_xticks(np.arange(180,300,10))
ax[0].set_yticks(np.arange(150,162,2))
ax[0].set_ylim([150,160])

ax[1].set_title('Segment 5')
ax[1].plot(t5,y5, linewidth=0.5, marker='.', markersize=2)
ax[1].grid(linestyle=':')
ax[1].set_ylabel('ms', fontsize=6)
ax[1].tick_params(axis='both', labelsize=6)
ax[1].set_xlabel('sample (t)', fontsize=6)
ax[1].set_xticks(np.arange(600,975,25))
ax[1].set_yticks(np.arange(154,168,2))
ax[1].set_ylim([154,166])

y0,y1 = ax[0].get_ylim()
#y2 = y0+(y1-y0)/2
y2_ = 0.99*y1
d = (y2_-y0)/6
y0 = y0+d

i = 0
for m,CP_list in CP_dict2.items():
    if m != 'pelt_np':
        for cp in CP_list:
            ax[0].axvline(cp, color='r', alpha=0.5, linewidth=0.5)
            ax[0].plot(cp, y0+i*d, 
                       marker=markers_dict[method_name(m)], 
                       markersize=3, 
                       color='r')
        i = i+1

if 'pelt_np' in methods2:
    i = i-1
    CP = np.array(CP_dict2['pelt_np'])-1
    for cp in CP:
        ax[0].axvline(cp, color='g', alpha=0.5, linewidth=1)
        ax[0].plot(cp, y0+(i+1)*d, 
                   marker=markers_dict['Pelt-NP'], 
                   markersize=4, 
                   color='g')

y0,y1 = ax[1].get_ylim()
#y2 = y0+(y1-y0)/2
y2_ = 0.99*y1
d = (y2_-y0)/6
y0 = y0+d

i = 0
for m,CP_list in CP_dict5.items():
    if m != 'pelt_np':
        for cp in CP_list:
            ax[1].axvline(cp, color='r', alpha=0.5, linewidth=0.5)
            ax[1].plot(cp, y0+i*d, 
                       marker=markers_dict[method_name(m)], 
                       markersize=3, 
                       color='r')
        i = i+1

if 'pelt_np' in methods5:
    i = i-1
    CP = np.array(CP_dict5['pelt_np'])-1
    for cp in CP:
        ax[1].axvline(cp, color='g', alpha=0.5, linewidth=1)
        ax[1].plot(cp, y0+(i+1)*d, 
                   marker=markers_dict['Pelt-NP'], 
                   markersize=4, 
                   color='g')

# draw the legend
lines_leg = [mlines.Line2D([], [], 
                           color='r', 
                           marker=markers_dict[method_name(m)], 
                           linewidth=0, 
                           markersize=3, 
                           label=method_name(m)) for m in methods_list
             if (m!='pelt_np' and m != 'vwcd')]


if 'vwcd' in methods_list:
    lines_leg = lines_leg + [mlines.Line2D([], [], 
                               color='tab:purple', 
                               marker=markers_dict['VWCD'], 
                               linewidth=0, 
                               markersize=3, 
                               label='VWCD')]


if 'pelt_np' in methods_list:
    lines_leg = lines_leg + [mlines.Line2D([], [], 
                               color='g', 
                               marker=markers_dict['Pelt-NP'], 
                               linewidth=0, 
                               markersize=4, 
                               label='Pelt-NP')]

_ = ax['legend'].legend(handles=lines_leg, 
                    loc='upper center',
                    ncol=3, 
                    fontsize=6,
                    handletextpad=0.01,
                    columnspacing=0.5)
