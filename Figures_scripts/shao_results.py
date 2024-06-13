# -*- coding: utf-8 -*-
"""
Shao Dataset - Change-point results
@author: Cleiton Moya de Almeida
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

methods = ['shewhart_ba', 'shewhart_ps', 'ewma_ba', 'ewma_ps',
           'cusum_2s_ba', 'cusum_2s_ps', 'cusum_wl_ba', 'cusum_wl_ps',
           'bocd_ba', 'bocd_ps', 'rrcf_ps', 'vwcd', 'pelt_np']

df = pd.concat([pd.read_pickle(f'../Experiment/results_shao/df_shao_{m}.pkl') for m in methods], 
               ignore_index=True)

# Include the method_type collumns to differentiate vanilla from non-vanilla
def method_type(m):
    if m[-2:] == 'ba':
        return 'Basic'
    elif m=='pelt_np':
        return 'Reference (off-line)'
    else:
        return 'Proposed'

def method_name(m):
    if m[:8] == 'cusum_2s':
        return '2S-Cusum'
    elif m[:8] == 'cusum_wl':
        return 'WL-Cusum'
    elif m[:4] == 'ewma':
        return 'EWMA'
    elif m[:8] == 'shewhart':
        return 'Shewhart'
    elif m=='pelt_np':
        return 'Pelt-NP'
    elif m=='vwcd':
        return 'VWCD'
    elif m=='rrcf_ps':
        return 'RRCF'
    elif m[:4]=='bocd':
        return 'BOCD'

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

df['method_type'] = [method_type(m) for m in df['method']]
df['method_name'] = [method_name(m) for m in df['method']]
df['method_order'] = [method_order(m) for m in df['method']]
df.sort_values(by='method_order', inplace=True)

metrics = ['precision', 'recall', 'f1']
metrics_name = ['Precision', 'Recall', 'F1 score']
reference1 = [0.38, 0.91, np.nan, 0.70] #HDP-HMM (Mouchet, 2019)

flierprops = {'markersize':2}
C0 = np.array([142, 186, 217])/255 # blue
C1 = np.array([255, 190, 134])/255 # orange
C2 = np.array([149, 207, 149])/255 # green
hatches = ['', '////']

# Boxplot of metrics
def add_median_labels(ax: plt.Axes, bx:'all', fmt: str = ".2f"):
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    median_lines = np.array(lines[start::lines_per_box])
    if bx!='all':
        median_lines = median_lines[bx]
        
    for i,median in zip(bx,median_lines):
        x, y = (data.mean() for data in median.get_data())
        
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        
        if i <= 10:
            y_ = y+0.48
        else:
            y_ = y-0.4
        text = ax.text(x, y_, f'{value:{fmt}}', ha='center', va='center',
                           fontweight='bold', color='r', fontsize=6)
        text.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='w'),
            path_effects.Normal(),
        ])


fig = plt.figure(constrained_layout=True, figsize=(5.5,3))
ax = fig.subplot_mosaic([['legend', 'legend', 'legend'],[0,1,2]], 
                          gridspec_kw={'height_ratios':[0.001, 1]})
ax['legend'].axis('off')
boxes_annot = [[11,12], [9,12], [9,12]]


methods_name = df.method_name.unique()
for j,met in enumerate(metrics):
    for i,m in enumerate(methods_name):
        df_ = df[df.method_name==m]
        if len(df_.method_type.unique())==2:
            if i==0:
                _ = sns.boxplot(data=df_, y="method_name", x=met, hue='method_type',
                                  ax=ax[j], saturation=1, notch=True, width=0.7,
                                  showmeans=False, flierprops=flierprops, legend=True,
                                  medianprops={'linewidth':1}, hue_order=['Basic', 'Proposed'],
                                  palette=[C0,C1])
                handles, labels = ax[j].get_legend_handles_labels()
                ax[j].get_legend().remove()
            else:
                _ = sns.boxplot(data=df_, y="method_name", x=met, hue='method_type',
                                  ax=ax[j], saturation=1, notch=True, width=0.7,
                                  showmeans=False, flierprops=flierprops, legend=False,
                                  medianprops={'linewidth':1}, hue_order=['Basic', 'Proposed'],
                                  palette=[C0,C1])
            
                
        else:
            if m=='Pelt-NP':
                _ = sns.boxplot(data=df_, y="method_name", x=met, hue='method_type',
                                  ax=ax[j], saturation=1, notch=True, width=0.7/2,
                                  showmeans=False, flierprops=flierprops, legend=True,
                                  medianprops={'linewidth':1}, 
                                  palette=[C2])
                handles, labels = ax[j].get_legend_handles_labels()
                ax[j].get_legend().remove()
            else:
                _ = sns.boxplot(data=df_, y="method_name", x=met, hue='method_type',
                                  ax=ax[j], saturation=1, notch=True, width=0.7/2,
                                  showmeans=False, flierprops=flierprops, legend=False,
                                  medianprops={'linewidth':1}, 
                                  palette=[C1])
    if j > 0:
        ax[j].set(yticklabels=[]) 
    ax[j].set_xlabel(metrics_name[j])
    ax[j].set_ylabel('')
    ax[j].grid(axis='x', linestyle=':')
    ax[j].set_xticks(np.arange(0,1.25,0.25))
    ax[j].tick_params(axis='both', labelsize=8)
    add_median_labels(ax[j], bx=boxes_annot[j])

ax['legend'].legend(handles, labels, loc="upper center", ncol=3, 
                    fontsize=8, frameon=True)


#Total number of changepoints per method
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

fig = plt.figure(constrained_layout=True, figsize=(4,3))
ax = fig.subplot_mosaic([['legend', 'legend'],[0,1]], 
                          gridspec_kw={'height_ratios':[0.001, 1]})
ax['legend'].axis('off')

df_ = df[df.method.isin(methods)]
df_ = df_.groupby(['method'], as_index=False, )['num_cp_pred'].sum()
df_['method_type'] = [method_type(m) for m in df_['method']]
df_['method_name'] = [method_name(m) for m in df_['method']]
df_['method_order'] = [method_order(m) for m in df_['method']]
df_.sort_values(by='method_order', inplace=True)

_ = sns.barplot(data=df_, y='method_name', x='num_cp_pred', hue='method_type',
                errorbar=None, palette=[C0, C1, C2], saturation=1, 
                ax=ax[0], zorder=2, width=1, 
                hue_order=['Basic', 'Proposed' ,'Reference (off-line)'])

ax[0].set_xlabel("Number of changepoints")
ax[0].set_ylabel("")
ax[0].tick_params(axis='both', labelsize=8)
handles, labels = ax[0].get_legend_handles_labels()
ax[0].grid(axis='x', zorder=1, linestyle=':')
ax[0].set_xticks(np.arange(0,6000,1000))
ax[0].set_xlim([0,5000])
ax[0].set_xticklabels(['0', '1k', '2k', '3k', '4k', '5k'])
ax[0].annotate(f"{df[df.method=='cusum_wl_ba'].num_cp_pred.sum()} \n(out of axis lim.)", (5000,3), (4800,4), arrowprops={'arrowstyle':'<-', 'color':'r'}, 
            horizontalalignment='right', verticalalignment='center', fontsize=6)

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
ax['legend'].legend(handles, labels, loc="upper center", 
                    ncol=3, fontsize=8, frameon=True, handletextpad=0.1,
                    columnspacing=1)