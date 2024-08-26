# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 01:22:28 2024

@author: cleiton
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

path = '../Dataset/shao/'
files = [f for f in os.listdir(path) if f[-3:]=='csv']

'''
#%%
if os.path.isfile('df_med_shao.pkl'):
    df_med = pd.read_pickle('df_med_shao.pkl')
else:
    Med = []
    for n,file in enumerate(files):
    
        df = pd.read_csv(f'{path}{file}', sep=';', parse_dates=[0], dayfirst=True)
        CP_aux = df.cp.values
        CP_label = np.argwhere(CP_aux==1)
    
        mean_t = np.round(df.epoch.diff().mean().seconds/60,1)
        num_days = (df.epoch.iloc[-1] - df.epoch.iloc[0]).days
        
        med = {'serie':file[:-3], 
               'num_med': len(df),
               'timestamp_i': df.epoch.iloc[0],
               'timestamp_f': df.epoch.iloc[-1],
               'num_cp':len(CP_label),
               'mean_t': mean_t,
               'num_days': num_days
               }
        Med.append(med)
    
    df_med = pd.DataFrame(Med)
    #df_med.to_pickle('df_med_shao.pkl')

#%% Summary statistics
print('Number of measurements:')
print(f'\ttotal: {df_med.num_med.sum()}')
print(f'\tmin: {df_med.num_med.min()}')
print(f'\tmax: {df_med.num_med.max()}')
print(f'\tmedian: {np.median(df_med.num_med)}')
print(f'\tmean: {np.mean(df_med.num_med)}')


print('\nNumber of measurements days:')
print(f'\tmin: {df_med.num_days.min()}')
print(f'\tmax: {df_med.num_days.max()}')
print(f'\tmedian: {np.median(df_med.num_days)}')
print(f'\tmean: {np.mean(df_med.num_days)}')

print('\nNumber of changepoints:')
print(f'\ttotal: {df_med.num_cp.sum()}')
print(f'\tmin: {df_med.num_cp.min()}')
print(f'\tmax: {df_med.num_cp.max()}')
print(f'\tmedian: {np.median(df_med.num_cp)}')
print(f'\tmean: {np.mean(df_med.num_cp)}')


#%% Meansurements and changepoints boxplot
fig,ax = plt.subplots(figsize=(4,2), ncols=2)
ax[0].set_title('Num. of days')
ax[0].boxplot(df_med.num_days, medianprops={'color':'red'})
ax[0].set_xticks([1], [''])
ax[0].set_box_aspect(1)

ax[1].set_title('Num. of changepoints')
ax[1].boxplot(df_med.num_cp, medianprops={'color':'red'})
ax[1].set_xticks([1], [''])
ax[1].set_box_aspect(1)
ax[1].set_ylim(0,100)
plt.tight_layout()
'''

#%% Examples of timeseries
N = 1000
np.random.seed(42)
files_ex = sorted(np.random.choice(files, size=6, replace=False))

fig = plt.figure(constrained_layout=True, figsize=(5,4))
ax = fig.subplot_mosaic([['legend', 'legend',],[0,1], [2,3], [4,5]], 
                          gridspec_kw={'height_ratios':[0.001, 1, 1, 1]})
ax['legend'].axis('off')

#fig,ax = plt.subplots(figsize=(5,4), nrows=3, ncols=2, layout='constrained')


first_change = True

for i,file in enumerate(files_ex):
    df = pd.read_csv(f'{path}{file}', sep=';', parse_dates=[0], dayfirst=True)[:N]
    CP_aux = df.cp.values
    CP_label = np.argwhere(CP_aux==1)
    y = df.rtt.values
    ax[i].set_title(file[:-4])
    ax[i].plot(y, linewidth=0.3)
    ax[i].tick_params(axis='both', labelsize=6)
    
    ax[i].set_xlabel('sample (t)', fontsize=6)
    ax[i].set_ylabel('ms', fontsize=6)
    if i == 0:
        y_0 = y
        CP_label_0 = CP_label
    else:
        for j,cp in enumerate(CP_label):
            if j == 0 and first_change:
                ax[i].axvline(cp, color='red', linewidth=0.5, label='change-point label')
                handles, labels = ax[i].get_legend_handles_labels()
                ax['legend'].legend(handles, labels, loc="upper center", 
                                    fontsize=8, frameon=True)
                first_change = False
            else:
                ax[i].axvline(cp, color='red', linewidth=0.5)
    #ax[i].set_xlim([0,1000])
ax[4].set_yticks([190,200,210])
ax[4].set_ylim([190,210])
ax[5].set_yticks([190,200,210])
ax[5].set_ylim([190,210])

x1, x2, y1, y2 = 180, 270, 150, 190  # subregion of the original image
axins = ax[0].inset_axes([0.32, 0.3, 0.45, 0.6],
    xlim=(x1, x2), ylim=(y1, y2), yticklabels=[])
axins.plot(range(190,265), y_0[190:265], linewidth=0.5)


for j,cp in enumerate(CP_label_0):
    axins.axvline(cp, color='red', linewidth=0.5)

#axins.axvline(66, linewidth=0.5, color='r', linestyle='--')
#axins.set_yticks([530, 540])
#axins.set_yticklabels([530, 550])
#axins.yaxis.tick_right()
#axins.grid(linestyle=':')

axins.tick_params(bottom=False, left=False) 
axins.set_xticklabels('') 
axins.tick_params(labelsize=6) 
_ = ax[0].indicate_inset_zoom(axins, edgecolor="black")