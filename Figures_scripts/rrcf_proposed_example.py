# -*- coding: utf-8 -*-
"""
RRCF proposed scheme for change-point detection
@author: Cleiton Moya de Almeida
"""
import numpy as np
import rrcf
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
np.random.seed(42)

#0..10
typeA_rtt = ['dca6326b9ca8_gig03_d_rttmean.txt',
             'dca6326b9ca8_gig04_d_rttmean.txt',
             'dca6326b9ca8_gru05_d_rttmean.txt',
             'dca6326b9ca8_rnp_rj_d_rttmean.txt',
             'e45f01359a20_gig01_d_rttmean.txt',
             'e45f01359a20_gig03_d_rttmean.txt',
             'e45f01359a20_gig04_d_rttmean.txt',
             'e45f01359a20_gru02_d_rttmean.txt',
             'e45f01359a20_gru05_d_rttmean.txt',
             'e45f01359a20_rnp_rj_d_rttmean.txt',
             'dca6326b9aa1_gig04_u_rttmean.txt']

#0..7
typeA_down = ['dca6326b9aa1_gig03_d_throughput.txt',
              'dca6326b9c99_gig01_d_throughput.txt',
              'dca6326b9c99_gig02_d_throughput.txt',
              'dca6326b9c99_gig03_d_throughput.txt',
              'dca6326b9c99_gig04_d_throughput.txt',
              'dca6326b9ca8_gig03_d_throughput.txt',
              'dca6326b9ca8_gig04_d_throughput.txt',
              'e45f01359a20_gig01_d_throughput.txt',
              'dca6326b9ca8_gig03_d_throughput.txt']

#0..8
typeAN_down = ['dca6326b9ca8_gig02_d_throughput.txt',
               'dca6326b9ca8_rnp_rj_d_throughput.txt',
               'e45f01359a20_gig03_d_throughput.txt',
               'e45f01963c21_gig01_d_throughput.txt',
               'e45f01963c21_gig03_d_throughput.txt',
               'dca6326b9ce4_gig01_d_throughput.txt',
               'dca6326b9ce4_gig02_d_throughput.txt',
               'dca6326b9ce4_gig03_d_throughput.txt',
               'dca6326b9ce4_rnp_sp_d_throughput.txt',
               'dca6326b9aa1_gru05_d_throughput.txt']

#0..8
others = ['e45f01963c21_gig03_d_throughput.txt',
          'e45f01963c21_gig03_d_rttmean.txt',
          'e45f01963c21_gig03_u_rttmean.txt',
          'e45f01963c21_rnp_sp_u_throughput.txt',
          'e45f01963c21_gig04_d_rttmean.txt',
          'e45f01963c21_gru02_d_throughput.txt',
          'e45f01963c21_gru02_d_rttmean.txt',
          'e45f01963c21_gru02_u_throughput.txt',
          'e45f01963c21_gru02_u_rttmean.txt']

file = typeAN_down[9]
y = np.loadtxt(f'../Dataset/ndt/{file}', usecols=1, delimiter=',')
verbose = False

# Hyperparamameters
num_trees = 40
shingle_size = 1
tree_size = 100
thr = 25
rl = 4

# PROPOSED

# Auxiliar variables
N = len(y)
c = 0               # statistic deviation counter
lcp = 0             # last change point
CP2 = []             # changepoint list 
cp = True
   
# Use the "shingle" generator to create rolling windows
shingles_stream = rrcf.shingle(y, size=shingle_size)
Scores2 = [np.nan]*(shingle_size-1)
Sb = 0
Sn = 0
ca = 0

S = {}
Anom = []
# For each shingle...
for index, point in enumerate(shingles_stream):
    
    t = index+shingle_size-1
    if cp:
        # Create a dict to store anomaly score of each point
        S = {}
        
        # Create a forest of empty trees
        forest = []
        for _ in range(num_trees):
            forest.append(rrcf.RCTree())
        n = 0
    
    if c==0:
        Sb = Sn
    # For each tree in the forest
    for tree in forest:
        # If tree is above permitted size, drop the oldest point (FIFO)
        if len(tree.leaves) > tree_size:
            tree.forget_point(list(tree.leaves)[1])
        
        # Insert the new point into the tree
        tree.insert_point(point, index=n)
        
        # Compute codisp on the new point and take the average among all trees
        if not n in S:
            S[n] = 0
        S[n] += tree.codisp(n)/num_trees
    #print(S[n])
        
    # if statistic deviate, remove the point from the tree and check the next.
    # if after rl points the statistic deviation persists, declare a changepoint
    Sn = S[n]
    Scores2.append(Sn)
    dev = Sn > thr
    if dev:
        if verbose>=2: print(f't={t}:possible changepoint')
        if t-1 not in Anom:
            Anom.append(t)
            ca = ca + 1
        c = c+1
        #print(f't={n}: possible changepoint')
        # Forget the point
        for tree in forest:
            tree.forget_point(n)
            #S[n]=tree.codisp(n)/num_trees
        S[n] = Sb
        Sn = Sb
        #print(f'n={n}, dev, S={S[n]}')

        # Check for changepoint
        if c==rl:
            lcp = t-rl
            if verbose: print(f't={t}:changepoint at {lcp}')
            CP2.append(lcp)
            Anom = Anom[:-ca]
            c = 0
            ca = 0
            cp = True
            
    else:
        c = 0
        ca = 0
        cp = False
    
    n = n+1
    

#%%
fig, ax = plt.subplots(nrows=2, figsize=(5, 2.7), 
                       sharex=True, layout='constrained')

ax[0].set_title('Proposed RRCF')
ax[0].scatter(Anom, y[Anom], color='r', s=5, label='anomaly')
ax[0].plot(y, linewidth=0.5)
if len(CP2)>0:
    for j,cp in enumerate(CP2):
        if j==0:
            ax[0].axvline(cp, color='r', linestyle='-', linewidth=0.5, label='change-point')
        else:
            ax[0].axvline(cp, color='r', linestyle='-', linewidth=0.5)
    ax[0].legend()
ax[0].tick_params(axis='both', labelsize=6)
ax[0].set_xticks(np.arange(0,550,50))
#ax[0].set_xlim([0,500])
ax[0].grid(axis='both', zorder=1, linestyle=':')
ax[0].set_ylabel('Mbits/s', fontsize=6)

ax[1].set_title('RRCF statistic')
#ax[1,1].plot(Scores2, linewidth=0.5, marker='o', markersize=1)
ax[1].plot(Scores2, linewidth=0.5)
ax[1].axhline(thr, color='r', linewidth=0.5, label='threshold')
ax[1].set_xlabel('sample (t)', fontsize=6)
ax[1].tick_params(axis='both', labelsize=6)
ax[1].grid(axis='both', zorder=1, linestyle=':')
ax[1].legend()
