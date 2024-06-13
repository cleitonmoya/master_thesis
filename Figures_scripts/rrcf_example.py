# -*- coding: utf-8 -*-
"""
RRCF anomaly detection
Example from the original paper
@author: Cleiton Moya de Almeida
"""

import numpy as np
import rrcf
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

# Generate data
n = 400
A = 50
center = 100
phi = 30
T = 2*np.pi/100
t_ = np.arange(n)
sin = A*np.sin(T*t_-phi*T) + center

# Insert anomalies
sin[235:255] = 80
sin[105] = 100
sin[305] = 100
sin[309] = 100 
Anom_label = [105, 305, 309] + list(range(235,255))
Anom = []

# Set tree parameters
num_trees = 40
shingle_size = 4
tree_size = 256
thr = 10

# Create a forest of empty trees
forest = []
for _ in range(num_trees):
    tree = rrcf.RCTree()
    forest.append(tree)
    
# Use the "shingle" generator to create rolling window
points = rrcf.shingle(sin, size=shingle_size)

# Create a dict to store anomaly score of each point
avg_codisp = {}

# For each shingle...
for index, point in enumerate(points):
    # For each tree in the forest...
    for tree in forest:
        # If tree is above permitted size, drop the oldest point (FIFO)
        if len(tree.leaves) > tree_size:
            tree.forget_point(index - tree_size)
        # Insert the new point into the tree
        tree.insert_point(point, index=index)
        # Compute codisp on the new point and take the average among all trees
        if not index in avg_codisp:
            avg_codisp[index] = 0
        avg_codisp[index] += tree.codisp(index) / num_trees
    
    # decide for anomaly
    dev = avg_codisp[index]  > thr
    if dev:
        t = index+shingle_size-1
        Anom.append(t)
   

# scores
S = [np.nan]*(shingle_size-1)+list(avg_codisp.values())

fig, ax = plt.subplots(nrows=3, figsize=(5,4), sharex=True, layout='constrained')
ax[0].set_title('Timeseries and labeled anomalies')
ax[0].plot(t_, sin, linewidth=0.5)
ax[0].scatter(Anom_label, sin[Anom_label], marker='o', s=5, label='label', color='C2')
ax[0].legend(loc='upper right')

ax[1].set_title('Timeseries and detected anomalies')
ax[1].plot(t_,sin, linewidth=0.5)
ax[1].scatter(Anom, sin[Anom], marker='o', s=5, label='detected', color='r')

x1, x2, y1, y2 = 102, 111, 45, 105 # subregion of the original image
axins = ax[1].inset_axes([0.32, 0.5, 0.2, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), yticklabels=[])
axins.plot(range(102,112), sin[102:112], linewidth=0.5, marker='o', markersize=1)
axins.set_xticks([105])
axins.tick_params(left=False) 
axins.tick_params(labelsize=6) 
_ = ax[1].indicate_inset_zoom(axins, edgecolor="black")
axins.scatter(Anom, sin[Anom], marker='o', s=5, color='r')
ax[1].legend(loc='upper right')


ax[2].set_title('RRCF statistic')
ax[2].plot(t_,S, linewidth=0.5)
x1, x2, y1, y2 = 102, 111, 0,  25 # subregion of the original image
axins = ax[2].inset_axes([0.32, 0.5, 0.2, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), yticklabels=[])
axins.plot(range(102,112), S[102:112], linewidth=0.5, marker='o', markersize=1)
axins.set_xticks([])
axins.tick_params(left=False) 
axins.tick_params(labelsize=6) 
_ = ax[2].indicate_inset_zoom(axins, edgecolor="black")
ax[2].axhline(thr, linewidth=0.5, color='r', label='threshold')
axins.axhline(thr, linewidth=0.5, color='r')
_ = ax[2].legend(loc='upper right')
ax[2].set_xlabel('sample (t)',fontsize=6)