# -*- coding: utf-8 -*-
"""
Change-point vs Anomalies
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

rng = np.random.default_rng(seed=42)
N = 20
y = rng.normal(10,1,N)

ya = [y.copy() for _ in range(6)]
yv = 20

ya[1][10] = yv
ya[2][10:12] = yv
ya[3][10:13] = yv
ya[4][10:16] = yv
ya[5][10:] = yv

x = np.arange(1,25,5)
fig,axs = plt.subplots(layout='constrained', nrows=2, ncols=3, 
                       figsize=(5,2.5), sharex=True, sharey=True)
cap = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']

for j,ax in enumerate(axs.ravel()):
    ax.set_ylim([0,25])
    ax.set_xticks(np.arange(0,25,5))
    ax.grid(linestyle=':')
    ax.set_title(f'case {cap[j]}')
    ax.plot(ya[j], linewidth=0.5, marker='.')
axs[1,0].set_xlabel('sample (t)', fontsize=6)
axs[1,1].set_xlabel('sample (t)', fontsize=6)
axs[1,2].set_xlabel('sample (t)', fontsize=6)

