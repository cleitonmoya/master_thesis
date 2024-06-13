# -*- coding: utf-8 -*-
"""
Change-point types
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

rng = np.random.default_rng(42)
y11 = rng.normal(loc=10,scale=1, size=200)
y12 = rng.normal(loc=5,scale=1, size=200)

y21 = rng.normal(loc=0,scale=1, size=200)
y22 = rng.normal(loc=0,scale=2, size=200)

y1 = np.concatenate((y11, y12))
y2 = np.concatenate((y21, y22))

fig,ax = plt.subplots(nrows=2, figsize=(5,3), sharex=True, layout='constrained')

ax[0].set_title('Additive change-point')
ax[0].plot(y1, linewidth=0.5)
ax[0].axvline(200, linewidth=1, color='r', label='change-point')
ax[0].legend()

ax[1].set_title('Non-additive change-point')
ax[1].plot(y2, linewidth=0.5)
ax[1].axvline(200, linewidth=1, color='r')
ax[1].set_xlabel('samples')