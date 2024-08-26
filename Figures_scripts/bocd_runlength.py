# -*- coding: utf-8 -*-
"""
BOCD Runlength example
@author: cleiton
"""

import numpy as np
from   scipy.stats import norm
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm


plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

rng = np.random.default_rng(42)
y1 = rng.normal(loc=1,scale=1, size=5)
y2 = rng.normal(loc=10,scale=1, size=3)
y3 = rng.normal(loc=5,scale=1, size=4)

y = np.concatenate((y1, y2, y3))
T = len(y)

R        = np.zeros((T+1, T+1))
R[0, 0]  = 1
max_R    = np.empty(T+1)
max_R[0] = 1

mu0 = 5
lamb0 = 1
H           = 1/10
mu_params   = np.array([mu0])
lamb_params = np.array([lamb0])

rl_pred = lambda x, mu, lamb: norm.pdf(x, mu, 1/lamb + 1)

for t in range(1, T+1):
    x = y[t-1]

    pis = np.array([rl_pred(x, mu_params[i], lamb_params[i]) for i in range(t)])

    R[t, 1:t+1] = R[t-1, :t] * pis * (1-H)
    R[t, 0]     = sum(R[t-1, :t] * pis * H)
    R[t, :]    /= sum(R[t, :])

    max_R[t] = np.argmax(R[t, :])

    offsets     = np.arange(1, t+1)
    mu_params   = np.append([mu0], (mu_params * offsets + x) / (offsets + 1))
    lamb_params = np.append([lamb0], lamb_params + 1)


fig, ax = plt.subplots(2, 1, figsize=(5,3), sharex=True, layout='constrained')
ax[0].set_title('Timeseries with change-points')
ax[0].plot(y, linewidth=0.5, marker='o', markersize=2)
ax[0].axvline(5, linewidth=1, linestyle='--', color='r', label='change-point')
ax[0].axvline(8, linewidth=1, linestyle='--', color='r')
ax[0].legend()
ax[0].set_xticks(np.arange(13))
ax[0].set_yticks(np.arange(-2,14,2))

ax[1].set_title('Run length posterior probability')
im = ax[1].imshow(np.rot90(R[1:]), aspect='auto', cmap='Blues', extent=[0,T,0,T], 
                  norm=LogNorm(vmin=0.001, vmax=1))

ax[1].set_ylim(1,6)
ax[1].plot(max_R[1:], color='r', label='MAP', marker='o', markersize=4)
ax[1].axvline(5, linewidth=1, linestyle='--', color='r')
ax[1].axvline(8, linewidth=1, linestyle='--', color='r')
ax[1].legend()
ax[1].set_xlabel('Sample (t)')
clb = fig.colorbar(im, ax=ax[1], pad=0.01)