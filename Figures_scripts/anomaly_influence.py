# -*- coding: utf-8 -*-
"""
Anomaly influence on change-point methods statistics
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

# Generate a random timeseries
rng = np.random.default_rng(seed=42)
mu1 = 5
mu2 = 7
s = 1
y = rng.normal(loc=mu1, scale=s, size=40)
T = len(y)

# Insert a point-anomaly
t_anom = 20
y_anom = 20
y[t_anom] = y_anom

# hyper-parameters
h = 5
delta = 2
lamb = 0.1
kd = 4

G = []
CP = []
H = []
Z = []
Ush = []
Lsh = []
Uew = []
Lew = []

gt = 0
z = y[0]

for t in range(T):
    
    ush = mu1 + 3*s
    lsh = mu1 - 3*s
    
    gt = gt + norm.logpdf(y[t], mu2, s) - norm.logpdf(y[t], mu1, s)
    gt = np.heaviside(gt,0)*gt
    
    z = lamb*y[t] + (1-lamb)*z
    uew = mu1 + kd*s*np.sqrt((lamb/(2-lamb)))
    lew = mu1 - kd*s*np.sqrt((lamb/(2-lamb)))
    
    G.append(gt)
    H.append(h)
    Z.append(z)
    Ush.append(ush)
    Lsh.append(lsh)
    Uew.append(uew)
    Lew.append(lew)        

    
# Plot
fig,ax = plt.subplots(nrows=3, figsize=(4,3.5), sharex=True, layout='constrained')
ax[0].set_title('Shewhart control chart')
ax[0].plot(y, linewidth=0.5, marker='o', markersize=1)
ax[0].scatter(t_anom, y_anom, marker='o', color='r', s=5, label='anomaly')
ax[0].plot(Ush, linewidth=0.5, linestyle='--', color='r', label=r'$\mu_0 \pm 3\sigma_0$')
ax[0].plot(Lsh, linewidth=0.5, linestyle='--', color='r')
ax[0].legend(fontsize=6)
ax[0].set_yticks(np.arange(0,30,5))
ax[0].set_ylim(0,25)
ax[0].set_xlim(0,40)

ax[1].set_title('CUSUM statistic')
ax[1].plot(G, linewidth=0.5, marker='o', markersize=1, label='statistic')
ax[1].plot(H, linewidth=0.5, linestyle='--', color='r', label='threshold')
ax[1].set_yticks(np.arange(0,35,5))
ax[1].legend(fontsize=6)

ax[2].set_title('EWMA statistic')
ax[2].plot(Z, linewidth=0.5, marker='o', markersize=1, label='statistic')
ax[2].plot(Uew, linestyle='--', linewidth=0.5, color='r', label='control limits')
ax[2].plot(Lew, linestyle='--', linewidth=0.5, color='r')
ax[2].legend(fontsize=6, loc='lower right')
ax[2].set_xlabel('sample (t)', fontsize=6)
ax[2].set_yticks(np.arange(0,10,2))
