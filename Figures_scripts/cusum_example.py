# -*- coding: utf-8 -*-
"""
CUSUM basic example
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

# Generate a random timeseries with change-point at t=tau
rng = np.random.default_rng(seed=41)
mu1 = 10
mu2 = 11
s = 1
tau = 30
y1 = rng.normal(loc=mu1, scale=s, size=tau)
y2 = rng.normal(loc=mu2, scale=s, size=20)
y = np.concatenate((y1, y2))
T = len(y)

# Cusum hyper-parameters
h = 5
delta = 2

G = []
CP = []
H = []
Ush = []
Lsh = []

gt = 0
llr = 0
first = True

Llr = []
for t in range(T):
    
    ush = mu1 + 3*s
    lsh = mu1 - 3*s
    
    gt = gt + norm.logpdf(y[t], mu2, s) - norm.logpdf(y[t], mu1, s)
    gt = np.heaviside(gt,0)*gt
    llr = llr + norm.logpdf(y[t], mu2, s) - norm.logpdf(y[t], mu1, s)
    
    if gt > h and first:
        CP = t
        print(f'changepoint at CP = {t}')
        first = False
    
    G.append(gt)
    H.append(h)
    Llr.append(llr)
    Ush.append(ush)
    Lsh.append(lsh)
     
    
# Plot
fig,ax = plt.subplots(nrows=3, figsize=(4,3.5), 
                      sharex=True, layout='constrained')
ax[0].set_title('Shewhart control chart')
ax[0].plot(y, linewidth=0.5, marker='o', markersize=1)
ax[0].hlines(y=mu1, xmin=0, xmax=30, label=r'$\mu$', color='C2', linewidth=0.5, linestyle='--')
ax[0].hlines(y=mu2, xmin=30, xmax=50,color='C2', linewidth=0.5, linestyle='--')
ax[0].axvline(x=tau, color='C2', linewidth=1, label='label')

ax[0].plot(Ush, linewidth=0.5, linestyle='-', color='r', label=r'$\mu_0 \pm 3\sigma$')
ax[0].plot(Lsh, linewidth=0.5, linestyle='-', color='r')
ax[0].legend(fontsize=6, loc='lower left')
ax[0].set_yticks(np.arange(6,15,2))
ax[0].grid(axis='both', linestyle=':')

ax[1].set_title('Log-likelihood ratio')
ax[1].plot(Llr, linewidth=0.5, marker='o', markersize=1)
ax[1].axvline(x=30, color='C2', linewidth=1, label='label')
ax[1].grid(axis='both', linestyle=':')


ax[2].set_title('CUSUM statistic')
ax[2].plot(G, linewidth=0.5, marker='o', markersize=1)
ax[2].plot(H, linewidth=0.5, linestyle='--', color='r', label='threshold')
ax[2].axvline(x=tau, color='C2', linewidth=1)
ax[2].axvline(x=CP, color='r', linewidth=1, label='change-point (detected)')
ax[2].grid(axis='both', linestyle=':')
ax[2].legend(fontsize=6)
ax[2].set_xlabel('sample (t)', fontsize=6)
