# -*- coding: utf-8 -*-
"""
ARL comparing - Shewhart, EWMA, CUSUM
@author: Cleiton Moya de Almeida
"""

import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import matplotlib.pyplot as plt
numpy2ri.activate()

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

spc = importr('spc')

ro.r('arl_sw1 <- sapply(seq(0,10,by=0.01), c=1, type="1", xshewhartrunsrules.arl)')
ro.r('arl_ew1 <- sapply(seq(0,10,by=0.01), l=0.10, c=4, sided="two",limits="fix",q=1, steady.state.mode="conditional",r=40, xewma.arl)')
ro.r('arl_cs1 <- sapply(seq(0,10,by=0.01), k=1, h=5, hs = 0, sided = "two", method = "igl", q = 1, r = 30, xcusum.arl)')


ro.r('arl_sw2 <- sapply(seq(0,10,by=0.01), c=1, type="1", xshewhartrunsrules.arl)')
ro.r('arl_ew2 <- sapply(seq(0,10,by=0.01), l=0.10, c=2.701, sided="two",limits="fix",q=1, steady.state.mode="conditional",r=40, xewma.arl)')
ro.r('arl_cs2 <- sapply(seq(0,10,by=0.01), k=0.5, h=4.774, hs = 0, sided = "two", method = "igl", q = 1, r = 30, xcusum.arl)')

arl_sw1 = ro.r['arl_sw1']
arl_ew1 = ro.r['arl_ew1']
arl_cs1 = ro.r['arl_cs1']

arl_sw2 = ro.r['arl_sw2']
arl_ew2 = ro.r['arl_ew2']
arl_cs2 = ro.r['arl_cs2']

Mu = np.arange(0,10.01,0.01)
fig,ax = plt.subplots(figsize=(5,2.5), ncols=2, layout='constrained', sharey=True)

ax[0].plot(Mu, arl_sw1, label=r'Shewhart ($\kappa=3$)')
ax[0].plot(Mu, arl_ew1, label=r'EWMA ($\lambda=0.1$, $\kappa_d=4$)')
ax[0].plot(Mu, arl_cs1, label=r'Cusum ($\delta=2$, $h=5$)')
ax[0].set_yscale('log')
ax[0].grid('both', linestyle=':')
ax[0].legend()
ax[0].set_ylim([10**0,10**5])
ax[0].set_xlim([0,10])
ax[0].set_xlabel(r'$\delta$')
ax[0].set_ylabel('samples')

ax[1].plot(Mu, arl_sw2, label=r'Shewhart ($\kappa=3$)')
ax[1].plot(Mu, arl_ew2, label=r'EWMA ($\lambda=0.1$, $\kappa_d=2.701$)')
ax[1].plot(Mu, arl_cs2, label=r'Cusum ($\delta=1$, $h=4.774$)')
ax[1].set_yscale('log')
ax[1].grid('both', linestyle=':')
ax[1].legend()
ax[1].set_ylim([10**0,10**5])
ax[1].set_xlim([0,10])
ax[1].set_xlabel(r'$\delta$')