# -*- coding: utf-8 -*-
"""
Isolation-based anomaly detection example
@author: Cleiton Moya de Almeida
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)


def count_points_region(R, points_set):
    xmin,xmax,ymin,ymax =  R
    c = 0
    for px,py in points_set:
        if (px >= xmin) and (px <= xmax) and (py >= ymin) and (py <= ymax):
            c = c+1
    return c

def randon_cut(R, anom):
    xmin,xmax,ymin,ymax =  R
    ax,ay = anom
    d = np.random.choice([0,1])
    if d == 0:
        cut = np.random.uniform(xmin,xmax)
        if cut < ax:
            xmin = cut
        else:
            xmax = cut
        cut_line = ((cut,cut),(ymin,ymax))
        
    else:
        cut = np.random.uniform(ymin,ymax)
        if cut < ay:
            ymin = cut
        else:
            ymax = cut
        cut_line = ((xmin,xmax),(cut,cut))
        
    R = (xmin,xmax,ymin,ymax)
    return cut_line,R

# Paramters
mean = [0, 0]
cov = [[1, 0], [0, 1]]

# Generate the point set
p1 = (4,4) # anomaly
p2 = (0,0) # normal point
x,y = np.random.multivariate_normal(mean, cov, 1000).T
point_set = set(zip(x,y))

fig,axs = plt.subplots(figsize=(5,2.5), layout='constrained', 
                       ncols=2, sharex=True, sharey=True)


axs[0].scatter(x,y, s=1, alpha=0.5)
axs[0].scatter(p1[0], p1[1], s=10, c='r',label='anomaly')
axs[0].scatter(p2[0], p2[1], s=1, color='C0', alpha=0.5)

axs[1].scatter(x,y, s=1, alpha=0.5)
axs[1].scatter(p2[0], p2[1], s=10, c='r',label='normal')
axs[1].scatter(p1[0], p1[1], s=1, color='C0', alpha=0.5)

xmin0, xmax0 = axs[0].get_xlim()
ymin0, ymax0 = axs[0].get_ylim()

for j in (0,1):
    axs[j].set_box_aspect(1)
    axs[j].legend()
    axs[j].set_xticks(range(-4,5))
    axs[j].set_yticks(range(-4,5))
    axs[j].set_xlim(xmin0, xmax0)
    axs[j].set_ylim(ymin0, ymax0)


R = (xmin0, xmax0, ymin0, ymax0)
n_lim = 100
n = 0
num_p = len(point_set)
while num_p != 0 and n < n_lim:
    cut_line, R = randon_cut(R, p1)
    xmin,xmax,ymin,ymax =  R
    num_p = count_points_region(R, point_set)
    axs[0].plot(cut_line[0], cut_line[1], c='r', linewidth=0.5)
    n = n+1
print(n)

R = (xmin0, xmax0, ymin0, ymax0)
n = 0
num_p = len(point_set)
while num_p != 0 and n < n_lim:
    cut_line, R = randon_cut(R, p2)
    xmin,xmax,ymin,ymax =  R
    num_p = count_points_region(R, point_set)
    axs[1].plot(cut_line[0], cut_line[1], c='r', linewidth=0.5)
    n = n+1
print(n)

axs[0].set_xlabel(r'$x_1$')
axs[1].set_xlabel(r'$x_1$')
axs[0].set_ylabel(r'$x_2$')
axs[1].set_ylabel(r'$x_2$')