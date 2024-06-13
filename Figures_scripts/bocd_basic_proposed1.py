# -*- coding: utf-8 -*-
"""
BOCD Basic vs Proposed - Example 1
@author: Cleiton Moya de Almeida 
"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.special import logsumexp
import time
from bocd import ConstantHazard, Gaussian, check_previous_cp

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

clients = ['dca6326b9aa1', 'dca6326b9ada', 'dca6326b9c99', 'dca6326b9ca8',
       'dca6326b9ce4', 'e45f01359a20', 'e45f01963bb8', 'e45f01963c21',
       'e45f01ad569d']
dict_client = {c:n+1 for n,c in enumerate(clients)}

# Data loading
file = 'dca6326b9ca8_rnp_rj_d_rttmean.txt'
y = np.loadtxt(f'../Dataset/ndt/{file}', usecols=1, delimiter=',').reshape(-1,1)
y = y[~np.isnan(y)]
w = 10
T = len(y)

verbose = False

# Hyperparameters
p_thr = 0.05        # probabily threshold to run lenght
K = 50              # maximum run lengh keep in memory
min_seg = 4

# Guassian model with Normal-Inverse Gamma prior and t-Student posterior
# Note that real-world implementation requires adapation to stream setting
mean0  = y[:w].mean()
kappa0 = 0.01
alpha0 = 0.01
omega0 = 0.1
model = Gaussian(mean0, kappa0, alpha0, omega0)
hazard = ConstantHazard(1e4) # Hazard probability

# Auxiliary variables initialization
CP0 = []            # list of changepoints
lcp = 0             # last changepoint (need to subtract 1)
log_message = 0
max_indices = np.array([0])     # indices keep in memory 
pmean = np.array([np.nan]*T)   # model's predictive mean.

# Lower triangular matrix with run posteriors of each run lenght size
log_R0 = -np.inf*np.ones((T+1, T+1)) 
log_R0[0, 0] = 0  # log 0 == 1
max_R0    = np.empty(T+1)
max_R0[0] = 1

startTime = time.time()
for t in range(1, T+1):
    
        # Observe new datum and datum before.
        x = y[t-1]
       
        # Evaluate the hazard function for this interval
        H = hazard(np.array(range(min(t, K))))
        log_H = np.log(H)
        log_1mH = np.log(1-H)

        # Make model predictions.
        pmean[t-1] = np.sum(np.exp(log_R0[t-1, :t]) * model.mu[:t])
        
        # Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(x, max_indices)
        
        # Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH
    
        # Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)
 
        # Calculate evidence
        new_log_joint = np.full(t+1, -np.inf)
        new_log_joint[0] = log_cp_prob
        new_log_joint[max_indices+1] = log_growth_probs
        
        # Determine run length distribution
        max_indices = (-new_log_joint).argsort()[:K]
        log_R0[t, :t+1]  = new_log_joint
        log_R0[t, :t+1] -= logsumexp(new_log_joint)
        r = np.exp(log_R0[t])
        
        max_R0[t] = np.argmax(log_R0[t, :])
        
        # Decide for a possible changepoint
        # If anomaly, update the model with the last measure (xb)
        # instead of the new one and pass the last message.
        # Wait to update with the new one only after a changepoint.
        if t>1 and r[t-lcp]<=p_thr:
            max_t0 = np.argmax(r)
            lcp=t-max_t0
            if check_previous_cp(lcp-1, CP0, min_seg):
                if verbose: print(f't={t-1} changepoint t={lcp-1} already identified')
            elif lcp-1>0:
                if verbose: print(f't={t-1} changepoint at t={lcp-1}')
                CP0.append(lcp-1)

        # Update sufficient statistic and pass the message
        model.update_params(x)
        log_message = new_log_joint[max_indices]


endTime = time.time()
elapsedTime = endTime-startTime
if verbose: print(f'\nElapsed time running: {elapsedTime:.1f}s')
R0 = np.exp(log_R0)



#  PROPOSED


# Update the joint probability and log_R matrix
def update_joint_prob(t, x, max_indices_):
    # Evaluate the hazard function for this interval
    global max_indices, new_log_joint
    
    H = hazard(np.array(range(min(t, K))))
    log_H = np.log(H)
    log_1mH = np.log(1-H)
    
    # 3. Evaluate predictive probabilities.
    log_pis = model.log_pred_prob(x=x, indices=max_indices_)
    
    # 4. Compute growth probabilities.
    log_growth_probs = log_pis + log_message + log_1mH
    
    # 5. Compute changepoint probabilities.
    log_cp_prob = logsumexp(log_pis + log_message + log_H)

    # 6. Compute joint prob
    new_log_joint = np.full(t+1, -np.inf)
    new_log_joint[0] = log_cp_prob
    new_log_joint[max_indices_+1] = log_growth_probs    
    
    # Run lenght probability matrix
    max_indices = (-new_log_joint).argsort()[:K]
    log_R[t, :t+1]  = new_log_joint
    log_R[t, :t+1] -= logsumexp(new_log_joint)


# During the possible change-point evaluation, the model are updated during
# min_seg time steps with dumb values (same value before the change-point)
# So, after confirming a change-points, it is necessary to ajdust the model
# takin in consideration the real values 
def adjust_R(t):
    global log_R, new_log_joint_bcp, max_indices, new_log_joint, log_message       
    log_R = log_R_bcp
    new_log_joint = new_log_joint_bcp
    max_indices = max_indices_bcp
    log_message = log_message_bcp

    for j in range(-min_seg+1,0):
        xt = y[t+j-1]
        #new_log_joint = update_joint_prob(t+j, xt, max_indices)
        update_joint_prob(t+j, xt, max_indices)
        model.update_params(xt)
        log_message = new_log_joint[max_indices]

    #new_log_joint = update_joint_prob(t, x, max_indices)
    update_joint_prob(t, x, max_indices)

# Auxiliary variables initialization
T = len(y)
CP = []             # list of changepoints
M0 = []             # list of post-change mean
M0_unique = []
c = 0               # counter of consecutive deviations
lcp = 0             # last changepoint (need to subtract 1)
log_message = 0
max_indices = np.array([0])     # indices keep in memory 
xbc = np.nan

# Lower triangular matrix with run posteriors of each run lenght size
log_R = -np.inf*np.ones((T+1, T+1)) 
log_R[0, 0] = 0  # log 0 == 1
max_R    = np.empty(T+1)
max_R[0] = 1

new_log_joint = np.nan
log_R_bcp = np.nan
new_log_joint_bcp = np.nan
max_indices_bcp = np.nan
log_message_bcp = np.nan


# Main algorithm
startTime = time.time()
for t in range(1, T+1):
        
    # Observe new datum
    x = y[t-1]
    
    # Model estimatd mean
    M0.append(model.mu[-1])
    
    # Update the joint probablity, but before save the current model state
    # in order to use in case of a deviation
    max_indices_b = max_indices # save the max indices before changepoint
    log_message_b = log_message
    new_log_joint_b = new_log_joint
    update_joint_prob(t, x, max_indices)
    
    # Decide for a change-point
    r = np.exp(log_R[t]) 
    deviation = t>1 and r[t-lcp]<=p_thr
    if deviation:
        if c==0:
            pcp = t - np.argmax(log_R[t])-1 # possible change-point
            if verbose>=2: print(f't={t}: possible changepoint at {pcp}')
            
            # save the model state before the changepoint
            xbc = y[t-2].mean()
            log_R_bcp = log_R
            max_indices_bcp = max_indices_b
            log_message_bcp = log_message_b
            new_log_joint_bcp = new_log_joint_b
        c=c+1
        
        # Changepoint confirmation
        if c==min_seg:
            adjust_R(t)
            max_t0 = np.argmax(r)
            lcp=t-max_t0
            if check_previous_cp(pcp, CP, min_seg):
                if verbose>=2: print(f't={t}: changepoint t={pcp} already labeled')
            elif pcp>0:
                if verbose: print(f't={t}: changepoint at t={pcp}')
                CP.append(pcp)
                M0_unique.append(M0[-1])
                if len(y[t:(t+3)]) > 0:   
                    mu = y[pcp+1:(pcp+4)].mean()
                    model.mu[-1] = mu
                    M0_unique.append(M0[-1])
            deviation = False
            c=0

    else:
        deviation = False
        c=0
    
    max_R[t] = np.argmax(log_R[t, :])
    
    # Update the sufficient statistics and pass the message
    if not deviation:
        model.update_params(x)
        log_message = new_log_joint[max_indices]
    
    else:
        update_joint_prob(t, xbc, max_indices_b)
        model.update_params(xbc)
        log_message = new_log_joint[max_indices]

    
    
endTime = time.time()
elapsedTime = endTime-startTime
R = np.exp(log_R)


fig,ax = plt.subplots(nrows=3, ncols=2, constrained_layout=True, 
                      figsize=(5,4), sharex=True)

for j in range(0,3):
    ax[j,0].xaxis.set_tick_params(labelbottom=True)
    ax[j,0].tick_params(axis='both', labelsize=6)
    ax[j,0].set_xticks(np.arange(0,1600,200))


# PROPOSED
for j in range(0,3):
    ax[j,1].xaxis.set_tick_params(labelbottom=True)
    ax[j,1].tick_params(axis='x', labelsize=6)
    ax[j,1].get_yaxis().set_visible(False)
    ax[j,1].set_xticks(np.arange(0,1600,200))


# CHANGEPOINTS
ax[0,0].set_title('BOCD - Basic')
ax[0,0].set_ylabel('ms', fontsize=6)
ax[0,0].plot(y, linewidth=0.5)
if len(CP0)>0:
    for i,cp in enumerate(CP0):
        if i==0:
            ax[0,0].axvline(cp, c='red', linewidth=0.5, label='changepoint')
        else:
            ax[0,0].axvline(cp, c='red', linewidth=0.5)
ax[0,0].legend(fontsize=6)

ax[0,1].set_title('BOCD - Proposed')
ax[0,1].plot(y, linewidth=0.5)
if len(CP)>0:
    for i,cp in enumerate(CP):
        if i==0:
            ax[0,1].axvline(cp, c='red', linewidth=0.5, label='change-point')
        else:
            ax[0,1].axvline(cp, c='red', linewidth=0.5)


# RUN LENGTH
ax[1,0].set_title('run length posterior prob.')
ax[1,0].set_ylabel('r(t)', fontsize=6)
im = ax[1,0].imshow(np.rot90(R0), aspect='auto', cmap='Blues', extent=[0,T,0,T], 
                  norm=LogNorm(vmin=0.0001, vmax=1))
_,yf1 = ax[1,0].get_ylim()
yf = np.round(np.argmax(R,axis=1).max())+yf1/10
ax[1,0].set_ylim(top=yf)


ax[1,1].set_title('run length posterior prob.')
im = ax[1,1].imshow(np.rot90(R), aspect='auto', cmap='Blues', extent=[0,T,0,T], 
                  norm=LogNorm(vmin=0.0001, vmax=1))
_,yf1 = ax[1,1].get_ylim()
yf2 = np.round(np.argmax(R,axis=1).max())+yf1/10
ax[1,1].set_ylim(top=yf2)
cbar = fig.colorbar(im, ax= ax[1,1], pad=0.01)
cbar.ax.tick_params(labelsize=6)


# MAP
ax[2,0].set_ylabel('r(t)', fontsize=6)
im = ax[2,0].imshow(np.rot90(R0), aspect='auto', cmap='Blues', extent=[0,T,0,T], 
                  norm=LogNorm(vmin=0.0001, vmax=1))
ax[2,0].set_ylim(top=yf)
ax[2,0].plot(max_R0[1:], color='r', label='MAP', linewidth=0.5)
ax[2,0].legend(fontsize=6)


im = ax[2,1].imshow(np.rot90(R), aspect='auto', cmap='Blues', extent=[0,T,0,T], 
                  norm=LogNorm(vmin=0.0001, vmax=1))
ax[2,1].set_ylim(top=yf2)
ax[2,1].plot(max_R[1:], color='r', label='MAP', linewidth=0.5)

fig.supxlabel('sample (t)', fontsize=6)
#plt.savefig('bocd_basic_proposed1.png', format='png', dpi=600)