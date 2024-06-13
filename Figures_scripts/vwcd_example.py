# -*- coding: utf-8 -*-
"""
Voting Windows Changepoint Detection example
@author: Cleiton Moya de Almeida
"""
import numpy as np
from scipy.stats import betabinom
import matplotlib.pyplot as plt
import time

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})

# Load the timeseries
file = 'dca6326b9c99_gig01_d_throughput.txt'
X = np.loadtxt(f'../Dataset/ndt/{file}', usecols=1, delimiter=',')

verbose = False

# Hyperparameters
w = 20              # window size
w0 = 20             # window used to estimate the post-change parameters
alpha = 1           # Beta-binomial hyperp - prior dist. window
beta = 1            # Beta-binomial hyperp - prior dist. window
p_thr = 0.6         # threshold probability to an window decide for a changepoint
pa_thr = 0.9        # threshold probabilty to decide for a changepoint
vote_n_thr = 10     # min. number of votes to decide for a changepoint
y0 = 0.5            # logistic prior hyperparameter
yw = 0.9            # logistic prior hyperparameter
aggreg = 'mean'

# Auxiliary functions
# Compute the window posterior probability given the log-likelihood and prior
# using the log-sum-exp trick
def pos_fun(ll, prior, tau):
    c = np.nanmax(ll)
    lse = c + np.log(np.nansum(prior*np.exp(ll - c)))
    p = ll[tau] + np.log(prior[tau]) - lse
    return np.exp(p)

# Aggregate a list of votes - compute the posterior probability
def votes_pos(vote_list, prior_v):
    vote_list = np.array(vote_list)
    prod1 = vote_list.prod()*prior_v
    prod2 = (1-vote_list).prod()*(1-prior_v)
    p = prod1/(prod1+prod2)
    return p

# Prior probabily for votes aggregation
def logistic_prior(x, w, y0, yw):
    a = np.log((1-y0)/y0)
    b = np.log((1-yw)/yw)
    k = (a-b)/w
    x0 = a/k
    y = 1./(1+np.exp(-k*(x-x0)))
    return y

# Compute the log-likelihood value for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def loglik(x,loc,scale):
    n = len(x)
    c = 1/np.sqrt(2*np.pi)
    y = n*np.log(c/scale) -(1/(2*scale**2))*((x-loc)**2).sum()
    return y

# Auxiliary variables
N = len(X)

# Prior probatilty for a changepoint in a window - Beta-Binomial
i_ = np.arange(0,w-3)
prior_w = betabinom(n=w-4,a=alpha,b=beta).pmf(i_)

# prior for votes aggregation
x_votes = np.arange(1,w+1)
prior_v = logistic_prior(x_votes, w, y0, yw) 

votes = {i:[] for i in range(N)} # dictionary of votes 
votes_agg = {}  # aggregated voteylims

lcp = 0 # last changepoint
CP = [] # changepoint list
M0 = [] # list of post-change mean
S0 = [] # list of post-change standard deviation
N_votes_tot = np.zeros(N)
N_votes_ele = np.zeros(N)

startTime = time.time()
for n in range(N):
    if n>=w-1:
        
        # estimate the paramaters (w0 window)
        if n == lcp+w0:
            # estimate the post-change mean and variace
            m_w0 = X[n-w0+1:n+1].mean()
            s_w0 = X[n-w0+1:n+1].std(ddof=1)
            M0.append(m_w0)
            S0.append(s_w0)
        
        # current window
        Xw = X[n-w+1:n+1]
        
        LLR_h = []
        for nu in range(1,w-3+1):
            # MLE and log-likelihood for H1
            x1 = Xw[:nu+1] #Xw até nu
            m1 = x1.mean()
            s1 = x1.std(ddof=1)
            if np.round(s1,3) == 0:
                s1 = 0.001
            logL1 = loglik(x1, loc=m1, scale=s1)
            
            # MLE and log-likelihood  for H2
            x2 = Xw[nu+1:]
            m2 = x2.mean()
            s2 = x2.std(ddof=1)
            if np.round(s2,3) == 0:
                s2 = 0.001
            logL2 = loglik(x2, loc=m2, scale=s2)

            # log-likelihood ratio
            llr = logL1+logL2
            LLR_h.append(llr)

        
        # Compute the posterior probability
        LLR_h = np.array(LLR_h)
        pos = [pos_fun(LLR_h, prior_w, nu) for nu in range(w-3)]
        pos = [np.nan] + pos + [np.nan]*2
        pos = np.array(pos)
        
        # Compute the MAP (vote)
        p_vote_h = np.nanmax(pos)
        nu_map_h = np.nanargmax(pos)
        
        # Store the vote
        j = n-w+1+nu_map_h # Adjusted index 
        votes[j].append(p_vote_h)
        
        # Aggregate the votes for X[n-w+1]
        votes_list = votes[n-w+1]
        elegible_votes = [v for v in votes_list if v > p_thr]
        num_votes_tot = len(votes_list)         # number of total votes
        num_votes_ele = len(elegible_votes)     # number of elegible votes
        N_votes_tot[n-w+1] = num_votes_tot
        N_votes_ele[n-w+1] = num_votes_ele
        
        # Decide for a changepoit
        if num_votes_ele >= vote_n_thr:
            if aggreg == 'posterior':
                agg_vote = votes_pos(elegible_votes, prior_v[num_votes_ele-1])
            elif aggreg == 'mean':
                agg_vote = np.mean(elegible_votes)
            votes_agg[n-w+1] = agg_vote
            if agg_vote >= pa_thr:
                if verbose: print(f'n={n}: Changepoint at n={n-w+1}, p={agg_vote:.2f}, num. votes={num_votes_ele}/{num_votes_tot}')
                lcp = n-w+1 # last changepoint
                CP.append(lcp)

endTime = time.time()
elapsedTime = endTime-startTime

if verbose: print(f'\nTotal: {len(CP)} changepoints')
if verbose: print(f'Elapsed time: {elapsedTime:.3f}s')


fig,ax=plt.subplots(figsize=(4.5,4), nrows=4, sharex=True, layout='constrained')
for ax_ in ax:
    ax_.tick_params(axis='both', labelsize=6)
    ax_.grid(linestyle=':')

ax[0].set_title('Voting window change-point detection')
ax[0].plot(X, linewidth=0.5)
if len(CP)>0:
    for j,cp in enumerate(CP):
        if j==0: 
            ax[0].axvline(cp, color='r', linewidth=0.5, label='change-point')
        else:
            ax[0].axvline(cp, color='r', linewidth=0.5)
    ax[0].legend(loc='lower right')
ax[0].set_ylabel('Mbits/s', fontsize=6)
ax[0].set_yticks([400,500,600])

x1, x2, y1, y2 = 55, 75, 520, 550 # subregion of the original image
axins = ax[0].inset_axes([0.15, 0.1, 0.2, 0.4],
    xlim=(x1, x2), ylim=(y1, y2), yticklabels=[])
axins.plot(range(55,76), X[55:76], linewidth=0.5, marker='o', markersize=1)
axins.axvline(66, linewidth=0.5, color='r', linestyle='--')

axins.set_yticks([530, 540])
axins.set_yticklabels([530, 550])
axins.yaxis.tick_right()
axins.grid(linestyle=':')

axins.tick_params(bottom=False) 
axins.set_xticklabels('') 
axins.tick_params(labelsize=6) 
_ = ax[0].indicate_inset_zoom(axins, edgecolor="black")


ax[1].set_title('Total number of votes')
N_votes_tot_idx = np.where(N_votes_tot)[0]
N_votes_tot_nonzero = N_votes_tot[N_votes_tot_idx] 
markerline, stemline, baseline = ax[1].stem(N_votes_tot_idx, N_votes_tot_nonzero)
plt.setp(markerline, markersize = 2)
plt.setp(baseline, linewidth=0)
plt.setp(stemline, linewidth=0.5)
ax[1].set_yticks([0,5,10,15,20])
ax[1].set_ylim([0,20])

ax[2].set_title('Number of votes above the threshold prob.')
N_votes_ele_idx = np.where(N_votes_ele)[0]
N_votes_ele_nonzero = N_votes_ele[N_votes_ele_idx] 
markerline, stemline, baseline = ax[2].stem(N_votes_ele_idx, N_votes_ele_nonzero)
plt.setp(markerline, markersize = 2)
plt.setp(baseline, linewidth=0)
plt.setp(stemline, linewidth=0.5)
ax[2].axhline(vote_n_thr, color='red', linewidth=0.5, label='threshold')
ax[2].legend(loc='lower right')
ax[2].set_yticks([0,5,10,15,20])
ax[2].set_ylim([0,20])

ax[3].set_title('Votes aggregation (change-point probability)')
if len(CP)>0:
    markerline, stemline, baseline = ax[3].stem(list(votes_agg), votes_agg.values())
    plt.setp(markerline, markersize = 2)
    plt.setp(baseline, linewidth=0)
    plt.setp(stemline, linewidth=0.5)
    ax[3].axhline(pa_thr, color='red', linewidth=0.5, label='threshold')
    ax[3].legend(loc='lower right')
ax[3].set_yticks([0,0.25,0.5,0.75,1])
ax[3].set_ylim(bottom=0)
ax[3].set_xlabel('sample (t)', fontsize=6)
ax[3].set_xticks(np.arange(0,275,25))
ax[3].set_xlim([0,250])