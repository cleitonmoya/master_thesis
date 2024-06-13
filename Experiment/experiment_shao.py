# -*- coding: utf-8 -*-
"""
Change-point experiment - Shao Dataset
Apply each method of changepoint_module to the time series of the Shao dataset

Requirements (inside the function):
----------
- time series csv files in '../Dataset/shao/'
- txt file listing the change points for each time series in '../Dataset/shao/'
- hyperparameters settings
- list of methods to test ('methods')

Result
------
- dataframe 'results_shao/df_shao_{m.__name__}.pkl' for each method m

@author: Cleiton Moya de Almeida
"""

import os
import numpy as np
import pandas as pd
from shao_benchmark import evaluation_window, f1_score
import changepoint_module as cm
import warnings

warnings.filterwarnings("error")  
  
# Commom hyper-parameters (all methods)
w0 = 10             # phase 1 estimating window size
rl = 4              # consecutive deviations to consider a changepoint
ka = 5              # kappa for anomaly
alpha_norm = 0.01   # normality test significace level
alpha_stat = 0.01   # statinarity test significance level
cs_max = 4          # maximum counter for process not stabilized
filt_per = 0.95     # outlier filtering percentil (first window or not. estab.)
max_var = 1.2       # maximum level of variance increasing to consider
we = 5              # window tolerance for evaluation

# Shewhart hyper-parameter
k = 4               # number of standard deviations to consider a deviation

# EWMA
lamb = 0.5          # EWMA 'lambda' hyperparameter
kd = 4              # EWMA 'kd' hyperparameter

# CUSUM
h = 6               # statistic threshold (in terms of sigma0)
delta = 3           # 2S-CUSUM hyperp. - deviation (in terms of sigma0) to detect
w1 = 5              # WL-CUSUM hyperp. - post-change estimating window size

# VWCD
wv = 20             # window-size
ab = 1              # Beta-binomial alpha and beta hyperp - prior dist. window
p_thr = 0.6         # threshold probability to an window decide for a changepoint
vote_p_thr = 0.9    # threshold probabilty to decide for a changepoint after aggregation
vote_n_thr = 0.7    # min. number of votes to decide for a changepoint (%)
y0 = 0.5            # Logistic prior hyperparameter
yw = 0.9            # Logistic prior hyperparameter
aggreg = 'mean'     # Aggregation function for the votes

# Non-Parametric PELT
pen = "MBIC"        # linear penality for the number of changepoints
min_seg_len = 4     # minimum segment size

# BOCD
lamb_bocd = 1e10    # lambda hyperparameter - prior for run length
kappa0 = 0.5        # Normal-inverse gamma prior hyperparameter
alpha0 = 0.01       # Normal-inverse gamma prior hyperparameter
omega0 = 1          # Normal-inverse gamma prior hyperparameter
w_bocd = 10         # Window-size to estimate the mean prior
K = 50              # run lenght extension cut
p_thr_rl = 0.05     # prob. threshold for online change-point decision
min_seg = 4         # vicinity for online change-point deciion

# RRCF
num_trees = 40      # number of trees
shingle_size = 2    # window length (subsequences size)
tree_size = 200     # maximum size of the trees
thr_rrcf = 20       # threshold for anomaly/chnage-point decision
rl_rrcf = 4         # number of consecutive deviations to consider a change-point

verbose = 1

# Read the files names
path = '../Dataset/shao/'
files = [f for f in os.listdir(path) if f[-3:]=='csv']
N_files = len(files)

sequential_ba = [cm.shewhart_ba, cm.ewma_ba, cm.cusum_2s_ba, cm.cusum_wl_ba]
sequential_ps = [cm.shewhart_ps, cm.ewma_ps, cm.cusum_2s_ps, cm.cusum_wl_ps]

# List the methods to apply
methods = [cm.shewhart_ba, cm.shewhart_ps, 
           cm.ewma_ba, cm.ewma_ps,
           cm.cusum_2s_ba, cm.cusum_2s_ps,
           cm.cusum_wl_ba, cm.cusum_wl_ps,
           cm.vwcd,
           cm.bocd_ba, cm.bocd_ps,
           cm.rrcf_ps, cm.pelt_np]

methods = [cm.ewma_ba, cm.ewma_ps]

for m in methods:

    print(f'\nExecuting {m.__name__}')
    Res = [] # list of dataframes with the distribution of results
    for n,file in enumerate(files):
    
        # Load the file
        y = np.loadtxt(f'{path}{file}', usecols=1, delimiter=';', skiprows=1)
        y[y<0]=0
        N = len(y)
        CP_label = np.loadtxt(f'{path}{file[:-4]}.txt').tolist()
        #label_aux = np.loadtxt(f'{path}{file}', usecols=2, delimiter=';', skiprows=1)
        #CP_label = (np.argwhere(label_aux==1).reshape(-1)).tolist()
    
        if verbose == 1: 
            if n > 0 and n%10==0:
                print(f'processing file {n+1}/{N_files}')
        if verbose == 2: 
                print(f'processing file {n+1}/{N_files}')
        
        # Maps the kargs for each method
        if m.__name__ == 'shewhart_ba':
            kargs = {'y':y, 'w':w0, 'k':k}
            
        elif m.__name__ == 'shewhart_ps' or m.__name__ == 'shewhart_ps2':
            kargs = {'y':y, 'w':w0, 'k':k, 'rl':rl, 'ka':ka, 
                    'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                    'filt_per':filt_per, 'max_var':max_var, 
                    'cs_max':cs_max}
        
        elif m.__name__ == 'ewma_ba':
            kargs = {'y':y, 'w':w0, 'kd':kd, 'lamb':lamb}
        
        elif m.__name__ == 'ewma_ps':
            kargs = {'y':y, 'w':w0, 'kd':kd, 'lamb':lamb, 'rl':rl, 'ka':ka, 
                    'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                    'filt_per':filt_per, 'max_var':max_var, 
                    'cs_max':cs_max}
        
        elif m.__name__ == 'cusum_2s_ba':
            kargs = {'y':y, 'w':w0, 'delta':delta, 'h':h}
        
        elif m.__name__ == 'cusum_2s_ps':
            kargs = {'y':y, 'w':w0, 'delta':delta, 'h':h, 
                     'rl':rl, 'k':k, 'ka':ka, 
                     'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                     'filt_per':filt_per, 'max_var':max_var, 
                     'cs_max':cs_max}
        
        elif m.__name__ == 'cusum_wl_ba':
            kargs = {'y':y, 'w0':w0, 'w1':w1, 'h':h}
        
        elif m.__name__ == 'cusum_wl_ps':
            kargs = {'y':y, 'w0':w0, 'w1':w1, 'h':h, 
                     'rl':rl, 'k':k, 'ka':ka, 
                     'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                     'filt_per':filt_per, 'max_var':max_var, 
                     'cs_max':cs_max}
            
        elif m.__name__ == 'vwcd':
            kargs = {'X':y, 'w':wv, 'w0':w0, 'ab':ab, 
                     'p_thr':p_thr, 'vote_p_thr':vote_p_thr, 
                     'vote_n_thr':vote_n_thr, 'y0':y0, 'yw':yw, 'aggreg':aggreg}
        
        elif m.__name__ == 'bocd_ba':
            kargs = {'y':y, 'w':w_bocd, 'lamb':lamb_bocd, 'kappa0':kappa0, 'alpha0':alpha0, 
                     'omega0':omega0, 'p_thr':p_thr_rl, 'K':K, 'min_seg':min_seg}
            
        elif m.__name__ == 'bocd_ps':
            kargs = {'y':y, 'w':w_bocd, 'lamb':lamb_bocd, 'kappa0':kappa0, 'alpha0':alpha0, 
                     'omega0':omega0, 'p_thr':p_thr_rl, 'K':K, 'min_seg':min_seg}
        
        elif m.__name__ == 'rrcf_ps':
            kargs = {'y':y, 'num_trees':num_trees, 'shingle_size':shingle_size,
                     'tree_size':tree_size, 'thr':thr_rrcf, 'rl': rl_rrcf}
        
        elif m.__name__ == 'pelt_np':
            kargs = {'y':y, 'pen':pen, 'minseglen':min_seg_len}
            
        else:
            print('Error: method not defined')

        # Call the methods
        num_anom_u = num_anom_l = M0 = S0 = None
        out = m(**kargs)
        if (m in sequential_ba or 
            m == cm.bocd_ba or
            m == cm.rrcf_ps or
            m == cm.pelt_np):
            CP_pred, elapsed_time= out
        elif m in sequential_ps:
            CP_pred, Anom_u, Anom_l, M0, S0, elapsed_time = out
            num_anom_u = len(Anom_u)
            num_anom_l = len(Anom_l)
        elif m == cm.vwcd or m == cm.bocd_ps:
            CP_pred, M0, S0, elapsed_time = out
        
        # Evaluate the result
        try:
            metrics = evaluation_window(CP_label, CP_pred, window=we)
        except RuntimeWarning:
            if verbose: print(f'Warning (low critical): {file}: Munkres overflow')
            continue
        if metrics['precision'] is None:
           metrics['precision'] = 0 
        if metrics['recall'] is None:
           metrics['recall'] = 0 
        
        # Store the results
        res = {'serie': file[:-3],
               'CP_label':CP_label,
               'method':m.__name__,
               'CP_pred':CP_pred,
               'num_cp_pred':len(CP_pred),
               'num_anom_u':num_anom_u,
               'num_anom_l':num_anom_l,
               'n': N,
               'tp': metrics['tp'],
               'fp': metrics['fp'],
               'fn': metrics['fn'],
               'tn': N - metrics['tp'] - metrics['fp'] - metrics['fn'],
               'precision':metrics['precision'],
               'recall':metrics['recall'],
               'f1': f1_score(metrics['precision'], metrics['recall']),
               'M0': M0, 
               'S0': S0,
               'elapsed_time': elapsed_time} 
        Res.append(res)
                
    # Dataframe with results
    df_res = pd.DataFrame(Res)
    pd.to_pickle(df_res, f'results_shao/df_shao_{m.__name__}.pkl')