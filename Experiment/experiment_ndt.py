# -*- coding: utf-8 -*-
"""
Change-point experiment - NDT Dataset
Apply each method of changepoint_module to the time series of the NDT dataset

Requirements (inside the function):
----------
- dataframe '../Dataset/df_series.pkl' with NDT time series information
- time series txt files in '../Dataset/ndt/'
- hyperparameters settings
- list of methods to test

Result
------
- dataframe 'results_ndt/df_ndt_{m.__name__}.pkl' for each method m

@author: Cleiton Moya de Almeida
"""

import numpy as np
import pandas as pd
import changepoint_module as cm

# read the dataframe with the time series information
df = pd.read_pickle('../Dataset/df_series_ndt.pkl')
N = len(df)

series_type = ['d_throughput', 'd_rttmean', 'u_throughput', 'u_rttmean']

# Sequential methpods - Basic implementations
sequential_ba = [cm.shewhart_ba, cm.ewma_ba, cm.cusum_2s_ba, cm.cusum_wl_ba]

# Sequential methods - Proposed implementations
sequential_ps = [cm.shewhart_ps, cm.ewma_ps, cm.cusum_2s_ps, cm.cusum_wl_ps]
pairs = zip(sequential_ba, sequential_ps)

# list if the methods
methods = [m for pair in pairs for m in pair]
methods = methods + [cm.vwcd, cm.bocd_ba, cm.bocd_ps, cm.rrcf_ps, cm.pelt_np]


# Methods hyperparameters
# Sequential methods
w0 = 10             # phase 1 estimating window size
rl = 4              # consecutive deviations to consider a changepoint
ka = 5              # kappa for anomaly
alpha_norm = 0.01   # normality test significace level
alpha_stat = 0.01   # statinarity test significance level
cs_max = 4          # maximum counter for process not stabilized
filt_per = 0.95     # outlier filtering percentil (first window or not. estab.)
max_var = 1.2       # maximum level of variance increasing to consider

# Shewhart
k = 3               # number of standard deviations to consider a deviation

# EWMA
lamb = 0.1          # EWMA 'lambda' hyperparameter
kd = 4              # EWMA 'kd' hyperparameter

# CUSUM
h = 5               # statistic threshold (in terms of sigma0)
delta = 2           # 2S-CUSUM hyperp. - deviation (in terms of sigma0) to detect
w1 = 10             # WL-CUSUM hyperp. - post-change estimating window size

# VWCD
wv = 20             # window-size
ab = 1              # Beta-binomial alpha and beta hyperp - prior dist. window
p_thr = 0.8         # threshold probability to an window decide for a changepoint
vote_p_thr = 0.9    # threshold probabilty to decide for a changepoint after aggregation
vote_n_thr = 0.5    # min. number of votes to decide for a changepoint
y0 = 0.5            # Logistic prior hyperparameter
yw = 0.9            # Logistic prior hyperparameter
aggreg = 'mean'     # aggregation function

# Non-Parametric PELT
pen = "MBIC"        # linear penality for the number of changepoints
min_seg_len = 4     # minimum segment size

# BOCD
lamb_bocd = 1e4     # lambda hyperparameter - prior for run length
kappa0 = 0.01       # Normal-inverse gamma prior hyperparameter
alpha0 = 0.01       # Normal-inverse gamma prior hyperparameter
omega0 = 0.1        # Normal-inverse gamma prior hyperparameter
w_bocd = 10         # Window-size to estimate the mean prior
K = 50              # run lenght extension cut
p_thr_rl = 0.05     # prob. threshold for online change-point decision
min_seg = 4         # vicinity for online change-point deciion

# RRCF
num_trees = 40      # number of trees
shingle_size = 2    # window length (subsequences size)
tree_size = 100     # maximum size of the trees
thr_rrcf = 20       # threshold for anomaly/chnage-point decision
rl_rrcf = 4         # number of consecutive deviations to consider a change-point


for m in methods:
    
    print(f"\nprocessing method {m.__name__}")
    
    Res = []    # auxiliar list to store the results    
    for n in range(N):
        
        client = df.iloc[n].client
        site = df.iloc[n].site
            
        # Prefixo do arquivo
        prefixo = client + "_" + site + "_"
        
        print(f"processing client-site {n+1}/{N}")
         
        for s_type in series_type:
            
            # Load the timeseries
            file = prefixo + s_type + ".txt"
            
            y = np.loadtxt(f'../Dataset/ndt/{file}', usecols=1, delimiter=',')
            
            # Remove possible nan values
            y = y[~np.isnan(y)]
           
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
                kargs = {'X':y, 'w':wv, 'w0':wv, 'ab':ab, 
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
            
            # Call the methods
            num_anom_u = num_anom_l = M0 = S0 = None
            out = m(**kargs)
            if (m in sequential_ba or 
                m == cm.pelt_np or
                m == cm.bocd_ba or
                m == cm.rrcf_ps):
                CP, elapsed_time = out
            elif m in sequential_ps:
                CP, Anom_u, Anom_l, M0, S0, elapsed_time = out
                num_anom_u = len(Anom_u)
                num_anom_l = len(Anom_l)
            elif m == cm.vwcd or m == cm.bocd_ps:
                CP, M0, S0, elapsed_time = out
            
            # Store the results
            res = {'client':client, 'site':site, 'serie':s_type,
                   'method':m.__name__, 'CP':CP, 'num_cp':len(CP), 
                   'num_anom_u':num_anom_u, 'num_anom_l':num_anom_l,
                   'M0':M0, 'S0':S0, 'elapsed_time':elapsed_time} 

            Res.append(res)

    # Dataframe with results
    df_res = pd.DataFrame(Res)
    pd.to_pickle(df_res, f'results_ndt/df_ndt_{m.__name__}.pkl')