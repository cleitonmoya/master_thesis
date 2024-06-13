# -*- coding: utf-8 -*-
"""
Change-point experiment - Grid search in Shao Dataset
Search for the best hyperparameters set for each proposed method
Do not consider the basic methods

Requirements (inside the function):
----------
- time series csv files in '../Dataset/shao/'
- txt file listing the change points for each time series in '../Dataset/shao/'
- hyperparameters settings
- List of methods and hyperparameters to search ('methods_list') setted by 
  the 'experiment' string
- experiment (string): options: 'sequential', 'vwcd', 'bocd', 'rrcf', 'all'

Result
------
- dataframe 'shao_results/df_shaogrid_{experiment}.pkl' with results for each
  pair method-hyperparameters
- dataframe 'results_shao/df_shaobest_{m.__name__}.pkl' with results for 
  for each method m and its best hyperparameters in terms of F1 score

@author: Cleiton Moya de Almeida
"""

import os
import numpy as np
import pandas as pd
from shao_benchmark import evaluation_window, f1_score
import changepoint_module as cm
from itertools import product
  
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

verbose = 0         # options: 0 (less verbose), 1, 2 (more verbose)

# Read the files names
path = '../Dataset/shao/'
files = [f for f in os.listdir(path) if f[-3:]=='csv']
N = len(files)

# List of methods and hyperparameters to search
methods_list = [
        {'method': cm.shewhart_ps, 'hyper': {'k':[1,2,3,4]}},
        
        {'method': cm.ewma_ps, 'hyper':{'lamb':[0.1, 0.2, 0.5], 
                                        'kd':[3, 4, 5]}},
        
        {'method': cm.cusum_2s_ps, 'hyper': {'h':[4, 5, 6], 
                                             'delta':[1, 2 ,3]}},
        
        {'method': cm.cusum_wl_ps, 'hyper': {'h':[4, 5, 6], 
                                             'w1':[5, 10]}},
        
        {'method': cm.vwcd, 'hyper': {'aggreg':['"mean"', '"posterior"'],
                                      'p_thr':[0.6, 0.8], 
                                      'vote_n_thr':[0.5, 0.7],
                                      'ab':[1, 5]}},
        
        {'method': cm.bocd_ps, 'hyper': {'lamb_bocd':[1e10, 1e20], 
                                         'kappa0':[0.01, 0.1, 0.5], 
                                         'alpha0':[0.01, 0.05, 0.1],
                                         'omega0':[0.1, 0.5, 1.0]}},
                                         
        {'method':cm.rrcf_ps, 'hyper':{'num_trees':[40],
                                       'tree_size':[75, 100, 200, 256],
                                       'thr_rrcf':[20, 25, 30, 35, 40]}}]


# Auxiliary list
sequential_ps = [cm.shewhart_ps, cm.ewma_ps, cm.cusum_2s_ps, cm.cusum_wl_ps]

methods_list = [methods_list[-1]]
for met in methods_list:

    Grid = [] # auxiliar list ot build a dataframe with the grid search results
    Dist = [] # list of dataframes with the distribution of results
    
    hyperparams = met['hyper']
    m = met['method']
    print(f'\nSearching for {m.__name__}')
    
    hyperparams_name = [k for k in hyperparams]
    params_v = list(product(*list(hyperparams.values())))
    
    for vec in params_v:
    
        # dynamic var attribution
        for j,hp in enumerate(hyperparams):
            exec(f'{hp}={vec[j]}')
        
        Res = []
        for n,file in enumerate(files):
        
            # Load the file
            y = np.loadtxt(f'{path}{file}', usecols=1, delimiter=';', skiprows=1)
            y[y<0]=0 # pre-processing (same procedure of Shao)
            CP_label = np.loadtxt(f'{path}{file[:-4]}.txt').tolist()
        
            if verbose == 1:
                if n > 0 and n%10==0:
                    print(f'processed {n}/{N} files')
            if verbose == 2: 
                    print(f'processing file {file}')
            
            # Maps the kargs for each method                
            elif m.__name__ == 'shewhart_ps':
                kargs = {'y':y, 'w':w0, 'k':k, 'rl':rl, 'ka':ka, 
                        'alpha_norm':alpha_norm, 'alpha_stat':alpha_stat, 
                        'filt_per':filt_per, 'max_var':max_var, 
                        'cs_max':cs_max}
            
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
            
            elif m.__name__ == 'bocd_ps':
                kargs = {'y':y, 'w':w_bocd, 'lamb':lamb_bocd, 'kappa0':kappa0, 'alpha0':alpha0, 
                         'omega0':omega0, 'p_thr':p_thr_rl, 'K':K, 'min_seg':min_seg}
            
            elif m.__name__ == 'rrcf_ps':
                kargs = {'y':y, 'num_trees':num_trees, 'shingle_size':shingle_size,
                         'tree_size':tree_size, 'thr':thr_rrcf, 'rl': rl_rrcf}
            
            # Call the methods
            num_anom_u = num_anom_l = M0 = S0 = None
            out = m(**kargs)
            if  m == cm.rrcf_ps:
                CP_pred, elapsed_time= out
            elif m in sequential_ps:
                CP_pred, Anom_u, Anom_l, M0, S0, elapsed_time = out
                num_anom_u = len(Anom_u)
                num_anom_l = len(Anom_l)
            elif m == cm.vwcd or m == cm.bocd_ps:
                CP_pred, M0, S0, elapsed_time = out
            
            # Evaluate the result
            metrics = evaluation_window(CP_label, CP_pred, window=we)
            if metrics['precision'] is None:
               metrics['precision'] = 0 
            if metrics['recall'] is None:
               metrics['recall'] = 0 
            
            # Store the results
            res = {'serie':file[:-3], 'CP_label':CP_label,
                   'method':m.__name__, 'CP_pred':CP_pred, 'num_cp_pred':len(CP_pred),
                   'num_anom_u':num_anom_u, 'num_anom_l':num_anom_l,
                   'precision':metrics['precision'],
                   'recall':metrics['recall'],
                   'f1':f1_score(metrics['precision'], metrics['recall']),
                   'M0':M0, 'S0':S0, 'elapsed_time':elapsed_time} 
            Res.append(res)
                
        
        # Dataframe with results
        df_res = pd.DataFrame(Res)
        Dist.append(df_res)
        
        prec = np.round(np.median(df_res.precision),3)
        recall = np.round(np.median(df_res.recall),3)
        f1 = np.round(np.median(df_res.f1),3)
        
        item = dict(zip(hyperparams_name, vec))
        
        grid_res = {'serie': file[:-3], 
                     'method': m.__name__, 
                     'hyper': item,
                     'CP_label': CP_label,
                     'CP_pred': CP_pred,
                     'num_cp_pred': len(CP_pred),
                     'num_anom_u': num_anom_u,
                     'num_anom_l': num_anom_l,
                     'n': N,
                     'tp': metrics['tp'],
                     'fp': metrics['fp'],
                     'fn': metrics['fn'],
                     'tn': N - metrics['tp'] - metrics['fp'] - metrics['fn'],
                     'precision': prec,
                     'recall': recall,
                     'f1': f1,
                     'M0': M0, 
                     'S0': S0,
                     'elapsed_time': elapsed_time}
        
        Grid.append(grid_res)     
        
        result= f'{item}: precision = {prec}, recall={recall}, f1={f1}'
        print(result)
    
    df_grid = pd.DataFrame(Grid)
    pd.to_pickle(df_grid, f'results_shao/df_shaogrid_{m.__name__}.pkl')

    j_max = df_grid['f1'].idxmax()
    print('Search result - best F1:')
    print(f'hyper: {df_grid.iloc[j_max].hyper}, F1: {df_grid.iloc[j_max].f1}')
    df_shao = pd.DataFrame(Dist[j_max])
    pd.to_pickle(df_shao, f'results_shao/df_shaobest_{m.__name__}.pkl')