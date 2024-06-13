# -*- coding: utf-8 -*-
"""
Online changepoint detection module

    - Basic implementation: suffix 'ba'
    - Proposed implementation: suffix 'ps'
    - Functions:
        Shewhart: shewhart_ba, shewhart_ps
        Exponential Weighted Moving Average: ewma_ba, ewma_ps
        Two-sided CUSUM: cusum_2s_ba, cusum_2s_ps
        Window-Limited CUSUM: cusum_wl_ba, cusum_wl_ps
        Voting Windows Changepoint Detection: vwcd
        Bayesian Online Changepoint Detection: bocd_ba, bocd_ps
        Robust Random Cut Forest: rrcf_ps
        Non-Parametric Pelt: pelt_np

@author: Cleiton Moya de Almeida
"""

import numpy as np
from scipy.stats import shapiro, betabinom
from scipy.special import logsumexp
from statsmodels.tsa.stattools import adfuller
from bocd import ConstantHazard, Gaussian, check_previous_cp
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
changepoint_np = importr('changepoint.np')
changepoint = importr('changepoint')
import rrcf
import time

verbose = False

# Shapiro-Wilk normality test
# H0: normal distribution
def normality_test(y, alpha):
    _, pvalue = shapiro(y)
    return pvalue > alpha


# Augmented Dickey-Fuller test for unitary root (non-stationarity)
# H0: the process has a unit root (non-stationary)
def stationarity_test(y, alpha):
    adf = adfuller(y)
    pvalue = adf[1]
    return pvalue < alpha


# Compute the log-pdf for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def logpdf(x,loc,scale):
    c = 1/np.sqrt(2*np.pi)
    y = np.log(c) - np.log(scale) - (1/2)*((x-loc)/scale)**2
    return y


# Compute the log-likelihood value for the normal distribution
# Obs.: the scipy built-in function logpdf does not use numpy and so is inneficient
def loglik(x,loc,scale):
    n = len(x)
    c = 1/np.sqrt(2*np.pi)
    y = n*np.log(c/scale) -(1/(2*scale**2))*((x-loc)**2).sum()
    return y


# Shewhart (X-chart) - Basic
def shewhart_ba(y, w, k):
    """
    Shewhart - basic implementation
   
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    k (int): number of standard deviations to consider a change-point 

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables
    CP = []
    lcp = 0

    startTime = time.time()
    for t, y_t in enumerate(y):

        if t >= lcp + w:
            
            if t==lcp+w:
                mu0 = y[lcp+1:t].mean()
                s0 = y[lcp+1:t].std()
                if verbose: print(f't={t}: mu0={mu0}, s0={s0}')
            
            # lower and upper control limits
            l = mu0 - k*s0
            u = mu0 + k*s0
            
            # Shewhart statistic deviation checking
            dev = y_t>=u or y_t<=l
            
            if dev:
               lcp = t
               if verbose: print(f't={t}: Changepoint at t={lcp-1}')
               CP.append(lcp-1)
               dev = False

    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, elapsedTime


# Shewhart (X-chart) - Proposed
def shewhart_ps(y, w, k, rl, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Shewhart - proposed implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    k (int): number of standard deviations to consider a deviation
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables initialization
    CP = []             # changepoint list 
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    lcp = 0             # last checked point
    win_t0 = 0          # learning window t0
    Win_period = []     # stabilization/learning windows
    c = 0               # statistic deviation counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0              # stabilization counter
    Mu0 = []            # phase 1 estimated mu0 at each t
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    Sigma0 = []         # phase 1 estimated sigma0
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    U = []              # upper control limit at each t
    L = []              # lower control limit at each t

    startTime = time.time()
    for t, y_t in enumerate(y):

        if (t >= lcp + w):
            
            # At process beginning and after a changepoint, 
            # check if the process is stable before estimating the parameters
            if t==lcp+w:
                yw = y[lcp:t]
                sw = yw.std(ddof=1)
                if np.round(sw,3) == 0:
                    sw = 0.001
                
                # Shapiro-Wiltker test for normality
                if not np.all(np.isclose(yw, yw[0])):
                    normality = normality_test(yw, alpha_norm)
                else:
                    normality = True
                
                # Check if the variance level increasing is acceptable
                # If its the first estimation, accept blindly, but filter possible outliers 
                # before estimating the mu0, s0
                first_est = len(S0_unique) == 0
                if not first_est:
                    sa = max(S0_unique)
                    dev_var = abs(sw - sa)/sa
                    var_acept =  dev_var <= max_var
                else:
                    var_acept = True
                    if verbose==2: print(f't={t}: Applying percentil filter 1')
                    yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                    if len(yw_f) != 0:
                        yw = yw_f
                
                # Stabilization criteria: normality and variance accepted
                stab = normality and var_acept
                
                # If process did not stabilize after cs_max, force the stabilization,
                # but filter possible outliers to estimate mu0, s0
                if stab or cs==cs_max:
                    if cs==cs_max:
                        if verbose: print(f"n={t}: Considering process stabilized")
                        if not first_est:
                            if verbose==2: print(f't={t}: Applying percentil filter 2')
                            yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                            if len(yw_f) != 0:
                                yw = yw_f
                    else:
                        if verbose==2: print(f"n={t}: Process stabilized, lcp={lcp}")
                    
                    # Pre-change parameters estimation
                    mu0 = yw.mean()
                    s0 = yw.std(ddof=1)
                    M0_unique.append(mu0)
                    S0_unique.append(s0)
                    if np.round(s0,3) == 0:
                        s0 = 0.001
                    Win_period.append((win_t0,t))
                    if verbose==2: print(f"n={t}: Estimated mu0={mu0:.2f}, sigma0={s0:.2f}")
                    
                    # Beyond the non-normality, if the last window was not stationary, 
                    # and now the process is normal and stationary, declare a changepoint
                    # +1: try skip to spurious anomalies before changepoint
                    if cs > 0 and cs < cs_max:
                        # check if the last windows was stationary
                        if not np.all(np.isclose(y[win_t0+1:lcp], y[win_t0+1:lcp][0])):
                            lw_stat = stationarity_test(y[win_t0+1:lcp], alpha_stat)
                        else:
                            lw_stat = True
                       
                        # check if process now is stationarity
                        if not np.all(np.isclose(y[lcp:t], y[lcp:t][0])):
                            cw_stat = stationarity_test(y[lcp:t], alpha_stat) 
                        else:
                            cw_stat = True        
                        
                        if verbose==2: print(f"n={t}: Last windows stat.: {lw_stat}; current window stat.: {cw_stat}; cs={cs}")
                        if not lw_stat and cw_stat:
                            if verbose: print(f"n={t}: Considering t={lcp-1} a changepoint")
                            CP.append(lcp-1)
                    cs = 0
                else:
                    if verbose==2: print(f"n={t}: Process not stabilized, normality={normality}, var_acept={var_acept}")
                    lcp=lcp+w
                    cs = cs+1

            # Lower and upper control limits for deviation
            u = mu0 + k*s0
            l = mu0 - k*s0
            
            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*s0
            anom_l = y_t <= mu0 - ka*s0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)
            
            # Check for statistic deviation
            dev = abs(y_t-mu0) >= k*s0
            
            if dev:
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1

                c = c+1
                
                if c==rl:
                    lcp = t-rl+1    # last check point
                    win_t0 = lcp    # learning window t0
                    if verbose: print(f'n={t}: Changepoint at t={lcp-1}')
                    CP.append(lcp-1)
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l]    
                    c = 0
                    ca_u = 0
                    ca_l = 0
            else:
                c = 0
                ca_u = 0
                ca_l = 0
            
        else:
            mu0 = np.nan
            s0 = np.nan
            l = np.nan
            u = np.nan
        
        Mu0.append(mu0)
        Sigma0.append(s0)
        U.append(u)
        L.append(l)
    endTime = time.time()
    elapsedTime = endTime-startTime

    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


# EWMA - Basic
def ewma_ba(y, w, kd, lamb):
    """
    Exponential Weighted Moving Average (EWMA) - basic implementation
   
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    kd (int): EWMA 'kd' hyperparameter
    lamb (float): EWMA 'lambda' hyperparameter

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables initialization
    CP = []
    lcp = 0

    startTime = time.time()
    for t,y_t in enumerate(y):

        if t >= lcp + w:
            
            # Phase 1 estimation
            if t == lcp+w:
                mu0 = np.mean(y[lcp+1:t])
                sigma0 = y[lcp+1:t].std(ddof=1)
                z = mu0 # reset the Z statistic
                if verbose: print(f't={t}: mu0={mu0}, sigma0={sigma0}')

            # Phase 2 statistic and limits estimation
            z = lamb*y[t] + (1-lamb)*z
            ucl = mu0 + kd*sigma0*np.sqrt((lamb/(2-lamb)))
            lcl = mu0 - kd*sigma0*np.sqrt((lamb/(2-lamb)))
            
            # verifica se há dev do moving range
            dev = z >ucl or z < lcl
            if dev:
                lcp = t
                if verbose: print(f't={t}: Changepoint at t={lcp-1}')
                CP.append(lcp-1)

    endTime = time.time()
    elapsedTime = endTime-startTime


    # Results
    if verbose: print(f'\nTotal: {len(CP)} changepoints')
    if verbose: print('Elapsed time: {:.3f}ms'.format(elapsedTime*1000))

    return CP, elapsedTime


# EWMA - Proposed
def ewma_ps(y, w, kd, lamb, rl, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Exponential Weighted Moving Average (EWMA) - proposed implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    kd (int): EWMA 'kd' hyperparameter
    lamb (float): EWMA 'lambda' hyperparameter
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables initialization
    Z = []              # EWMA statistic over time
    U = []              # EWMA upper control limit
    L = []              # EWMA lower control limit
    CP = []             # list of changepoints
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    Mu0 = []            # estimated mu0 over time
    Sigma0 = []         # estimated sigma0 over time
    Win_period = []     # list of windows (t0,tf) period
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    lcp = 0             # last check point (different from changepoint)
    win_t0 = 0          # learning window t0
    c = 0               # sucessive deviation counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0              # stabilization counter

    startTime = time.time()
    for t, y_t in enumerate(y):

        if t >= lcp+w:
            
            if t == lcp+w:

                yw = y[lcp:t]
                sw = yw.std(ddof=1)
                if np.round(sw,3) == 0:
                    sw = 0.001
                
                # Shapiro-Wiltker test for normality
                if not np.all(np.isclose(yw, yw[0])):
                    normality = normality_test(yw, alpha_norm)
                else:
                    normality = True

                # Check if the variance level increasing is acceptable
                # If its the first window, accept blindly, but filter possible
                # outliers before estimating the mu0, sigma0
                first_est = len(S0_unique) == 0
                if not first_est:
                    sa = max(S0_unique)
                    dev_var = abs(sw - sa)/sa
                    var_acept =  dev_var <= max_var
                else:
                    var_acept = True
                    if verbose==2: print(f't={t}: Applying percentil filter 1')
                    yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                    if len(yw_f) != 0:
                        yw = yw_f
                    
                # Stabilization criteria: normality and variance accepted
                stab = normality and var_acept

                # If process did not stabilize after cs_max, force the stabilization,
                # but filter possible outliers to estimate mu0, sigma0
                if stab or cs==cs_max:
                    if cs==cs_max:
                        yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                        if len(yw_f) != 0:
                            yw = yw_f
                        if verbose: print(f"n={t}: Considering process stabilized")
                    else:
                        if verbose: print(f"n={t}: Process stabilized")
     
                    # Phase 1 parameters estimation
                    mu0 = yw.mean()
                    sigma0 = yw.std(ddof=1)
                    if np.round(sigma0,3) == 0:
                        sigma0 = 0.001
                    M0_unique.append(mu0)
                    S0_unique.append(sigma0)
                    if np.round(sigma0,3) == 0:
                        sigma0 = 0.001
                    Win_period.append((win_t0,t))
                    z=mu0
                    if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")

                    # Beside the non-normality, if the last window was not stationary, 
                    # and now the process is normal and statonary, consider a changepoint
                    if cs > 0 and cs < cs_max:
                        # check if the last windows was stationary
                        if not np.all(np.isclose(y[win_t0+1:lcp], y[win_t0+1:lcp][0])):
                            lw_stat = stationarity_test(y[win_t0+1:lcp], alpha_stat)
                        else:
                            lw_stat = True
                       
                        # check if process now is stationarity
                        if not np.all(np.isclose(y[lcp:t], y[lcp:t][0])):
                            cw_stat = stationarity_test(y[lcp:t], alpha_stat) 
                        else:
                            cw_stat = True        
                        
                        if verbose==2: print(f"n={t}: Last windows stat.: {lw_stat}; current window stat.: {cw_stat}; cs={cs}")
                        if not lw_stat and cw_stat:
                            if verbose: print(f"n={t}: Considering t={lcp-1} a changepoint")
                            CP.append(lcp-1)
                    cs = 0
                else:
                    if verbose: print(f"n={t}: Process not stabilized, normal={normality}, var_acetp={var_acept}")
                    lcp=lcp+w
                    cs = cs+1        

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)

            # EWMA statistic update
            za = z
            z = lamb*y[t] + (1-lamb)*z
            ucl = mu0 + kd*sigma0*np.sqrt((lamb/(2-lamb)))
            lcl = mu0 - kd*sigma0*np.sqrt((lamb/(2-lamb)))

            # check for statistic deviation
            dev = z >ucl or z < lcl
            if dev:
                c=c+1
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                
                # confirms the changepoint and resets the ewma statistic
                if c == rl:
                    lcp = t-rl+1    # last check point
                    win_t0 = lcp    # learning window t0
                    if verbose: print(f'n={t}: Changepoint at t={lcp-1}')
                    CP.append(lcp-1)
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l] 
                    c = 0
                    ca_u = 0
                    ca_l = 0
                Z.append(z)
                z = za
            else:
                c=0
                ca_u = 0
                ca_l = 0
                Z.append(z)
        else:
            z = np.nan
            ucl = np.nan
            lcl = np.nan
            mu0 = np.nan
            sigma0 = np.nan
            Z.append(z)
            
        U.append(ucl)
        L.append(lcl)
        Mu0.append(mu0)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime

    Z = np.array(Z)
    Mu0 = np.array(Mu0)
    Sigma0 = np.array(Sigma0)
            
    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


# Two-Sided CUSUM - Basic
def cusum_2s_ba(y, w, delta, h):
    """
    Two-sided CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    delta (int/float): deviation (in terms of sigma0) to detect
    h (float): statistic threshold (in terms of sigma0)

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables
    CP = []
    Ut = 0
    Lt = 0
    lcp = 0

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w:
            if Ut is np.nan:
                Ut = 0
                Lt = 0
            
            # Phase 1 parameters updating
            if t==lcp+w:
                mu0 = y[lcp+1:t].mean()
                sigma0 = y[lcp+1:t].std(ddof=1)
                Ht = h*sigma0
            
            # Phase 2 CUSUM statitics computing
            Ut = Ut + y_t - mu0 - delta*sigma0/2
            Ut = np.heaviside(Ut,0)*Ut
            Lt = Lt - y_t + mu0 - delta*sigma0/2
            Lt = np.heaviside(Lt,0)*Lt

            # check for statistic deviation
            dev = Ut > Ht or Lt > Ht
            if dev:
                lcp = t    
                if verbose: print(f't={t}: Changepoint at t={lcp-1}')    
                CP.append(lcp-1)
                dev = False        
                
        else:
            Ut = np.nan
            Lt = np.nan

    endTime = time.time()
    elapsedTime = endTime-startTime

    # Results
    if verbose: print(f'\nTotal: {len(CP)} changepoints')
    if verbose: print('Elapsed time: {:.3f}ms'.format(elapsedTime*1000))

    return CP, elapsedTime


# Two-Sided CUSUM - Proposed
def cusum_2s_ps(y, w, delta, h, rl, k, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Two-sided CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    delta (int/float): deviation (in terms of sigma0) to detect
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables initialization
    Gu = []             # CUSUM upper statistic over time
    Gl = []             # CUSUM lower statistic over time
    H = []              # CUSUM statistic threshold over time
    CP = []             # list of changepoints
    Anom_u = []         # up point anomalies list
    Anom_l = []         # low point anomalie list
    Mu0 = []            # estimated mu0 over time
    Sigma0 = []         # esimated sigma0 over time
    Win_period = []     # list of windows (t0,tf) period
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    gu = 0              # CUSUM upper statistic
    gl = 0              # CUSUM lower statistic
    lcp = 0             # last check point (different from changepoint)
    win_t0 = 0          # learning window t0
    c = 0               # sucessive outlier counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0              # stabilization counter

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w:
            if gu is np.nan:
                gu = 0
                gl = 0
            
            if t==lcp+w:
                
                # At process beginning and after a changepoint, 
                # check if the process is stable before estimating the parameters
                if t==lcp+w:
                    yw = y[lcp:t]
                    sw = yw.std(ddof=1)
                    if np.round(sw,3) == 0:
                        sw = 0.001
                    
                    # Shapiro-Wiltker test for normality
                    if not np.all(np.isclose(yw, yw[0])):
                        normality = normality_test(yw, alpha_norm)
                    else:
                        normality = True
                    
                    # Check if the variance level increasing is acceptable
                    # If its the first window, accept blindly, but filter possible outliers 
                    # before estimating the mu0, s0
                    first_est = len(S0_unique) == 0
                    if not first_est:
                        sa = max(S0_unique)
                        dev_var = abs(sw - sa)/sa
                        var_acept =  dev_var <= max_var
                    else:
                        var_acept = True
                        if verbose==2: print(f't={t}: Applying percentil filter 1')
                        yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                        if len(yw_f) != 0:
                            yw = yw_f
                    
                    # Stabilization criteria: normality and variance accepted
                    stab = normality and var_acept
                    
                    # If process did not stabilize after cs_max, force the stabilization,
                    # but filter possible outliers to estimate mu0, s0
                    if stab or cs==cs_max:
                        if cs==cs_max:
                            yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                            if len(yw_f) != 0:
                                yw = yw_f
                            if verbose: print(f"n={t}: Considering process stabilized")
                        else:
                            if verbose: print(f"n={t}: Process stabilized")
                        
                        # Phase 1 parameters estimation
                        mu0 = yw.mean()
                        sigma0 = yw.std(ddof=1)
                        if np.round(sigma0,3) == 0:
                            sigma0 = 0.001
                        M0_unique.append(mu0)
                        S0_unique.append(sigma0)
                        Win_period.append((win_t0,t))
                        
                        if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")
                        
                        # Beyond the non-normality, if the last window was not stationary, 
                        # and now the process is normal and statonary, consider a changepoint
                        if cs > 0 and cs < cs_max:
                            # check if the last windows was stationary
                            if not np.all(np.isclose(y[win_t0+1:lcp], y[win_t0+1:lcp][0])):
                                lw_stat = stationarity_test(y[win_t0+1:lcp], alpha_stat)
                            else:
                                lw_stat = True
                           
                            # check if process now is stationarity
                            if not np.all(np.isclose(y[lcp:t], y[lcp:t][0])):
                                cw_stat = stationarity_test(y[lcp:t], alpha_stat) 
                            else:
                                cw_stat = True        
                            
                            if verbose==2: print(f"n={t}: Last windows stat.: {lw_stat}; current window stat.: {cw_stat}; cs={cs}")
                            if not lw_stat and cw_stat:
                                if verbose: print(f"n={t}: Considering t={lcp-1} a changepoint")
                                CP.append(lcp-1)
                        cs = 0
                    else:
                        if verbose: print(f"n={t}: Process not stabilized, sw={yw.std(ddof=1)}")
                        lcp=lcp+w
                        cs = cs+1
            
            # control limit for deviation
            ht = h*sigma0 
            
            # CUSUM statistics update
            gua = gu
            gla = gl
            gu = gu + y_t - mu0 - delta*sigma0/2
            gu = np.heaviside(gu,0)*gu
            gl = gl - y_t + mu0 - delta*sigma0/2
            gl = np.heaviside(gl,0)*gl

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)
            
            # check for statistic deviation
            dev = gu > ht or gl > ht
            if dev:
                c=c+1
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                    
                if c == rl: # confirma o changepoint e reinicia o cusum
                    lcp = t-rl+1    # last check point
                    win_t0 = lcp    # learning window t0
                    if verbose: print(f'n={t}: Changepoint at t={lcp-1}')
                    CP.append(lcp-1)
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l]    
                    c = 0
                    ca_u = 0
                    ca_l = 0
                    
                Gu.append(gu)
                Gl.append(gl)
                gu = gua
                gl = gla
                
            else:
                c=0
                ca_u = 0
                ca_l = 0
                Gu.append(gu)
                Gl.append(gl)

        else:
            gu = np.nan
            gl = np.nan
            ht = np.nan
            mu0 = np.nan
            sigma0 = np.nan
            Gu.append(gu)
            Gl.append(gl)
            
        H.append(ht)
        Mu0.append(mu0)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime

# Window Limited CUSUM - Basic
def cusum_wl_ba(y, w0, w1, h):
    """
    Window-limited CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w0 (int): pre-change estimating window size
    w1 (int): post-change estimating window size
    h (float): statistic threshold (in terms of sigma0)

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables
    lcp = 0
    CP = []
    St = 0

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w0:
            if St is np.nan:
                St = 0 
                
            # Phase 1 parameters learning
            if t == lcp+w0:
                m0 = y[lcp+1:t].mean()
                s0 = y[lcp+1:t].std(ddof=1)
                if np.round(s0,3) == 0:
                    s0 = 0.001
                Ht = h*s0
            
            # Phase 2 parameters earning
            m1 = y[t-w1:t].mean()
            s1 = y[t-w1:t].std(ddof=1)
            if np.round(s1,3) == 0:
                s1 = 0.001
            
            # Phase 2 CUSUM statistic computing
            St = np.heaviside(St,0)*St
            St = St + logpdf(y_t, m1, s1) - logpdf(y_t, m0, s0)
            
            # Check for statistic deviation
            dev = St > Ht
            if dev:
                lcp=t
                if verbose: print(f'Changepoint at t={lcp-1}')
                CP.append(lcp-1)
                dev = False
        else:
            St = np.nan

    endTime = time.time()
    elapsedTime = endTime-startTime
           
    return CP, elapsedTime
    
# Window Limited CUSUM - Proposed
def cusum_wl_ps(y, w0, w1, h, rl, k, ka, alpha_norm, alpha_stat, filt_per, max_var, cs_max):
    """
    Window-limited CUSUM - basic implementation
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    w0 (int): pre-change estimating window size
    w1 (int): post-change estimating window size
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    Anom_u (list): upper anomalies
    Anom_l (list): lower anomalies
    M0_unique (list): estimated mean of the segments
    S0_unique (list): estimated standar deviation of the segments
    elapsedTime (float): running-time (microseconds)
    """
    
    # Auxiliary variables initialization
    S = []              # CUSUM statistic over time
    H = []              # CUSUM statistic threshold over time
    CP = []             # list of changepoints
    Anom_u = []         # upper point anomalies list
    Anom_l = []         # lowwer point anomalie list
    Mu0 = []            # estimated mu0 over time
    Sigma0 = []         # estimated sigma0 over time
    Mu1 = []            # estimated mu1 over time
    Win_period = []     # list of windows (t0,tf) period
    M0_unique = []      # phase 1 estimated mu0 after each changepoint
    S0_unique = []      # phase 1 estimated sigma0 after each changepoint
    st = 0              # CUSUM statistic
    Sta = 0             # CUSUM statisitc before deviation
    lcp = 0             # last check point (different from changepoint)
    win_t0 = 0          # learning window t0
    c = 0               # sucessive outlier counter
    ca_u = 0            # up point up counter
    ca_l = 0            # low point anomaly counter
    cs = 0              # stabilization counter

    startTime = time.time()
    for t, y_t in enumerate(y):
        
        if t >= lcp+w0:
            
            #if not dev, update mu:
            if t==lcp+w0:
                
                yw = y[lcp:t]
                sw = yw.std(ddof=1)
                if np.round(sw,3) == 0:
                    sw = 0.001
                # Shapiro-Wiltker test for normality
                if not np.all(np.isclose(yw, yw[0])):
                    normality = normality_test(yw, alpha_norm)
                else:
                    normality = True
                
                # Check if the variance level increasing is acceptable
                # If its the first window, accept blindly, but filter possible outliers 
                first_est = len(S0_unique) == 0
                if not first_est:
                    sa = max(S0_unique)
                    dev_var = abs(sw - sa)/sa
                    var_acept =  dev_var <= max_var
                else:
                    var_acept = True
                    if verbose==2: print(f't={t}: Applying percentil filter 1')
                    yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                    if len(yw_f) != 0:
                        yw = yw_f
               
                
                # Stabilization criteria: normality and variance accepted
                stab = normality and var_acept
                
                # If process did not stabilize after cs_max, force the stabilization,
                # but filter possible outliers to estimate mu0, s0
                if stab or cs==cs_max:
                    if cs==cs_max:
                        yw_f = yw[(yw>np.quantile(yw,1-filt_per)) & (yw<np.quantile(yw,filt_per))]
                        if len(yw_f) != 0:
                            yw = yw_f
                        if verbose: print(f"n={t}: Considering process stabilized")
                    else:
                        if verbose: print(f"n={t}: Process stabilized")
                    
                    # Phase 1 parameters estimation
                    mu0 = yw.mean()
                    sigma0 = yw.std(ddof=1)
                    if np.round(sigma0,3) == 0:
                        sigma0 = 0.001
                    M0_unique.append(mu0)
                    S0_unique.append(sigma0)
                    Win_period.append((win_t0,t))
                    
                    if verbose: print(f"n={t}: Estimated mu0={mu0}, sigma0={sigma0}")
                    
                    # Beside the non-normality, if the last window was not stationary, 
                    # and now the process is normal and statonary, consider a changepoint
                    if cs > 0 and cs < cs_max:
                        # check if the last windows was stationary
                        if not np.all(np.isclose(y[win_t0+1:lcp], y[win_t0+1:lcp][0])):
                            lw_stat = stationarity_test(y[win_t0+1:lcp], alpha_stat)
                        else:
                            lw_stat = True
                       
                        # check if process now is stationarity
                        if not np.all(np.isclose(y[lcp:t], y[lcp:t][0])):
                            cw_stat = stationarity_test(y[lcp:t], alpha_stat) 
                        else:
                            cw_stat = True        
                        
                        if verbose==2: print(f"n={t}: Last windows stat.: {lw_stat}; current window stat.: {cw_stat}; cs={cs}")
                        if not lw_stat and cw_stat:
                            if verbose: print(f"n={t}: Considering t={lcp-1} a changepoint")
                            CP.append(lcp-1)
                    cs = 0
                else:
                    if verbose: print(f"n={t}: Process not stabilized, norm={normality}, var_acept={var_acept}, sw={yw.std(ddof=1)}")
                    lcp=lcp+w0
                    cs = cs+1

            # control limit for deviation
            ht = h*sigma0 

            # Phase 2 paramters estimation
            mu1 = y[t-w1:t].mean()
            sigma1 = y[t-w1:t].std(ddof=1)
            if np.round(sigma1,3) == 0:
                sigma1 = 0.001
            
            # CUSUM statistics update
            if st is np.nan:
                st = 0
            Sta = st
            st = np.heaviside(st,0)*st
            st = st + logpdf(y_t, mu1, sigma1) - logpdf(y_t, mu0, sigma0)

            # Check for point anomaly (upper and low)
            anom_u = y_t >= mu0 + ka*sigma0
            anom_l = y_t <= mu0 - ka*sigma0
            if anom_u:
                Anom_u.append(t)
            if anom_l:
                Anom_l.append(t)
            
            # check for statistic deviation
            dev = st > ht
            if dev:
                c=c+1              
                if anom_u:
                    ca_u = ca_u+1 
                elif anom_l:
                    ca_l = ca_l+1
                # confirms the changepoint and resets the cusum statistic
                if c == rl:
                    lcp = t-rl+1    # last check point
                    win_t0 = lcp    # learning window t0
                    if verbose: print(f'n={t}: Changepoint at t={lcp-1}')
                    CP.append(lcp-1)
                    if ca_u > 0:
                        Anom_u = Anom_u[:-ca_u]
                    if ca_l > 0:
                        Anom_l = Anom_l[:-ca_l]  
                    c = 0
                    ca_u = 0
                    ca_l = 0
                    
                S.append(st)
                st = Sta
                
            else:
                c=0
                ca_u = 0
                ca_l = 0
                S.append(st)

        else:
            st = np.nan
            ht = np.nan
            mu0 = np.nan
            mu1 = np.nan
            sigma0 = np.nan
            S.append(st)

        H.append(ht)
        Mu0.append(mu0)
        Mu1.append(mu1)
        Sigma0.append(sigma0)
    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, Anom_u, Anom_l, M0_unique, S0_unique, elapsedTime


# Voting Windows Changepoint Detection
def vwcd(X, w, w0, ab, p_thr, vote_p_thr, vote_n_thr, y0, yw, aggreg):
    """
    Voting Windows Changepoint Detection
   
    Parameters:
    ----------
    X (numpy array): the input time-series
    w (int): sliding window size
    w0 (int): pre-chage estimating window size
    h (float): statistic threshold (in terms of sigma0)
    rl (int): number of consecutives deviation to consider a change-point
    ka (int): number of standard deviations to consider a point-anomaly
    alpha_norm (float): Shapyro-Wilker test significance level
    alpha_stat (float): ADF test significance level
    filt_per (float): outlier filter percentil (first window or not. estab.)
    max_var (float): maximum increased variance allowed to consider stab.
    cs_max (int); maximum counter for process not stabilized

    Returns:
    -------
    CP (list): change-points
    M0 (list): estimated mean of the segments
    S0 (list): estimated standar deviation of the segments
    elapsedTime (float): running-time  (microseconds)
    """
    
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
    
    # Auxiliary variables
    N = len(X)
    vote_n_thr = np.floor(w*vote_n_thr)

    # Prior probatilty for a changepoint in a window - Beta-B
    i_ = np.arange(0,w-3)
    prior_w = betabinom(n=w-4,a=ab,b=ab).pmf(i_)

    # prior for vot aggregation
    x_votes = np.arange(1,w+1)
    prior_v = logistic_prior(x_votes, w, y0, yw) 

    votes = {i:[] for i in range(N)} # dictionary of votes 
    votes_agg = {}  # aggregated voteylims

    lcp = 0 # last changepoint
    CP = [] # changepoint list
    M0 = [] # list of post-change mean
    S0 = [] # list of post-change standard deviation

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
            #for nu in range(w):
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
            
            # Store the vote if it meets the hypothesis test threshold
            if p_vote_h >= p_thr:
                j = n-w+1+nu_map_h # Adjusted index 
                votes[j].append(p_vote_h)
            
            # Aggregate the votes for X[n-w+1]
            votes_list = votes[n-w+1]
            num_votes = len(votes_list)
            if num_votes >= vote_n_thr:
                if aggreg == 'posterior':
                    agg_vote = votes_pos(votes_list, prior_v[num_votes-1])
                elif aggreg == 'mean':
                    agg_vote = np.mean(votes_list)
                votes_agg[n-w+1] = agg_vote
                
                # Decide for a changepoit
                if agg_vote > vote_p_thr:
                    if verbose: print(f'Changepoint at n={n-w+1}, p={agg_vote}, n={num_votes} votes')
                    lcp = n-w+1 # last changepoint
                    CP.append(lcp)

    endTime = time.time()
    elapsedTime = endTime-startTime
    return CP, M0, S0, elapsedTime


# PELT Non-Parametric with MBIC penalty
def pelt_np(y, pen, minseglen):
    """
    Non-parametric Prunex Exat Linear Time (Pelt)
    
    Parameters:
    ----------
    y (numpy array): the input time-series
    pen (string): penalty type; possible choices: "None", "SIC", "BIC", 
        "MBIC", "AIC", "Hannan-Quinn"
    minseglen (int): minimum size of each segment

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    y = [i if i > 0 else 1e3 for i in y]
    startTime = time.time()
    CP = [int(i) for i in changepoint.cpts(changepoint_np.cpt_np(FloatVector(y), penalty=pen, minseglen=minseglen))]
    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, elapsedTime


# BOCD - Basic
def bocd_ba(y, w, lamb, kappa0, alpha0, omega0, p_thr, K, min_seg):
    """
    Bayesian Online Changepoint Detection (BOCD) - basic implementation
   
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    lamb (float): lambda hyperparameter - prior for run length
    kappa0 (float): Normal-inverse gamma prior hyperparameter
    alpha0 (float): Normal-inverse gamma prior hyperparameter
    omega0 (float): Normal-inverse gamma prior hyperparameter
    p_thr (float): prob. threshold for online change-point decision
    K (int): run lenght extension cut
    min_seg (int): vicinity for online change-point deciion

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    mean0  = y[:w].mean()
    model = Gaussian(mean0, kappa0, alpha0, omega0)
    hazard = ConstantHazard(lamb) # Hazard probability
    
    # Auxiliary variables initialization
    T = len(y)
    CP = []             # list of changepoints
    lcp = 0             # last changepoint (need to subtract 1)
    log_message = 0
    max_indices = np.array([0])     # indices keep in memory 
    pmean = np.array([np.nan]*T)   # model's predictive mean.

    # Lower triangular matrix with run posteriors of each run lenght size
    log_R = -np.inf*np.ones((T+1, T+1)) 
    log_R[0, 0] = 0  # log 0 == 1

    startTime = time.time()
    for t in range(1, T+1):
        
            # Observe new datum and datum before.
            x = y[t-1]
           
            # Evaluate the hazard function for this interval
            H = hazard(np.array(range(min(t, K))))
            log_H = np.log(H)
            log_1mH = np.log(1-H)

            # Make model predictions.
            pmean[t-1] = np.sum(np.exp(log_R[t-1, :t]) * model.mu[:t])
            
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
            log_R[t, :t+1]  = new_log_joint
            log_R[t, :t+1] -= logsumexp(new_log_joint)
            r = np.exp(log_R[t])
            
            # Decide for a possible changepoint
            # If anomaly, update the model with the last measure (xb)
            # instead of the new one and pass the last message.
            # Wait to update with the new one only after a changepoint.
            if t>1 and r[t-lcp]<=p_thr:
                max_t0 = np.argmax(r)
                lcp=t-max_t0
                if check_previous_cp(lcp-1, CP, min_seg):
                    if verbose: print(f't={t-1} changepoint t={lcp-1} already identified')
                elif lcp-1>0:
                    if verbose: print(f't={t-1} changepoint at t={lcp-1}')
                    CP.append(lcp-1)

            # Update sufficient statistic and pass the message
            model.update_params(x)
            log_message = new_log_joint[max_indices]

    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, elapsedTime


def bocd_ps(y, w, lamb, kappa0, alpha0, omega0, p_thr, K, min_seg):
    """
    Bayesian Online Changepoint Detection (BOCD) - proposed implementation
   
    Parameters:
    ----------
    y (numpy array): the input time-series
    w (int): estimating window size
    lamb (float): lambda hyperparameter - prior for run length
    kappa0 (float): Normal-inverse gamma prior hyperparameter
    alpha0 (float): Normal-inverse gamma prior hyperparameter
    omega0 (float): Normal-inverse gamma prior hyperparameter
    p_thr (float): prob. threshold for online change-point decision
    K (int): run lenght extension cut
    min_seg (int): vicinity for online change-point deciion

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    
    mean0  = y[:w].mean()
    model = Gaussian(mean0, kappa0, alpha0, omega0)
    hazard = ConstantHazard(1e4) # Hazard probability
     
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
     
    new_log_joint = np.nan
     
    log_R_bcp = np.nan
    new_log_joint_bcp = np.nan
    max_indices_bcp = np.nan
    log_message_bcp = np.nan
     

    # Update the joint probability and log_R matrix
    def update_joint_prob(t, x, max_indices_):
        # Evaluate the hazard function for this interval
        nonlocal max_indices, new_log_joint
        
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
        nonlocal log_R, new_log_joint_bcp, max_indices, new_log_joint, log_message       
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
    #R = np.exp(log_R)
    M0_unique.append(model.mu[-1])
    
    return CP, M0_unique, None, elapsedTime


def rrcf_ps(y, num_trees, shingle_size, tree_size, thr, rl):
    """
    Robust Random Cut Forest RRCF) - proposed implementation
   
    Parameters:
    ----------
    y (numpy array): the input time-series
    num_trees (int): number of trees
    shingle_size (int): window length (subsequences size)
    tree_size (int): maximum size of the trees
    thr (float): threshold for anomaly/chnage-point decision
    rl: number of consecutive deviations to consider a change-point

    Returns
    -------
    CP (list): change-points 
    elapsedTime (float): running-time (microseconds)
    """
    np.random.seed(42)
    
    # Auxiliar variables
    w = shingle_size
    c = 0               # statistic deviation counter
    ca = 0
    n = 0               # iteration   
    Anom = []
    CP = []             # changepoint list 
    Scores = [np.nan]*(w-1)
    S = {}              # dictionary of scores
 
    # Use the "shingle" generator to create rolling windows
    shingles_stream = rrcf.shingle(y, size=w)
 
    # Create a forest
    def createNewForest(num_trees):
        forest = []
        for _ in range(num_trees):
            forest.append(rrcf.RCTree())
        return forest
 
 
    # Insert a new point in all the trees of the forest. Return the avg. score.
    def insertPoint(x):
        nonlocal forest, S
        for tree in forest:
            # If tree is above permitted size, drop the oldest point (FIFO)
            if len(tree.leaves) > tree_size:
                tree.forget_point(list(tree.leaves)[1])
            
            # Insert the new point into the tree
            tree.insert_point(xw, index=n)
            
            # Compute codisp on the new point and take the average among all trees
            if not n in S:
                S[n] = 0
            S[n] += tree.codisp(n)/num_trees
        return S[n]
 
 
    # Forget a a specific point (by index) in all the trees
    def forgetPoint(n):
        nonlocal forest
        for tree in forest:
            tree.forget_point(n)
 
    # 1. Create a forest a dict that stores the score of each point
    forest = createNewForest(num_trees)
 
    startTime = time.time()
    for tw, xw in enumerate(shingles_stream):
        
        t = tw+w-1
 
        # 2. update the forest inserting the new point
        Sn = insertPoint(xw)
        
        # 3. if statistic deviate, remove the point from the tree and check the next.
        # if after rl points the statistic deviation persists, declare a changepoint
        Scores.append(Sn)
        dev = Sn > thr
        if dev:
            if verbose>=2: print(f't={t}:possible changepoint')
            if t-w+1 not in Anom:
                Anom.append(t)
                ca = ca + 1
 
            c = c+1
            
            # Forget the point
            forgetPoint(n)
            
            # Check for changepoint
            if c==rl:
                lcp = t-rl
                if verbose: print(f't={t}:changepoint at {lcp}')
                CP.append(lcp)
                Anom = Anom[:-ca]
                c = 0
                ca = 0
                
                # Reset the forest
                forest = createNewForest(num_trees)
                n = 0
                S = {} # create a new dict
                continue
        else:
            c = 0
            ca = 0
        
        n = n+1
    endTime = time.time()
    elapsedTime = endTime-startTime
    
    return CP, elapsedTime