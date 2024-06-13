# Master thesis repository

## Description
This repository contains the code and data used in my master thesis **Change-point detection in time series: a study of online methods using computer network measeurements**, completed in the [Systems Engineering and Computer Science Program](https://www.cos.ufrj.br/index.php/en/) of the [COPPE/UFRJ](https://coppe.ufrj.br/en/home-en/). If you find any bug or have some doubt or observation, please let me know.

## How to cite
> ALMEIDA, C. M. 2024. *Change-point Detection in Time Series: A Study of Online Methods Using Computer Network Measurements. *M.Sc.* thesis. COPPE/UFRJ, Rio de Janeiro, RJ, Brasil.

```BibTeX
@mastersthesis{cma2024thesis,
	title={Change-point Detection in Time Series: A Study of Online Methods Using Computer Network Measurements},
 	author={Almeida, Cleiton Moya de},
 	year={2024},
	school={COPPE/UFRJ}
	address={Rio de Janeiro, RJ, Brasil}
	type={Master's thesis}
}
```

## Instructions
- The changepoint methods are implemented in the module [Experiment/changepoint_module.py](Experiment/changepoint_module.py).
- There are three experiments:
	- *NDT Dataset*: run the file [Experiment/experiment_ndt.py](Experiment/experiment_ndt.py) to evaluate the methods using the NDT dataset. The hyperparameters are setted in the body of the script.
		- For each method, a [pickled Pandas object](https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html) containing the results is created in the folder [Experiment/results_ndt].
	- *Shao Dataset*: run the file [Experiment/experiment_shao.py](Experiment/experiment_shao.py) to evaluate the methods using the Shao dataset. 
		- For each method, a pickled Pandas object containing the results is created in the folder [Experiment/results_shao].
	- *Shao dataset grid search*: run the file [Experiment/experiment_shao.py](Experiment/experiment_shao.py) to reproduce the grid search used for hyperparameters tunning. Two pickle files are created: 
		- `df_shaogrid_*.pkl`: dataframe containing the evaluation summary for each hyperparameter set;
		- `df_shaobest_*.pkl`: dataframe containing the dataset evaluation with the best hyperparamter set finded;
	- Be aware that the VWCD and RRCF methods can take several hours to complete. You can easily modify the script to run only selected methods.
- The graphics can be reproduced using the *scripts* in the folder [Figures_scripts](Figures_scripts/). I used the [Spyder](https://github.com/spyder-ide/spyder) environment to run the scripts and manually save the figures.

## Dependencies

### Python

The code was developed and tested with **Python 3.9** using the following packages:

- Numpy 1.23.5
- Scipy 1.11.4
- Pandas 1.5.2
- Statsmodels 0.14.1
- Matplotlib 3.8.2
- R2Py 3.5.11
- rrcf 0.4.4
- Munkres 1.1.4
- Seaborn 0.13.1

### R 

- To run the The Pelt-NP algorithm, we use the **R 4.3.2** with the package [changepoint.np](https://cran.r-project.org/web/packages/changepoint.np/index.html):
	- changepoint.np 1.0.5
- We call R function inside Python code using the `R2Py` package. The environment variables `R_HOME` and `R_USER` shall be properly setted.
- To plot the Average Run Lenght functions (Figure 3.2), we used the package [Statistical Process Control](https://cran.r-project.org/web/packages/spc/index.html): 
	- spc 0.6.8

## Other repositories

- [Raspberry data collection](https://github.com/danielatk/wptagent-automation): Code used in the Raspberry PI clients to automate the data collection.
- [Shao Dataset](https://github.com/WenqinSHAO/rtt): Wenqing Shao provided labeled datasets of round trip time and scripts to detect and evaluate changepoint methods. In this work, we used the [real_trace_labelled](https://github.com/WenqinSHAO/rtt/tree/master/dataset/real_trace_labelled) dataset and also the benchmark funtions. Thanks Wenqin Shao to share your work.
- [Mlab NDT](https://github.com/m-lab/ndt7-client-go): The NDT client reference implementation, in Go. We compiled this client to ARM (Raspberry PI OS 64bits) to execute the NDT tests.
- [bocd](https://github.com/gwgundersen/bocd): Python implementation of Bayesian Online Changepoint Detection for a normal model with unknown mean parameter
- [DSM-bocd](https://github.com/maltamiranomontero/DSM-bocd): Provide python implementations for BOCD using the normal model with unknown mean and variance parameter, multivariate gaussian and the DSM-BOCD.
- [rrcf](https://github.com/kLabUM/rrcf): Python implementation of the Robust Random Cut Forest algorithm.
- [Ruptures](https://github.com/deepcharles/ruptures): Python library for off-line change point detection (unfortunatelly, does no implement the Pelt-NP neither the MBIC loss fuctiom).
- [scp](https://cran.r-project.org/web/packages/spc/index.html): R package Statistical Process Control – Calculation of ARL and Other Control Chart Performance Measures
- [changepoint.np](https://cran.r-project.org/web/packages/changepoint.np/index.html): R implementation of the Pelt-NP