# Online Gaussians

Authors: Abhishek, Ross, Qianfei, Arni, Moe

Repo Structure:
- notebooks_old: Contains some old notebooks throughout the implementation of our algorihms or generation of old datasets, but they are no longer used.
- folder *dataset*
    - 6_Gauss_Blobs.npy: contains data points from 6 Gaussian blobs.
    - dist_shift_add_gaus.npy: contains the 1D drifting dataset with 1 Gaussian distribution generating more data points at the second half of the dataset.
    - dist_shift_remove_gaus.npy: contains the 1D drifting dataset but with 1 Gaussian distribution generating less data points at the second half of the dataset.
    - Gaussian_blobs_drift.npy: contains the 2D drifting dataset with each Gaussian distribution having a velocity when generating the data points (i.e., the distribution is moving when generating data points).
    - shuffled.npy: shuffled data points from a 1D Gaussian distribution.
- folder *functions*: contains functions to generate each dataset and to plot graphs, including the loss plot and the trace of a algorithm's outputted centers after each data point.
- incremental_EM: The implementation of online incremental EM with Gaussian Mixture models, with plots of running the algorithm on some basic datasets 
- stepwise_EM: the Tmplmentation of online stepwise EM with Gaussian Mixture models, with plots of running the algorithm on the same basic dataset
- loans_drift_learn.ipynb: contains experiments and visualization code with Loan Dataset using iEM, sEM, Batch GMM, and SSU. 
