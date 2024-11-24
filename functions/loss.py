import numpy as np
from scipy.optimize import linear_sum_assignment

def get_log_likelihood_individual(param1, param2, sigma=1.0):
    # calculate LL Loss between two individual params
    return - np.sum((param1 - param2) ** 2) / (sigma**2)

def get_loss_matrix(param_set1, param_set2):
    # compute the loss matrix between cluster i in param set 1 and param set 2
    assert len(param_set1) == len(param_set2), 'Different Number of Clusters'
    num_clusters = len(param_set1)
    loss_matrix = np.zeros((num_clusters, num_clusters))
    for i in num_clusters:
        for j in num_clusters:
            loss_matrix[i, j] = get_log_likelihood_individual(param_set1[i], param_set2[j])
    return loss_matrix

def log_likliehood(param_set1, param_set2):
    # find the most likely alignment between parameter sets by calculating cost matrix
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    loss_matrix = get_loss_matrix(param_set1, param_set2)
    row_ind, col_ind = linear_sum_assignment(loss_matrix)
    return loss_matrix[row_ind, col_ind].sum()
