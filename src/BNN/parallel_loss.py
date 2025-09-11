import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)

import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *

from utils import get_data, mean_squared_error, mean_squared_error_bayes, neg_log_likelihood, neg_log_likelihood_one

torch.set_default_dtype(torch.float64)
'''
# Loop body function for a single sample
def process_sample(j, X_data, Y_data, bnn_pars, cfg):
    sampled_index = np.random.randint(0, len(Y_data.squeeze([-1,1])))
    yj = Y_data.squeeze([1])[sampled_index].to(torch.float64)
    xj = X_data.squeeze([-1,1])[sampled_index]

    bnn_one_pars = bnn_pars.copy()
    bnn_one_pars['x'] = xj.requires_grad_(False)

    current_dist = start_SOGA(cfg, bnn_one_pars, pruning='ranking')
    loss_j = neg_log_likelihood_one(yj, current_dist)

    return loss_j


def run_parallel(batch_size, X, Y, bnn_pars, cfg):
    losses = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_sample, j, X, Y, bnn_pars, cfg)
            for j in range(batch_size)
        ]
        for f in futures:
            losses.append(f.result())
    return torch.sum(losses)
'''
from concurrent.futures import ThreadPoolExecutor

def process_sample(j, X_data, Y_data, bnn_pars, cfg):
    sampled_index = np.random.randint(0, len(Y_data.squeeze([-1,1])))
    yj = Y_data.squeeze([1])[sampled_index].to(torch.float64)
    xj = X_data.squeeze([-1,1])[sampled_index]

    #bnn_one_pars = bnn_pars.copy()
    bnn_pars['x'] = xj.requires_grad_(False) 

    current_dist = start_SOGA(cfg, bnn_pars, pruning='ranking')
    loss_j = neg_log_likelihood_one(yj, current_dist)

    return loss_j

def run_parallel(batch_size, X, Y, bnn_pars, cfg):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_sample, j, X, Y, bnn_pars, cfg)
            for j in range(batch_size)
        ]
        # collect results into a single tensor
        losses = torch.stack([f.result() for f in futures])

    total_loss = losses.sum()   # still a tensor
    return total_loss
