import torch
import numpy as np
from torch.distributions import MultivariateNormal

# Floors a continuous (non-delta) component's per-dimension std, to avoid the unbounded
# density of a Gaussian mixture component as std -> 0+ (likelihood, and therefore NLL, is
# otherwise unbounded for any data point landing near a near-degenerate component's mean --
# a real MLE/posterior singularity, not a bug in whichever method produced the candidate).
#
# Set to match smoothcfg.py's own smooth_eps default (1e-3, see its SMOOTH_EPS/smooth_eps
# constants and smooth_asgmt's degenerate-gm case), NOT an arbitrary value -- that is the
# std the *real* program's own literal sigma=0.00 declarations actually get evaluated at:
# smooth_cfg rewrites a literal "sigma=0.00" it finds in program text into
# `+ gm([1.],[0.],[smooth_eps])` before compilation, so the real program's "discrete"
# dimensions are never truly delta/point-mass by the time compute_likelihood runs (only a
# candidate program's placeholder-ized sigma -- named "_sigmaN", which smooth_cfg
# deliberately skips since it can't know the value yet -- can reach the exact-zero delta
# branch below). Flooring a fitted/sampled near-zero-but-nonzero sigma to less than that
# same 1e-3 would let a candidate look "more discrete than the real program is willing to
# claim to be", which is exactly the unfair comparison this floor exists to prevent.
MIN_SIGMA = 1e-3

def compute_likelihood(output_dist, data_var_list, data):
    """ computes the likelihood of output_dist with respect to variables data_var_list sampled in data """

    data = torch.tensor(data)
    likelihood = 0
    # extract indexes of the variables in the data
    try:
        data_var_index = [output_dist.var_list.index(element) for element in data_var_list ]
    except ValueError:  # if the program doesn't have all the variables we are using for the likelihood
            return torch.tensor(-np.inf)
    except:
            raise
    for k in range(output_dist.gm.n_comp()):
        # extract the covariance matrix only for the variables in the data
        sigma = output_dist.gm.sigma[k][data_var_index][:, data_var_index]
        # first I consider the mu only for variables in the data
        mu = output_dist.gm.mu[k][data_var_index]
        # selects indices of delta (discrete) variables and non-delta (continuous) variables
        #deltas = np.where(np.diag(sigma) == 0)[0]
        #not_deltas = np.where(np.diag(sigma) != 0)[0]
        diag = torch.diag(sigma)

        # Indices where diagonal elements are zero or non-zero
        deltas = torch.where(diag == 0)[0]
        not_deltas = torch.where(diag != 0)[0]
        # saves means of delta and non-delta variables and covariance matrix of non-delta
        mu_delta = mu[deltas]
        mu_not_delta = mu[not_deltas]
        sigma_not_delta = sigma[not_deltas][:, not_deltas]
        # computes pdf of non-delta variables
        if len(mu_not_delta) >= 1:  # if there is at least one continuous variable
            diag_idx = torch.arange(len(not_deltas))
            std_diag = torch.sqrt(sigma_not_delta[diag_idx, diag_idx])
            floored_std_diag = torch.clamp(std_diag, min=MIN_SIGMA)
            if not torch.equal(std_diag, floored_std_diag):
                sigma_not_delta = sigma_not_delta.clone()
                sigma_not_delta[diag_idx, diag_idx] = floored_std_diag ** 2
            continuous_pdf = output_dist.gm.pi[k]*MultivariateNormal(mu_not_delta, sigma_not_delta).log_prob(data[:,not_deltas]).exp()
        else:
            continuous_pdf = output_dist.gm.pi[k]*torch.ones(len(data))
        # computes pmf of delta variables
        if len(mu_delta) >= 1:   # if there is at least one discrete variable
            discrete_pmf = torch.all((mu_delta == data[:, deltas]),dim=1)
        else:
            discrete_pmf = torch.ones(len(data))
        #except ValueError:  # if the covariance matrix is singular
        #    return torch.tensor(-np.inf)
        #except:
        #    raise
        likelihood += continuous_pdf*discrete_pmf # sums likelihood of every data over all components
    
    return torch.sum(torch.log(likelihood))/len(data)