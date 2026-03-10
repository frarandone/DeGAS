# Contains some general purpose classes, functions and variables used by the SOGA Python Libraries. 
# In particular it contains:
# - import statement for auxiliary Python libraries;
# - definition of tolerance parameters used by various functions;
# - classes definition for representing distributions and Gaussian Mixtures;
# - function definitions for numerical stability of the covariance matrices;
# - function definitions invoked by multiple functions in different libraries.

# TO DO:
# -  add controls on the attributes of GaussianMix (lenghts of pi, mu, sigma, dimensions of mu and sigma)

# AUXILIARY LIBRARIES 
import torch
import torch.distributions as distributions
import botorch.utils.probability.mvnxpb as mvn
#from copy import deepcopy, copy

#from sympy import *
import re
#import numpy as np
#from scipy.stats import norm
#from scipy.stats import truncnorm
#from scipy.stats import multivariate_normal as mvnorm
from itertools import product
from functools import partial


### TOLERANCE PARAMETERS 

EPS = 1e-5              # used on the diagonal of the initial distribution
#DELTA_EIG = 1e-8        # when making the cov matrix psd increases eigenvalues by this
TOL_EIG = 1e-15          # eigenvalues below this value are considered zero
TOL_PROB = 1e-10# 1e-4         # probability below prob_tol are treated as zero
TOL_ERR = 5e-3          # error tolerance (print an error message if error is above)
INFTY = 1e10            # infinity
# SMOOTHING PARAMETERS
SMOOTH_EPS = 1e-3    # starting noise for smoothing
SMOOTH_DELTA = 1e-3  # addition to gaussian noise for smoothing


#delta_tol = 1e-10 # if the 1-norm of a covariance matrix is <= delta_tol the corresponding Gaussian component is treated as a delta
#prob_tol = 1e-10 # probability below prob_tol are treated as zero
#eig_tol = 1e-4

### CLASSES FOR DISTRIBUTIONS AND GAUSSIAN MIXTURES

class GaussianMix():
    """ A Gaussian Mixtures is represented by a list of mixing coefficients (stored in pi), a list of means (stored in mu) and a list of covariance matrices (stored in sigma)."""

    def __init__(self, pi, mu, sigma):
        self.pi = pi         # pi is a tensor (c, 1) where c is the number of components
        self.mu = mu         # mu is a tensor (c, d) where d is the dimension of the space
        self.sigma = sigma   # sigma is a tensor (c, d, d) where d is the dimension of the space

    def n_comp(self):
        return self.pi.shape[0]

    def n_dim(self):
        return self.mu.shape[1]

    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + '\nmu: ' + str(self.mu) + '\nsigma: ' + str(self.sigma)
        return str_repr

    def comp(self, k):
        return GaussianMix(torch.tensor([[1.]]), torch.clone(self.mu[k,:].unsqueeze(0)), torch.clone(self.sigma[k,:,:].unsqueeze(0)))

    # Pdfs
    def comp_pdf(self, x, k):
        if self.n_dim() > 1:
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
            except ValueError:
                sigma = self.sigma[k]
                eigs, _ = torch.linalg.eigh(sigma)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(sigma == sigma.T)
                if not is_psd:
                    print(eigs)
                    print('matrix is not psd!')
                    print(sigma)
                    raise
                if not is_sym:
                    self.sigma[k] = make_sym(self.sigma[k])
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).log_prob(x)).reshape(x.shape)

    def marg_comp_pdf(self, x, k, idx):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if isinstance(idx, list):
            cov_submatrix = torch.clone(self.sigma[k][torch.tensor(idx, device=self.sigma.device, dtype=torch.long).unsqueeze(1), torch.tensor(idx, device=self.sigma.device, dtype=torch.long)])
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], cov_submatrix).log_prob(x))
            except ValueError:
                eigs, _ = torch.linalg.eigh(cov_submatrix)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(cov_submatrix == cov_submatrix.T)
                if not is_psd:
                    print(eigs)
                    print('matrix k={} is not psd!'.format(k))
                    print(cov_submatrix)
                    raise
                if not is_sym:
                    print('matrix k={} is not symmetric!'.format(k))
                    self.sigma[k][torch.tensor(idx, device=self.sigma.device, dtype=torch.long).unsqueeze(1), torch.tensor(idx, device=self.sigma.device, dtype=torch.long)] = new_cov_submatrix = make_sym(cov_submatrix)
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], covariance_matrix=new_cov_submatrix).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).log_prob(x))


    def pdf(self, x):
        comp_pdfs = torch.stack([self.comp_pdf(x, k) for k in range(self.n_comp())], dim=1)
        pdf = torch.matmul(comp_pdfs.squeeze(2), self.pi.view(-1, 1))
        return pdf


    def marg_pdf(self, x, idx):
        comp_pdfs = torch.stack([self.marg_comp_pdf(x, k,idx) for k in range(self.n_comp())], dim=1)
        pdf = torch.matmul(comp_pdfs, self.pi)
        return pdf

    # Cdfs

    def comp_cdf(self, x, k):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if self.n_dim() > 1:
            return mvncdf(x, self.mu[k], self.sigma[k])
        else:
            return distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).cdf(x)

    def marg_comp_cdf(self, x, k, idx):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if isinstance(idx, list):
            # Create index tensor on same device as sigma to avoid device transfers
            idx_tensor = torch.tensor(idx, device=self.sigma.device, dtype=torch.long)
            cov_submatrix = torch.clone(self.sigma[k][idx_tensor.unsqueeze(1), idx_tensor])
            return mvncdf(x, self.mu[k][idx], cov_submatrix)
        else:
            return distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).cdf(x)


    def cdf(self, x):
        comp_cdfs = torch.stack([self.comp_cdf(x, k) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf

    def marg_cdf(self, x, idx):
        comp_cdfs = torch.stack([self.marg_comp_cdf(x, k, idx) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf


    # Moments of mixtures
    def mean(self):
        return torch.sum(self.pi * self.mu, dim=0)

    def cov(self):
        pi = self.pi.view(-1, 1, 1)
        v = self.mu - self.mean()
        cov = (pi * self.sigma).sum(dim=0) + torch.mm(v.t(), self.pi*v)
        return cov

    # utilities

    def delete_zeros(self):
        mask = (self.pi >= TOL_PROB).squeeze(1)   # (c,)
        if mask.all():
            return
        self.pi = self.pi[mask]                   # shape (k, 1)
        self.pi = self.pi / self.pi.sum()
        self.mu = self.mu[mask]
        self.sigma = self.sigma[mask]

class GaussianMixGPU():
    """ A Gaussian Mixtures is represented by a list of mixing coefficients (stored in pi), a list of means (stored in mu) and a list of covariance matrices (stored in sigma)."""
    
    def __init__(self, pi, mu, sigma):
        # Ensure all tensors are on the same device without forcing .cuda()
        # All inputs should already be tensors on the correct device from calling code
        if isinstance(pi, torch.Tensor) and isinstance(mu, torch.Tensor) and isinstance(sigma, torch.Tensor):
            # Get target device from first tensor
            target_device = pi.device
            # Only move if devices differ (avoid unnecessary transfers)
            self.pi = pi if pi.device == target_device else pi.to(target_device)
            self.mu = mu if mu.device == target_device else mu.to(target_device)
            self.sigma = sigma if sigma.device == target_device else sigma.to(target_device)
        else:
            # Fallback: convert to tensors (should rarely happen in GPU code)
            self.pi = pi if isinstance(pi, torch.Tensor) else torch.tensor(pi)
            self.mu = mu if isinstance(mu, torch.Tensor) else torch.tensor(mu)
            self.sigma = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(sigma)
            # Ensure all on same device
            target_device = self.pi.device
            if self.mu.device != target_device:
                self.mu = self.mu.to(target_device)
            if self.sigma.device != target_device:
                self.sigma = self.sigma.to(target_device)
    
    def n_comp(self):
        return self.pi.shape[0]
    
    def n_dim(self):
        return self.mu.shape[1]
    
    def __repr__(self):
        str_repr = 'pi: ' + str(self.pi) + '\nmu: ' + str(self.mu) + '\nsigma: ' + str(self.sigma)
        return str_repr
    
    def comp(self, k):
        # Create tensor on same device as self.pi
        device = self.pi.device
        return GaussianMixGPU(torch.tensor([[1.]], device=device), torch.clone(self.mu[k,:].unsqueeze(0)), torch.clone(self.sigma[k,:,:].unsqueeze(0)))
        
    # Pdfs 
    def comp_pdf(self, x, k):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if self.n_dim() > 1:
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
            except ValueError:
                sigma = self.sigma[k]
                eigs, _ = torch.linalg.eigh(sigma)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(sigma == sigma.T)
                if not is_psd:
                    print(eigs)
                    print('matrix is not psd!')
                    print(sigma)
                    raise 
                if not is_sym:
                    self.sigma[k] = make_sym(self.sigma[k])
                return torch.exp(distributions.MultivariateNormal(self.mu[k], covariance_matrix=self.sigma[k]).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).log_prob(x)).reshape(x.shape)
            
    def marg_comp_pdf(self, x, k, idx):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if isinstance(idx, list):
            cov_submatrix = torch.clone(self.sigma[k][torch.tensor(idx, device=self.sigma.device, dtype=torch.long).unsqueeze(1), torch.tensor(idx, device=self.sigma.device, dtype=torch.long)])
            try:
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], cov_submatrix).log_prob(x))
            except ValueError:
                eigs, _ = torch.linalg.eigh(cov_submatrix)
                is_psd = torch.all(eigs > 0)
                is_sym = torch.all(cov_submatrix == cov_submatrix.T)
                if not is_psd:
                    print(eigs)
                    print('matrix k={} is not psd!'.format(k))
                    print(cov_submatrix)
                    raise 
                if not is_sym:
                    print('matrix k={} is not symmetric!'.format(k))
                    self.sigma[k][torch.tensor(idx, device=self.sigma.device, dtype=torch.long).unsqueeze(1), torch.tensor(idx, device=self.sigma.device, dtype=torch.long)] = new_cov_submatrix = make_sym(cov_submatrix)
                return torch.exp(distributions.MultivariateNormal(self.mu[k][idx], covariance_matrix=new_cov_submatrix).log_prob(x))
        else:
            return torch.exp(distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).log_prob(x))

    
    def pdf(self, x):
        comp_pdfs = torch.stack([self.comp_pdf(x, k) for k in range(self.n_comp())], dim=1)
        pdf = torch.matmul(comp_pdfs.squeeze(2), self.pi.view(-1, 1))
        return pdf
        
    
    def marg_pdf(self, x, idx):
        x = x.to(self.mu.device)
        if isinstance(idx, list):
            idx_t = torch.tensor(idx, device=self.sigma.device, dtype=torch.long)
            mu_sub = self.mu[:, idx_t]                              # (c, d')
            sigma_sub = self.sigma[:, idx_t.unsqueeze(1), idx_t]   # (c, d', d')
            try:
                # Batched MultivariateNormal: batch_shape=(1,c), event_shape=(d')
                mvn_dist = distributions.MultivariateNormal(
                    mu_sub.unsqueeze(0),
                    covariance_matrix=sigma_sub.unsqueeze(0)
                )
                comp_pdfs = mvn_dist.log_prob(x.unsqueeze(1)).exp()  # (n, c)
            except ValueError:
                comp_pdfs = torch.stack([self.marg_comp_pdf(x, k, idx) for k in range(self.n_comp())], dim=1)
        else:
            norm_dist = distributions.Normal(
                self.mu[:, idx].unsqueeze(0),
                torch.sqrt(self.sigma[:, idx, idx]).unsqueeze(0)
            )
            comp_pdfs = norm_dist.log_prob(x.unsqueeze(1)).exp()  # (n, c)
        pdf = torch.matmul(comp_pdfs, self.pi)
        return pdf

    def marg_log_pdf(self, x, idx):
        """Log marginal PDF via logsumexp — numerically stable for high-dimensional idx."""
        x = x.to(self.mu.device)
        log_pi = torch.log(self.pi.squeeze(1))  # (c,)
        if isinstance(idx, list):
            idx_t = torch.tensor(idx, device=self.sigma.device, dtype=torch.long)
            mu_sub = self.mu[:, idx_t]                              # (c, d')
            sigma_sub = self.sigma[:, idx_t.unsqueeze(1), idx_t]   # (c, d', d')
            try:
                mvn_dist = distributions.MultivariateNormal(
                    mu_sub.unsqueeze(0),
                    covariance_matrix=sigma_sub.unsqueeze(0)
                )
                log_comp_pdfs = mvn_dist.log_prob(x.unsqueeze(1))   # (n, c)
            except ValueError:
                log_comp_pdfs = torch.stack([
                    distributions.MultivariateNormal(mu_sub[k], covariance_matrix=sigma_sub[k]).log_prob(x)
                    for k in range(self.n_comp())
                ], dim=1)
        else:
            norm_dist = distributions.Normal(
                self.mu[:, idx].unsqueeze(0),
                torch.sqrt(self.sigma[:, idx, idx]).unsqueeze(0)
            )
            log_comp_pdfs = norm_dist.log_prob(x.unsqueeze(1))      # (n, c)
        return torch.logsumexp(log_comp_pdfs + log_pi.unsqueeze(0), dim=1)  # (n,)

    # Cdfs

    def comp_cdf(self, x, k):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if self.n_dim() > 1:
            return mvncdf(x, self.mu[k], self.sigma[k])
        else:
            return distributions.Normal(self.mu[k], torch.sqrt(self.sigma[k])).cdf(x)

    def marg_comp_cdf(self, x, k, idx):
        # Ensure x is on the same device as the distribution
        x = x.to(self.mu.device)
        if isinstance(idx, list):
            # Create index tensor on same device as sigma to avoid device transfers
            idx_tensor = torch.tensor(idx, device=self.sigma.device, dtype=torch.long)
            cov_submatrix = torch.clone(self.sigma[k][idx_tensor.unsqueeze(1), idx_tensor])
            return mvncdf(x, self.mu[k][idx], cov_submatrix)
        else:
            return distributions.Normal(self.mu[k][idx], torch.sqrt(self.sigma[k][idx,idx])).cdf(x)


    def cdf(self, x):
        comp_cdfs = torch.stack([self.comp_cdf(x, k) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf

    def marg_cdf(self, x, idx):
        comp_cdfs = torch.stack([self.marg_comp_cdf(x, k, idx) for k in range(self.n_comp())], dim=1)
        cdf = torch.matmul(comp_cdfs, self.pi.view(-1, 1))
        return cdf


    # Moments of mixtures
    def mean(self):
        return torch.sum(self.pi * self.mu, dim=0)

    def cov(self):
        pi = self.pi.view(-1, 1, 1)
        v = self.mu - self.mean()
        cov = (pi * self.sigma).sum(dim=0) + torch.mm(v.t(), self.pi*v)
        return cov

    # utilities

    def delete_zeros(self):
        mask = (self.pi >= TOL_PROB).squeeze(1)   # (c,)
        if mask.all():
            return
        self.pi = self.pi[mask]                   # shape (k, 1)
        self.pi = self.pi / self.pi.sum()
        self.mu = self.mu[mask]
        self.sigma = self.sigma[mask]


class Dist():
    """ A distribution is given by a ordered list of variable names, stored in var_list, and a Gaussian Mixture, stored in gm, describing the joint distribution over the variable vector"""
    def __init__(self, var_list, gm):
        self.var_list = var_list
        self.gm = gm
        
    def __str__(self):
        return 'Dist<{},{}>'.format(self.var_list, self.gm)
    
    def __repr__(self):
        return str(self)
    
class DistGPU():
    """ A distribution is given by a ordered list of variable names, stored in var_list, and a Gaussian Mixture, stored in gm, describing the joint distribution over the variable vector"""
    def __init__(self, var_list, gm):
        self.var_list = var_list
        self.gm = gm
        
    def __str__(self):
        return 'Dist<{},{}>'.format(self.var_list, self.gm)
    
    def __repr__(self):
        return str(self)

    def get_device(self):
        return self.gm.pi.get_device()

### CDF FUNCTION OF MULTIVARIATE GAUSSIAN

def mvncdf(x, mean, cov):
    # Ensure x has a batch dimension
    if x.dim() == 1:
        x = x.unsqueeze(0)
    batch_size = x.shape[0]
    dim = x.shape[1]
    device = x.device
    
    # Vectorize bounds creation instead of nested list comprehensions
    # Create bounds tensor: shape (batch_size, dim, 2)
    # First column is -inf, second column is x[i, j] - mean[j]
    bounds = torch.zeros(batch_size, dim, 2, device=device)
    bounds[:, :, 0] = -torch.inf
    bounds[:, :, 1] = x - mean.unsqueeze(0)  # Broadcasting: (batch_size, dim) - (dim,)
    
    # Initialize result tensor on same device
    res = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        result = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds[i]).solve())
        if result.isnan():
            cov_sym = make_sym(cov)
            result = torch.exp(mvn.MVNXPB(covariance_matrix=cov_sym, bounds=bounds[i]).solve())
        res[i] = result
    return res

# THIS VERSION ONLY WORKS FOR SINGLE SAMPLES x
#def mvncdf(x, mean, cov):
#    bounds = torch.tensor([[-torch.inf, x[i] - mean[i]] for i in range(len(x))])
#    res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
#    if res.isnan():
#        cov = make_sym(cov)
#        res = torch.exp(mvn.MVNXPB(covariance_matrix=cov, bounds=bounds).solve())
#    return res

### CUSTOM CLASS FOR UNIVARIATE TRUNCATED NORMAL

class TruncatedNormal():
    """ Univariate Truncated Normal distribution. Helps computing moments using torch utils."""

    def __init__(self, loc, scale, a, b):
        self.loc = loc
        self.scale = scale
        self.low_bound = a
        self.up_bound = b

        # auxiliary normal
        self.norm = distributions.Normal(torch.zeros(self.loc.shape), torch.ones(self.scale.squeeze(2).shape))

        # rescaled bounds
        self.alpha = (self.low_bound - self.loc)/(self.scale.squeeze(2))
        self.beta = (self.up_bound - self.loc)/(self.scale.squeeze(2))
        self.phi_alpha = self.norm.log_prob(self.alpha).exp()
        self.phi_beta = self.norm.log_prob(self.beta).exp()

        # normalization constant
        self.norm_const = self.norm.cdf(self.beta) - self.norm.cdf(self.alpha)
    # mean
    def mean(self):
        return self.loc + self.scale.squeeze(2)*(self.phi_alpha - self.phi_beta)/self.norm_const

    # variance
    def var(self):
        prod_beta = self.beta * self.phi_beta
        prod_alpha = self.alpha * self.phi_alpha  
        return (self.scale.squeeze(2)**2*(torch.tensor(1.) - (prod_beta - prod_alpha)/self.norm_const - ((self.phi_alpha - self.phi_beta)/self.norm_const)**2)).unsqueeze(2)

class TruncatedNormalGPU():
    """ Univariate Truncated Normal distribution. Helps computing moments using torch utils."""

    def __init__(self, loc, scale, a, b):
        self.loc = loc
        self.scale = scale
        self.low_bound = a
        self.up_bound = b

        # auxiliary normal on same device as loc
        device = loc.device
        self.norm = distributions.Normal(torch.zeros(self.loc.shape, device=device), torch.ones(self.scale.squeeze(2).shape, device=device), validate_args=False)

        # rescaled bounds
        self.alpha = (self.low_bound - self.loc)/(self.scale.squeeze(2))
        self.beta = (self.up_bound - self.loc)/(self.scale.squeeze(2))
        self.phi_alpha = self.norm.log_prob(self.alpha).exp()
        self.phi_beta = self.norm.log_prob(self.beta).exp()

        # normalization constant
        self.norm_const = self.norm.cdf(self.beta) - self.norm.cdf(self.alpha)
    # mean
    def mean(self):
        return self.loc + self.scale.squeeze(2)*(self.phi_alpha - self.phi_beta)/self.norm_const

    # variance
    def var(self):
        prod_beta = self.beta * self.phi_beta
        prod_alpha = self.alpha * self.phi_alpha  
        # Create tensor on same device as scale
        return (self.scale.squeeze(2)**2*(torch.tensor(1., device=self.scale.device) - (prod_beta - prod_alpha)/self.norm_const - ((self.phi_alpha - self.phi_beta)/self.norm_const)**2)).unsqueeze(2)

### FUNCTIONS FOR PARSING

def extend_dist(self, dist):
    """ Extends the current distribution with the auxiliary variables. Returns a new GaussianMix object."""
    device = dist.get_device() # pytorch tensor get_device returns GPU id 0,1,etc or -1 (eg for CPU)
    if device >= 0:
        device = f'cuda:{device}'
    else:
        device = 'cpu'


    if len(self.aux_pis) > 0:
        old_dim = dist.gm.n_dim()
        new_dim = old_dim + len(self.aux_pis)
        n_comp = dist.gm.n_comp()

        # Pre-compute all combinations to enable vectorization
        combinations = list(product(*[range(len(mean)) for mean in self.aux_means]))
        n_combinations = len(combinations)
        
        # Pre-allocate tensors with correct size to avoid repeated vstack operations.
        # Use torch.empty for tensors that are fully overwritten in the loop (avoids CUDA memset).
        # Cross-blocks (old ⊥ aux) are zeroed explicitly once; diagonal blocks filled per iteration.
        n_total = n_combinations * n_comp
        dtype = dist.gm.pi.dtype
        new_pis = torch.empty((n_total, 1), device=device, dtype=dtype)
        new_mus = torch.empty((n_total, new_dim), device=device, dtype=dtype)
        new_sigmas = torch.empty((n_total, new_dim, new_dim), device=device, dtype=dtype)
        new_sigmas[:, :old_dim, old_dim:] = 0  # old vars ⊥ aux vars
        new_sigmas[:, old_dim:, :old_dim] = 0
        
        # Process all combinations in batch
        for idx, part in enumerate(combinations):
            start_idx = idx * n_comp
            end_idx = (idx + 1) * n_comp
            
            # for each combination multiplies the original weight by the weights of the combination
            aux_pi = torch.prod(torch.stack([self.aux_pis[i][part[i]] for i in range(len(part))]))
            new_pis[start_idx:end_idx] = aux_pi * dist.gm.pi
            
            # for each combination creates new means
            # Create aux_mu directly on device
            aux_mu_parts = [self.aux_means[i][part[i]] for i in range(len(part))]
            aux_mu = torch.hstack(aux_mu_parts).to(device) if len(aux_mu_parts) > 0 else torch.empty(0, device=device)
            new_mus[start_idx:end_idx] = torch.cat([dist.gm.mu, aux_mu.expand(n_comp, len(aux_mu))], dim=1)
            
            # for each combination creates new covs
            aux_sigma = torch.diag(torch.hstack([self.aux_covs[i][part[i]] for i in range(len(part))]))
            # Create aux_sigmas directly on device
            new_sigmas[start_idx:end_idx, :old_dim, :old_dim] = dist.gm.sigma
            new_sigmas[start_idx:end_idx, old_dim:, old_dim:] = aux_sigma

        extended_gm = GaussianMixGPU(new_pis, new_mus, new_sigmas)
        extended_gm.delete_zeros()

        return extended_gm
    else:
        return dist.gm

        
### FUNCTIONS FOR NUMERICAL STABILITY OF COVARIANCE MATRICES

# this is commented because it breaks the gradient, it is better not to cause non-psd matrices (use smoothing instead)
#def make_psd(sigma):
#    """
#    Triggered when sigma is not positive semidefinite. Sets to 1e-10 negative eigenvalues of sigma. If the eigenvalues or the total error in the substitution are above a certain threshold prints an error message.
#    """
#    eig, M = torch.linalg.eigh(sigma)
#    add = 0
#    c_it = 0
#    if torch.all(eig > TOL_EIG):
#        #print('not correcting')
#        return sigma
#    while not torch.all(eig > TOL_EIG):
#    #while True:
#        c_it+=1
#        add += DELTA_EIG
#        for i, e in enumerate(eig):
#            if e <= TOL_EIG:
#                eig[i] = add
#        new_sigma = torch.mm(torch.mm(M, torch.diag(eig)), M.t())
#        eig, M = torch.linalg.eigh(new_sigma)
#    rel_err = torch.sum(torch.abs(new_sigma-sigma))
#    if rel_err > TOL_ERR:
#        print('Warning: eigenvalue substitution led to an error of: {}'.format(rel_err))
#    #print('corrected output', new_sigma)
#    return new_sigma

def make_sym(sigma, eig_tol=1e-3):
    """
    Makes a 2D PyTorch tensor symmetric by averaging mismatched elements.
    Prints an error message if the error exceeds eig_tol.
    """
    # Ensure sigma is a PyTorch tensor
    sigma = sigma.clone()  # To avoid modifying the input tensor
    # Average the matrix with its transpose
    symmetric_sigma = (sigma + sigma.T) / 2
    # Compute the difference introduced by the symmetrization
    diff = torch.abs(symmetric_sigma - sigma)
    # Find indices where the difference exceeds the tolerance
    indices = torch.nonzero(diff > TOL_ERR, as_tuple=False)
    for i, j in indices:
        print(f"Substituting {sigma[i, j].item()} with {symmetric_sigma[i, j].item()}")
    return symmetric_sigma