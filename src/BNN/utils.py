import torch.nn.functional as F
import torch
import numpy as np

#  the non-linearity we use in our neural network
def nonlin(x):
    return F.relu(x)


def get_data(N, D_X, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = np.linspace(-1, 1, N)
    X = np.power(X[:, np.newaxis], np.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = np.dot(X, W) + 0.5 * np.power(0.5 + X[:, 1], 2.0) * np.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= np.mean(Y)
    Y /= np.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = np.linspace(-1.3, 1.3, N_test)
    X_test = np.power(X_test[:, np.newaxis], np.arange(D_X))

    # Convert all tensors to float32
    return (
        torch.tensor(X[:, 1:2], dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        torch.tensor(X_test[:, 1:2], dtype=torch.float32),
    )

#LOSSES

def mean_squared_error(y_true, dist):
    return torch.mean((y_true - dist.gm.mean()) ** 2)

def mean_squared_error_bayes(y_true, dist):
    #This works for the means but of course not for the variances
    return torch.mean((y_true - dist.gm.mean()[:-2]) ** 2)

def neg_log_likelihood(y_true, dist):
    #Calculate the log-likelihood of the data given the distribution
    neg_log_likelihood = 0
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    neg_log_likelihood = - torch.log(dist.gm.marg_pdf(y_true, idx))
    return torch.sum(neg_log_likelihood)

def neg_log_likelihood_one(y_true, dist):
    #Calculate the log-likelihood of the single data point given the distribution
    return - torch.log(dist.gm.marg_pdf(y_true, dist.var_list.index('y')))