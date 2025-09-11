import torch.nn.functional as F
import torch
import numpy as np
import random
import torch.nn as nn

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)

from sogaPreprocessor import *
from producecfg import *
from smoothcfg import *
from libSOGA import *
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import time

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


def generate_bnn_parameters(n, m, sigma_init=0.1, seed=None):
    """
    Generate parameter dictionary for a fully-connected BNN.
    
    Args:
        n (int): number of hidden layers
        m (int): number of neurons per hidden layer
        sigma_init (float): initial sigma for all params
        seed (int, optional): random seed for reproducibility
    
    Returns:
        dict: dictionary with keys like 'muw1', 'sigmaw1', 'mub1', 'sigmab1', ...
    """
    if seed is not None:
        random.seed(seed)

    params = {}
    w_counter = 1
    b_counter = 1

    # Hidden layers
    for layer in range(n):
        n_inputs = m if layer > 0 else 1  # first layer takes _x (1 input)
        for i in range(m):  # each neuron
            # weights for this neuron
            for j in range(n_inputs):
                params[f"muw{w_counter}"] = random.uniform(-1, 1)
                params[f"sigmaw{w_counter}"] = sigma_init
                w_counter += 1
            # bias for this neuron
            params[f"mub{b_counter}"] = random.uniform(-1, 1)
            params[f"sigmab{b_counter}"] = sigma_init
            b_counter += 1

    # Output layer (one neuron)
    for j in range(m):  # one weight per hidden unit
        params[f"muw{w_counter}"] = random.uniform(-1, 1)
        params[f"sigmaw{w_counter}"] = sigma_init
        w_counter += 1
    params[f"mub{b_counter}"] = random.uniform(-1, 1)
    params[f"sigmab{b_counter}"] = sigma_init

    return params

def append_observe_block(code: str, statement: str, constraint: str) -> str:
    """
    Append a logical observation block to DSL code.

    Args:
        code (str): The existing DSL code as a string.
        statement (str): Condition for the if statement.
        constraint (str): Constraint to observe inside the block.

    Returns:
        str: The updated DSL code with the appended observation block.
    """
    block = []
    block.append(f"if {statement} {{")
    block.append(f"    observe({constraint});")
    block.append("} else {")
    block.append("    skip;")
    block.append("} end if;")
    return code + "\n" + "\n".join(block)


def generate_bnn_code_relu(n, m):
    """
    Generate DSL code for a fully-connected Bayesian Neural Network
    with n hidden layers and m neurons per hidden layer.

    - One multiplication per line
    - Multiple additions allowed in one line
    - Reuses a single `tmp` variable for extra multiplications
    - Noisy-ReLU activations
    - Gaussian likelihood noise at output
    """

    code = []
    w_counter = 1
    b_counter = 1

    # --- Declare weights and biases ---
    for layer in range(n):
        for i in range(m):  # each neuron in this layer
            for j in range(m if layer > 0 else 1):  # inputs: m from prev layer, or 1 from x
                code.append(f"w{w_counter} = gm([1.], [_muw{w_counter}], [_sigmaw{w_counter}]);")
                w_counter += 1
            code.append(f"b{b_counter} = gm([1.], [_mub{b_counter}], [_sigmab{b_counter}]);")
            b_counter += 1

    # Output layer weights + bias
    for j in range(m):
        code.append(f"w{w_counter} = gm([1.], [_muw{w_counter}], [_sigmaw{w_counter}]);")
        w_counter += 1
    code.append(f"b{b_counter} = gm([1.], [_mub{b_counter}], [_sigmab{b_counter}]);")
    b_counter += 1
    code.append("")

    # --- Forward pass ---
    prev_layer_vars = ["_x"]

    for layer in range(n):
        layer_vars = []
        for i in range(m):
            a_var = f"a{layer}{i}"

            # First input
            wi = (layer * m * len(prev_layer_vars)) + (i * len(prev_layer_vars)) + 1
            code.append(f"{a_var} = {prev_layer_vars[0]} * w{wi};")

            # Remaining inputs -> reuse tmp
            for j, prev in enumerate(prev_layer_vars[1:], start=1):
                wi = (layer * m * len(prev_layer_vars)) + (i * len(prev_layer_vars)) + j + 1
                code.append(f"tmp = {prev} * w{wi};")
                code.append(f"{a_var} = {a_var} + tmp;")

            # Add bias
            bi = (layer * m) + i + 1
            code.append(f"{a_var} = {a_var} + b{bi};")

            # Noisy ReLU
            code.append(f"if {a_var} < 0.0 {{")
            code.append(f"    {a_var} = gauss(0., 0.01);")
            code.append("} else {")
            code.append("    skip;")
            code.append("} end if;")
            code.append("")

            layer_vars.append(a_var)
        prev_layer_vars = layer_vars

    # --- Output layer ---
    # First multiplication
    wi = w_counter - m
    code.append(f"y = {prev_layer_vars[0]} * w{wi};")

    # Remaining neurons
    for j, var in enumerate(prev_layer_vars[1:], start=1):
        wi = w_counter - m + j
        code.append(f"tmp = {var} * w{wi};")
        code.append(f"y = y + tmp;")

    # Add final bias
    code.append(f"y = y + b{b_counter-1};")

    code.append("")
    code.append("y = y + gauss(0., 0.1);")

    return "\n".join(code)


def generate_bnn_code_observe(n, m):
    """
    Generate DSL code for a fully-connected Bayesian Neural Network
    with n hidden layers and m neurons per hidden layer.

    - One multiplication per line
    - Multiple additions allowed in one line
    - Reuses a single `tmp` variable for extra multiplications
    - Logical constraints via `observe(a_i > 0)`
    - Gaussian likelihood noise at output
    """

    code = []
    w_counter = 1
    b_counter = 1

    # --- Declare weights and biases ---
    for layer in range(n):
        for i in range(m):  # each neuron in this layer
            for j in range(m if layer > 0 else 1):  # inputs: m from prev layer, or 1 from x
                code.append(f"w{w_counter} = gm([1.], [_muw{w_counter}], [_sigmaw{w_counter}]);")
                w_counter += 1
            code.append(f"b{b_counter} = gm([1.], [_mub{b_counter}], [_sigmab{b_counter}]);")
            b_counter += 1

    # Output layer weights + bias
    for j in range(m):
        code.append(f"w{w_counter} = gm([1.], [_muw{w_counter}], [_sigmaw{w_counter}]);")
        w_counter += 1
    code.append(f"b{b_counter} = gm([1.], [_mub{b_counter}], [_sigmab{b_counter}]);")
    b_counter += 1
    code.append("")

    # --- Forward pass ---
    prev_layer_vars = ["_x"]

    for layer in range(n):
        layer_vars = []
        for i in range(m):
            a_var = f"a{layer}{i}"

            # First input
            wi = (layer * m * len(prev_layer_vars)) + (i * len(prev_layer_vars)) + 1
            code.append(f"{a_var} = {prev_layer_vars[0]} * w{wi};")

            # Remaining inputs -> reuse tmp
            for j, prev in enumerate(prev_layer_vars[1:], start=1):
                wi = (layer * m * len(prev_layer_vars)) + (i * len(prev_layer_vars)) + j + 1
                code.append(f"tmp = {prev} * w{wi};")
                code.append(f"{a_var} = {a_var} + tmp;")

            # Add bias
            bi = (layer * m) + i + 1
            code.append(f"{a_var} = {a_var} + b{bi};")

            # Logical constraint instead of noisy ReLU
            code.append(f"observe({a_var} > 0);")
            code.append("")

            layer_vars.append(a_var)
        prev_layer_vars = layer_vars

    # --- Output layer ---
    # First multiplication
    wi = w_counter - m
    code.append(f"y = {prev_layer_vars[0]} * w{wi};")

    # Remaining neurons
    for j, var in enumerate(prev_layer_vars[1:], start=1):
        wi = w_counter - m + j
        code.append(f"tmp = {var} * w{wi};")
        code.append(f"y = y + tmp;")

    # Add final bias
    code.append(f"y = y + b{b_counter-1};")

    code.append("")
    code.append("y = y + gauss(0., 0.1);")

    return "\n".join(code)


def optimize(params, cfg, X, Y, batch_size = 20, steps=5000, lr=0.001):
    """
    Optimize the BNN parameters using Adam optimizer.

    Args:
        params (dict): dictionary of BNN parameters to optimize
        loss_fn (function): loss function that takes (y_true, dist) and returns a scalar loss
        Y (torch.Tensor): true output values
        cfg: SOGA configuration object
        steps (int): number of optimization steps
        lr (float): learning rate for Adam optimizer
    """
    for key, value in params.items():
        params[key] = torch.tensor(value, requires_grad=True)    

    optimizer = torch.optim.Adam([params[key] for key in params.keys()], lr)
    total_start = time.time()

    for i in range(steps):
        optimizer.zero_grad()  # Reset gradients
        loss = 0
        for j in range(batch_size):
            sampled_index = np.random.randint(0, len(Y.squeeze([-1,1])))
            yj = Y.squeeze([1])[sampled_index].to(torch.float64)
            xj = X.squeeze([-1,1])[sampled_index]
            params['x'] = xj.requires_grad_(False)

            current_dist = start_SOGA(cfg, params, pruning='ranking')

            loss += neg_log_likelihood_one(yj, current_dist)

        # Backpropagate
        loss.backward(retain_graph=True)
        optimizer.step()

        # if the paramters contains word 'sigma' (like sigmax) are smaller than 0, set them to e-10
        for key in params.keys():
            if 'sigma' in key and params[key].item() < 1e-6:
                params[key] = torch.tensor(1e-6, requires_grad=True)

        # Print progress
        if i % int(steps/10) == 0:
            out = ''
            
            out = out + f" loss: {loss.item()}"

            for key in params.keys():
                out = out + key + ': ' + str(params[key].item()) + ' '

            print(out)

    total_end = time.time()

    print('Optimization performed in ', round(total_end-total_start, 3))

    return params


def make_dataset(func_id: int, N: int, N_test: int, sigma: float):

    # Choose input range depending on function
    if func_id == 1:  # cos(2x) + sin(x)
        x_min, x_max = 0, 3*np.pi
    elif func_id == 2:  # 0.1x^2
        x_min, x_max = -10, 10
    elif func_id == 3:  # -(1+x)sin(1.2x)
        x_min, x_max = -5, 5
    elif func_id == 4:  # MLP
        x_min, x_max = -3, 3
    elif func_id == 5:
        x_min, x_max = -4, 4
    else:
        raise ValueError("func_id must be 1,2,3,4, or 5")

    # Training and test inputs
    X = np.linspace(x_min, x_max, N)[:, None]
    X_test = np.linspace(x_min, x_max, N_test)[:, None]

    # Compute noiseless function values
    if func_id == 1:
        f_train = np.cos(2*X) + np.sin(X)
        f_test  = np.cos(2*X_test) + np.sin(X_test)

    elif func_id == 2:
        f_train = 0.1 * X**2
        f_test  = 0.1 * X_test**2

    elif func_id == 3:
        f_train = -(1+X) * np.sin(1.2*X)
        f_test  = -(1+X_test) * np.sin(1.2*X_test)

    elif func_id == 4:
        # Random MLP with weights ~ N(0, I)
        class RandomMLP(nn.Module):
            def __init__(self, in_dim=1, hidden=10, out_dim=1):
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden)
                self.fc2 = nn.Linear(hidden, hidden)
                self.fc3 = nn.Linear(hidden, out_dim)
                # sample weights from N(0,1)
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0., std=1.0)
                        nn.init.normal_(m.bias, mean=0., std=1.0)
            def forward(self, x):
                x = torch.tanh(self.fc1(x))
                x = torch.tanh(self.fc2(x))
                return self.fc3(x)

        mlp = RandomMLP()
        f_train = mlp(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        f_test  = mlp(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

    elif func_id == 5:
        # y = -x^4 +3x^2 +1
        f_train = -X**4 + 3*X**2 + 1
        f_test  = -X_test**4 + 3*X_test**2 + 1


    # Add Gaussian noise
    Y = f_train + sigma * np.random.randn(N, 1)
    Y_test = f_test + sigma * np.random.randn(N_test, 1)

    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    return X, Y, X_test, Y_test