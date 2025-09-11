import torch
import torch.nn.functional as F
import pyro
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions import constraints
import pyro.distributions as dist
import matplotlib.pyplot as plt

from utils import nonlin   # your custom nonlinearity (e.g. ReLU/tanh)


# --------- MODEL ---------
def model(X, Y, n: int, m: int, D_Y: int = 1):
    """
    Bayesian Neural Network model with n hidden layers of size m.
    """
    D_X = 1
    N = X.shape[0]

    z = X  # input

    # Hidden layers
    in_dim = D_X
    for i in range(1, n + 1):
        # Sample weights and biases for layer i
        w = pyro.sample(f"w{i}", dist.Normal(torch.zeros(in_dim, m), torch.ones(in_dim, m)).to_event(2))
        b = pyro.sample(f"b{i}", dist.Normal(torch.zeros(m), torch.ones(m)).to_event(1))

        # Expand for batch multiplication
        w_broadcast = w.expand(N, -1, -1)   # (N, in_dim, m)
        b_broadcast = b.unsqueeze(0).unsqueeze(0).expand(N, 1, m)  # (N, 1, m)

        # Linear transform + nonlinearity
        z = nonlin(torch.bmm(z, w_broadcast) + b_broadcast)  # (N, 1, m)

        in_dim = m  # for next layer

    # Final output layer
    w_out = pyro.sample("w_out", dist.Normal(torch.zeros(in_dim, D_Y), torch.ones(in_dim, D_Y)).to_event(2))
    b_out = pyro.sample("b_out", dist.Normal(torch.zeros(D_Y), torch.ones(D_Y)).to_event(1))

    b_out_broadcast = b_out.unsqueeze(0).unsqueeze(0).expand(N, 1, D_Y)
    z_out = torch.bmm(z, w_out.unsqueeze(0).expand(N, -1, -1)) + b_out_broadcast  # (N,1,D_Y)

    sigma_obs = torch.tensor(0.1)

    with pyro.plate("data", N):
        pyro.sample("Y", dist.Normal(z_out.squeeze(-1), sigma_obs).to_event(1), obs=Y.squeeze(-1))


# --------- GUIDE ---------
def guide(X, Y, n: int, m: int, D_Y: int = 1):
    """
    Variational distribution for BNN with n hidden layers of size m.
    """
    D_X = 1
    in_dim = D_X

    # Hidden layers
    for i in range(1, n + 1):
        # Variational params
        w_loc = pyro.param(f"w{i}_loc", torch.zeros(in_dim, m))
        w_scale = pyro.param(f"w{i}_scale", torch.ones(in_dim, m), constraint=constraints.positive)

        b_loc = pyro.param(f"b{i}_loc", torch.zeros(m))
        b_scale = pyro.param(f"b{i}_scale", torch.ones(m), constraint=constraints.positive)

        # Sample from variational distribution
        pyro.sample(f"w{i}", dist.Normal(w_loc, w_scale).to_event(2))
        pyro.sample(f"b{i}", dist.Normal(b_loc, b_scale).to_event(1))

        in_dim = m

    # Final layer variational params
    w_out_loc = pyro.param("w_out_loc", torch.zeros(in_dim, D_Y))
    w_out_scale = pyro.param("w_out_scale", torch.ones(in_dim, D_Y), constraint=constraints.positive)

    b_out_loc = pyro.param("b_out_loc", torch.zeros(D_Y))
    b_out_scale = pyro.param("b_out_scale", torch.ones(D_Y), constraint=constraints.positive)

    pyro.sample("w_out", dist.Normal(w_out_loc, w_out_scale).to_event(2))
    pyro.sample("b_out", dist.Normal(b_out_loc, b_out_scale).to_event(1))


# --------- TRAINING ---------
def train_vi(model, guide, X, Y, n, m, num_steps=10000, lr=0.01):
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(num_steps):
        loss = svi.step(X, Y, n, m)
        losses.append(loss)
        if step % max(1, num_steps // 10) == 0:
            print(f"Step {step} - Loss: {loss:.4f}")

    print("Training complete!")

    # Plot loss curve, after the first 10% of steops to avoid large initial values
    losses = losses[int(0.1 * num_steps):]
    plt.figure(figsize=(6,4))
    plt.plot(losses, label="ELBO Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    return losses


# --------- PREDICTION ---------
def predict_vi(X_test, params, n, m, D_Y=1):
    """
    Use variational means (loc) for prediction.
    """
    D_X = 1
    z = X_test
    in_dim = D_X

    # Hidden layers
    for i in range(1, n + 1):
        w_loc = params(f"w{i}_loc")
        b_loc = params(f"b{i}_loc")
        z = F.relu(torch.matmul(z, w_loc) + b_loc)  # shape: (N, m)
        in_dim = m

    # Output layer
    w_out_loc = params("w_out_loc")
    b_out_loc = params("b_out_loc")

    z_out = torch.matmul(z, w_out_loc) + b_out_loc
    return z_out
