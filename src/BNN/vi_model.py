import torch
import torch.nn.functional as F
import pyro.optim
from pyro.infer import SVI, Trace_ELBO
from pyro.distributions import constraints
import pyro.distributions as dist

from utils import nonlin

def model(X, Y, D_H, D_Y=1):
    D_X = 1  # Number of input features
    N = X.shape[0]  # Number of data points

    # Sample first layer weights and biases
    w1 = pyro.sample("w1", dist.Normal(torch.zeros(D_X, D_H), torch.ones(D_X, D_H)).to_event(2))  # Shape: (1, 1, 2)
    b1 = pyro.sample("b1", dist.Normal(torch.zeros(D_H), torch.ones(D_H)).to_event(1))  # Shape: (2,)

    #w2 = pyro.sample("w2", dist.Normal(torch.zeros(D_X, D_H), torch.ones(D_X, D_H)).to_event(2))  # Shape: (1, 1, 2)
    #b2 = pyro.sample("b2", dist.Normal(torch.zeros(D_H), torch.ones(D_H)).to_event(1))  # Shape: (2,)

    # Broadcast w1 and b1 to match the batch size
    w1_broadcasted = w1.expand(N, -1, -1)  # Shape: (10, 1, 2)
    b1_broadcasted = b1.unsqueeze(0).unsqueeze(0).expand(N, 1, D_H)  # Shape: (10, 1, 2)

    #w2_broadcasted = w2.expand(N, -1, -1)  # Shape: (10, 1, 2)
    #b2_broadcasted = b2.unsqueeze(0).unsqueeze(0).expand(N, 1, D_H)  # Shape: (10, 1, 2)

    # Perform batch matrix multiplication
    z1 = nonlin(torch.bmm(X, w1_broadcasted) + + b1_broadcasted)  # Shape: (10, 1, D_H)
    #z2 = nonlin(torch.bmm(X, w2_broadcasted) + b2_broadcasted)  # Shape: (10, 1, D_H)

    # Sample final layer weights and biases
    w3 = pyro.sample("w3", dist.Normal(torch.zeros(D_H, D_Y), torch.ones(D_H, D_Y)).to_event(2))  # Shape: (2, D_Y)
    #w4 = pyro.sample("w4", dist.Normal(torch.zeros(D_H, D_Y), torch.ones(D_H, D_Y)).to_event(2))  # Shape: (2, D_Y)
    
    b3 = pyro.sample("b3", dist.Normal(torch.zeros(D_Y), torch.ones(D_Y)).to_event(1))  # Shape: (D_Y,)

    # Broadcast b3 to match the batch size
    b3_broadcasted = b3.unsqueeze(0).unsqueeze(0).expand(N, 1, D_Y)  # Shape: (10, 1, D_Y)

    # Compute final output
    z3 = torch.bmm(z1, w3.unsqueeze(0).expand(N, -1, -1))  + b3_broadcasted  #+ torch.bmm(z2, w4.unsqueeze(0).expand(N, -1, -1)) # Shape: (10, 1, D_Y)

    # Observation noise (fixed for simplicity)
    sigma_obs = torch.tensor(0.1)

    # Use pyro.plate to handle batch dimensions
    with pyro.plate("data", N):
        pyro.sample("Y", dist.Normal(z3.squeeze(-1), sigma_obs).to_event(1), obs=Y.squeeze(-1))

# Define the guide function for VI
def guide(X, Y, D_H, D_Y=1):
    D_X = 1
    N = X.shape[0]  # Number of data points

    # Variational parameters for the first layer
    w1_loc = pyro.param("w1_loc", torch.zeros(D_X, D_H))
    w1_scale = pyro.param("w1_scale", torch.ones(D_X, D_H), constraint=constraints.positive)
    #w2_loc = pyro.param("w2_loc", torch.zeros(D_X, D_H))
    #w2_scale = pyro.param("w2_scale", torch.ones(D_X, D_H), constraint=constraints.positive)
    b1_loc = pyro.param("b1_loc", torch.zeros(D_H))
    b1_scale = pyro.param("b1_scale", torch.ones(D_H), constraint=constraints.positive)
    #b2_loc = pyro.param("b2_loc", torch.zeros(D_H))
    #b2_scale = pyro.param("b2_scale", torch.ones(D_H), constraint=constraints.positive)

    # Sample first layer weights and biases
    pyro.sample("w1", dist.Normal(w1_loc, w1_scale).to_event(2))
    pyro.sample("b1", dist.Normal(b1_loc, b1_scale).to_event(1))
    #pyro.sample("w2", dist.Normal(w2_loc, w2_scale).to_event(2))
    #pyro.sample("b2", dist.Normal(b2_loc, b2_scale).to_event(1))

    # Variational parameters for the final layer
    w3_loc = pyro.param("w3_loc", torch.zeros(D_H, D_Y))
    w3_scale = pyro.param("w3_scale", torch.ones(D_H, D_Y), constraint=constraints.positive)
    #w4_loc = pyro.param("w4_loc", torch.zeros(D_H, D_Y))
    #w4_scale = pyro.param("w4_scale", torch.ones(D_H, D_Y), constraint=constraints.positive)
    b3_loc = pyro.param("b3_loc", torch.zeros(D_Y))
    b3_scale = pyro.param("b3_scale", torch.ones(D_Y), constraint=constraints.positive)

    # Sample final layer weights and biases
    pyro.sample("w3", dist.Normal(w3_loc, w3_scale).to_event(2))
    #pyro.sample("w4", dist.Normal(w4_loc, w4_scale).to_event(2))
    pyro.sample("b3", dist.Normal(b3_loc, b3_scale).to_event(1))


# Function to train the BNN using Variational Inference
def train_vi(model, guide, X, Y, D_H, num_steps=10000, lr=0.01):
    pyro.clear_param_store()
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(num_steps):
        loss = svi.step(X, Y, D_H)
        losses.append(loss)
        if step % int(num_steps/10) == 0:
            print(f"Step {step} - Loss: {loss}")

    print("Training complete!")
    return losses

def predict_vi(X_test, params):
    # Retrieve learned variational parameters from Pyro's parameter store
    w1_loc = params("w1_loc")
    b1_loc = params("b1_loc")
    #w2_loc = params("w2_loc")
    #b2_loc = params("b2_loc")
    w3_loc = params("w3_loc")
    #w4_loc = params("w4_loc")
    b3_loc = params("b3_loc")

    # Compute the activations of the hidden layer
    z1 = F.relu(torch.matmul(X_test, w1_loc) + b1_loc)
    #z2 = F.relu(torch.matmul(X_test, w2_loc) + b2_loc)

    # Compute the final output
    z3 = torch.matmul(z1, w3_loc) + b3_loc #+ torch.matmul(z2, w4_loc)

    return z3