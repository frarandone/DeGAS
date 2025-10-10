import torch
import torch.distributions.constraints as constraints
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.infer import MCMC, NUTS
import pyro.distributions as dist
from time import time
import numpy as np
import matplotlib.pyplot as plt


def run_inference(model, guide, model_params, n_steps=1000, lr=0.05):
    # Setup the optimizer
    adam_params = {"lr": lr}
    optimizer = Adam(adam_params)
    
    # Setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    # Initialize the parameters
    pyro.clear_param_store()
    
    # Perform inference
    loss_list = []
    total_start = time()
    for step in range(n_steps):
        loss = svi.step(model_params)
        loss_list.append(loss)
        if step % int(n_steps/10) == 0:
            print(f"Step {step} : loss = {loss}")
    total_end = time()
    print('Inference performed in ', round(total_end-total_start, 3))
    
    return loss_list

def get_model_guide(program):
    if program == "bernoulli":
        return model_bernoulli, guide_bernoulli
    elif program == "burglary":
        return model_burglary, guide_burglary
    elif program == "clickgraph":
        return model_clickgraph, guide_clickgraph
    else:
        raise ValueError("Program not recognized")

def model_bernoulli(params):
    N, y = params
    # Prior for theta
    p = pyro.sample("p", dist.Uniform(0,1))

    # Likelihood
    with pyro.plate("data_plate", N):
        y = pyro.sample("y", dist.Bernoulli(p), obs=y)

def guide_bernoulli(params):
    p_map = pyro.param('p_map', torch.tensor(0.5))
    pyro.sample("p", dist.Delta(p_map))


def model_burglary(params):
    """
    params: torch.tensor of shape [N, 6]
            columns = [burglary, earthquake, alarm, maryWakes, phoneWorking, called]
    """
    N, data = params

    # Split columns
    burglary_obs     = data[:, 0]
    earthquake_obs   = data[:, 1]
    alarm_obs        = data[:, 2]
    maryWakes_obs    = data[:, 3]
    phoneWorking_obs = data[:, 4]
    called_obs       = data[:, 5]

    pb = pyro.sample("pb", dist.Beta(1., 1.))
    pe = pyro.sample("pe", dist.Beta(1., 1.))

    with pyro.plate("data", N):

        # --- Burglary ---
        burglary = pyro.sample("burglary",
                               dist.Bernoulli(pb),
                               obs=burglary_obs)

        # --- Earthquake ---
        earthquake = pyro.sample("earthquake",
                                 dist.Bernoulli(pe),
                                 obs=earthquake_obs)

        # --- Alarm ---
        alarm_prob = torch.where(
            (burglary.bool() | earthquake.bool()),
            torch.tensor(1.0),
            torch.tensor(0.0)
        )
        alarm = pyro.sample("alarm", dist.Bernoulli(alarm_prob), obs=alarm_obs)

        # --- Mary wakes ---
        mary_prob = torch.where(
            alarm.bool() & earthquake.bool(), torch.tensor(0.8),
            torch.where(alarm.bool(), torch.tensor(0.6), torch.tensor(0.2))
        )
        pyro.sample("maryWakes", dist.Bernoulli(mary_prob), obs=maryWakes_obs)

        # --- Phone working ---
        phone_prob = torch.where(earthquake.bool(), torch.tensor(0.7), torch.tensor(0.99))
        pyro.sample("phoneWorking", dist.Bernoulli(phone_prob), obs=phoneWorking_obs)

        # --- Called ---
        call_prob = torch.where(
            maryWakes_obs.bool() & phoneWorking_obs.bool(),
            torch.tensor(1.0),
            torch.tensor(0.0)
        )
        pyro.sample("called", dist.Bernoulli(call_prob), obs=called_obs)



def guide_burglary(params):
    # Define variational parameters (learnable)
    pb_map = pyro.param("pb_map", torch.tensor(0.5), constraint=dist.constraints.unit_interval)
    pe_map = pyro.param("pe_map", torch.tensor(0.5), constraint=dist.constraints.unit_interval)

    pyro.sample("pb", dist.Beta(pb_map * 10, (1 - pb_map) * 10))
    pyro.sample("pe", dist.Beta(pe_map * 10, (1 - pe_map) * 10))


def model_clickgraph(params):
    #observing only click0 and click1
    N, data = params
    p = pyro.sample("p", dist.Uniform(0,1))

    with pyro.plate("data", N):
        # Prior on similarity variable
        sim = pyro.sample("sim", dist.Bernoulli(p))

        # Priors on latent click probabilities
        beta1 = pyro.sample("beta1", dist.Uniform(0., 1.))
        # Conditional definition for beta2
        beta2_same = beta1
        beta2_diff = pyro.sample("beta2_diff", dist.Uniform(0., 1.))
        beta2 = sim * beta2_same + (1 - sim) * beta2_diff

        # Observed clicks
        click0 = pyro.sample("click0", dist.Bernoulli(beta1), obs=data[:, 0])
        click1 = pyro.sample("click1", dist.Bernoulli(beta2), obs=data[:, 1])


def guide_clickgraph(params):
    N, data = params
    # Variational posterior for global latent variable p
    p_map = pyro.param("p_map", torch.tensor(0.5), constraint=dist.constraints.unit_interval)
    pyro.sample("p", dist.Delta(p_map))
