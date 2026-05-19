"""Simulator replicating the behavior of src/BPMN/programs/debug.soga

Usage:
    python src/BPMN/debug_sim.py --n 10000

The script samples process completion time `t` following the SOGA model:
- tacts[i] = mu_i^2 + Normal(0, sigma_i^2) where sigma for activities is sqrt(10)
- p_j = p_j^2 + Normal(0, sigma_p^2) where sigma_p = sqrt(0.1)
- Gateway decisions are Bernoulli(p_j) (clipped to [0,1])

Returns samples and prints mean/std and saves samples to debug_samples.npy
"""
from __future__ import annotations
import argparse
try:
    import numpy as np
    _HAS_NUMPY = True
    default_rng = np.random.default_rng
    sqrt = np.sqrt
except Exception:
    import math, random
    _HAS_NUMPY = False
    class _SimpleRNG:
        def __init__(self, seed=None):
            self._rng = random.Random(seed)
        def uniform(self, a, b):
            return self._rng.random() * (b - a) + a
        def normal(self, loc, scale):
            u1 = self._rng.random()
            u2 = self._rng.random()
            z0 = math.sqrt(-2 * math.log(max(u1, 1e-15))) * math.cos(2 * math.pi * u2)
            return loc + z0 * scale
    def default_rng(seed=None):
        return _SimpleRNG(seed)
    def sqrt(x):
        return math.sqrt(x)
from typing import Dict

# default params chosen to mirror the notebook snippet (before sqrt step)
DEFAULT_PARAMS = {
    # these are the values used in the notebook BEFORE they apply `**0.5`;
    # the SOGA program expects `_mu` such that `_mu * _mu` gives the intended mean.
    # Here we accept the same notebook "raw" values and compute `_mu = sqrt(raw)`.
    'mu0': 60.,
    'mu1': 20.,
    'mu2': 20.,
    'mu3': 90.,
    'mu4': 120.,
    'mu5': 60.,
    'mu6': 90.,
    'p0': 0.9,
    'p1': 0.3,
    'p2': 0.3,
}

# noise variances inferred from the gm(...) calls in the SOGA file
ACT_NOISE_VAR = 10.0
P_NOISE_VAR = 0.1


def simulate_once(params: Dict[str, float], rng: np.random.Generator = None) -> float:
    """Simulate a single run of the `debug.soga` model and return total time `t`.

    params: keys 'mu0'..'mu6' and 'p0'..'p2' are the raw values as in the notebook.
            The function will compute `_mu = sqrt(raw)` to match the SOGA semantics.
    """
    if rng is None:
        rng = np.random.default_rng()

    # compute _mu values (the notebook does this: params[key] = params[key]**0.5)
    mus = [sqrt(params[f'mu{i}']) for i in range(7)]

    # sample activity durations: tacts[i] = _mu_i * _mu_i + Normal(0, ACT_NOISE_VAR)
    # note: _mu_i * _mu_i == original raw mu value, so mean duration = params['mui']
    tacts = [params[f'mu{i}'] + rng.normal(0.0, sqrt(ACT_NOISE_VAR)) for i in range(7)]

    # enforce positive durations like the SOGA `observe(tacts[i] > 0.)`
    tacts = [max(1e-8, x) for x in tacts]

    # sample gateway probabilities: p_j = _p_j * _p_j + Normal(0, P_NOISE_VAR)
    # where _p_j = sqrt(params['pj']) per same convention
    ps = []
    for j in range(3):
        raw = params[f'p{j}']
        _p = sqrt(raw)
        p = _p * _p + rng.normal(0.0, sqrt(P_NOISE_VAR))
        # clip to [0,1]
        p = min(1.0, max(0.0, p))
        ps.append(p)

    p0, p1, p2 = ps

    # begin execution following debug.soga logic
    t = 0.0
    endacts = [0.0] * 7

    endacts[0] = t + tacts[0]
    t = endacts[0]

    if rng.uniform(0.0, 1.0) < p0:
        branch0 = t
        branch1 = t

        if rng.uniform(0.0, 1.0) < p1:
            branch00 = branch0
            branch01 = branch0

            endacts[1] = branch00 + tacts[1]
            branch00 = endacts[1]

            endacts[2] = branch01 + tacts[2]
            branch01 = endacts[2]

            endacts[3] = branch01 + tacts[3]
            branch01 = endacts[3]

            # optional W_Call_after_offer
            if rng.uniform(0.0, 1.0) < p2:
                endacts[4] = branch01 + tacts[4]
                branch01 = endacts[4]

            # join left outer branch
            branch0 = max(branch00, branch01)

        else:
            endacts[5] = branch1 + tacts[5]
            branch1 = endacts[5]

        # join outer parallel
        t = max(branch0, branch1)
    else:
        # skip outer block
        pass

    endacts[6] = t + tacts[6]
    t = endacts[6]

    return float(t)


def run_samples(n: int, params: Dict[str, float], seed: int | None = None):
    rng = default_rng(seed)
    if _HAS_NUMPY:
        samples = np.empty(n, dtype=float)
        for i in range(n):
            samples[i] = simulate_once(params, rng)
        return samples
    else:
        samples = [simulate_once(params, rng) for _ in range(n)]
        return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10000, help='number of samples')
    parser.add_argument('--seed', type=int, default=0, help='rng seed')
    args = parser.parse_args()

    samples = run_samples(args.n, DEFAULT_PARAMS, seed=args.seed)

    if _HAS_NUMPY:
        print(f'samples: n={len(samples)} mean={samples.mean():.4f} std={samples.std():.4f}')
        np.save('debug_samples.npy', samples)
    else:
        import statistics
        mean = statistics.mean(samples)
        std = statistics.pstdev(samples)
        print(f'samples: n={len(samples)} mean={mean:.4f} std={std:.4f}')
        with open('debug_samples.txt', 'w') as f:
            for s in samples:
                f.write(str(s) + '\n')
