import math
import torch as t
import numpy as np
from torch.distributions import Normal, MultivariateNormal
from timeit import default_timer as timer

p = 0.01
Nx = 50#63
Nz = 50#64
sx = 0.1
sx2 = sx**2

t.manual_seed(-1)
probs = p*t.ones(Nz, Nx) + (1-p)*t.eye(Nz, Nx)
conn = t.rand(Nz, Nx) < probs
conn_f = conn.float()

z_gen = t.randn(Nz)
x_gen = z_gen @ conn_f + sx*t.randn(Nx)

def logmeanexp(x, dim, keepdim=False):
    return t.logsumexp(x, dim=(dim,), keepdim=keepdim) - math.log(x.shape[dim])

methods = []
Ks = []
seeds = []
ests = []
times = []

#### GT
gt_lp = MultivariateNormal(t.zeros(Nx), sx2*t.eye(Nx) + conn_f.t() @ conn_f).log_prob(x_gen)
print(gt_lp)

#### IWAE
def iwae(K, seed):
    t.manual_seed(seed)
    t0 = timer()

    Z = t.randn(K, Nz)
    est = logmeanexp(Normal(Z @ conn.float(), sx).log_prob(x_gen).sum(-1), dim=0).item()

    t1 = timer()

    methods.append("IWAE")
    Ks.append(K)
    seeds.append(seed)
    ests.append(est)
    times.append(t1-t0)

for K in [2**i for i in range(4, 25)]:
    print(K)
    for seed in range(3):
        iwae(K, seed)


def tmc(K, seed):
    t.manual_seed(seed)
    t0 = timer()

    #sample zs
    zs = [t.randn([*(i*[1]), K, *((Nz-i-1)*[1])]) for i in range(Nz)]
    
    #compute Nx factors
    lps = []
    #i indexes factors (related to data points)
    for i in range(Nx):
        mu = 0.
        for j in range(Nz):
            if conn[j, i]:
                mu = mu + zs[j]
        lps.append(Normal(mu, sx).log_prob(x_gen[i]))
    
    
    
    #lps not empty
    total = 0.
    while lps:
        #find the variable associated with fewest factors (but at least one factor)
        total_factors = np.zeros(Nz, dtype=np.int)
        for lp in lps:
            total_factors += (np.array(lp.shape) != 1)
        total_factors += 1000 * (total_factors == 0)
        j = np.argmin(total_factors)
    
        #combine all factors associated with that variable
        lp = t.zeros(Nz*[1])
        for i in range(len(lps))[::-1]:
            if 1 != lps[i].shape[j]:
                lp = lp + lps[i]
                del lps[i]
    
        #sum over variable, if you end up with a scalar, put it in total, otherwise, put it back in list
        vs = np.sum(np.array(lp.shape)!=1)
        if 0 == vs:
            total += lp
        elif 1 == vs:
            total += logmeanexp(lp, j)
        else:
            lps.append(logmeanexp(lp, j, keepdim=True))

    t1 = timer()

    methods.append("TMC")
    Ks.append(K)
    seeds.append(seed)
    ests.append(total.item())
    times.append(t1-t0)
    

for K in [2**i for i in range(2, 8)]:
    for seed in range(3):
        tmc(K, seed)


