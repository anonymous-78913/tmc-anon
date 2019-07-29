import math
from timeit import default_timer as timer
import sys
import os
import re
import torch as t
from lvm import *

output_filename = sys.argv[1]
basename = os.path.basename(output_filename)

vae_smc_tmc = basename[:3]
VAE_SMC_TMC = {
    "vae" : VAE, 
    "smc" : SMC, 
    "tmc" : TMC,
    "grt" : GRT
}[vae_smc_tmc]
cpu_gpu = basename[4:7]
cpu_cuda = {
    "cpu" : "cpu", 
    "gpu" : "cuda"
}[cpu_gpu]

NK = re.findall('[0-9]+', basename)
N = int(NK[0])
K = int(NK[1])
print(N)
print(K)
#K = int(basename[8:-4])

iters = 100
#N = 500
sw = 1.
sz = 1.
sx = 1.

t.manual_seed(1)
p, q, x = pqx(N, sw=sw, sz=sz, sx=sx)
getattr(x, cpu_cuda)()

mod = VAE_SMC_TMC(p, q, K)
getattr(mod, cpu_cuda)()

if vae_smc_tmc in ["vae", "smc", "tmc"]:
    res = []
    start = timer()
    for i in range(iters):
        t.manual_seed(i)
        res.append(mod(x).detach().cpu().numpy())
    total_time = timer() - start

    mean_time = total_time / iters

    import numpy as np
    res = np.array(res)
    mean = res.mean()
    std = res.std()
    sterror = std / math.sqrt(iters)

elif vae_smc_tmc in ["grt"]:
    mean = mod(x).detach().cpu().numpy()
    std = 0.
    sterror = 0.
    mean_time = 0.

else:
    raise Exception()



import pandas as pd
pd.DataFrame({
    'time' : [mean_time],
    'K' : [K],
    'N' : [N],
    'mean' : [mean],
    'std' : [std],
    'sterror' : [sterror],
    'method' : [vae_smc_tmc],
    'cpu_gpu' : [cpu_gpu]
}).to_csv(output_filename, index=False)
