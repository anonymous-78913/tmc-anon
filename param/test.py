from lvm import *
from torch.distributions import MultivariateNormal as MVN

t.manual_seed(1)
N = 100
sw = 1.
sz = 1.
sx = 0.1
p, q, x = pqx(N, sw=sw, sz=sz, sx=sx)

print(MVN(t.zeros(N), sw**2 * t.ones(N, N) + (sz**2+sx**2)*t.eye(N)).log_prob(x.cpu()))

            
t.manual_seed(1)
vae = VAE(p, q, 10000).cuda()
for i in range(10):
    print(vae(x))

iters = 100

tmc = TMC(p, q, 501, 502).cuda()
tmcs = []
#print()
for i in range(iters):
    t.manual_seed(i)
    res = tmc(x)
    tmcs.append(res.detach().cpu().numpy())
    #print(res)

smc = SMC(p, q, 500).cuda()
smcs = []
#print()
for i in range(iters):
    t.manual_seed(i)
    res = smc(x)
    smcs.append(res.detach().cpu().numpy())
    #print(res)

print()
import numpy as np
tmcs = np.array(tmcs)
smcs = np.array(smcs)
print(tmcs.mean())
print(smcs.mean())
print(tmcs.std())
print(smcs.std())
