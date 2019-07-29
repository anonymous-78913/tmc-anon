import math
import torch as t
import torch.nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions import MultivariateNormal as MVN


def logmeanexp(x, dim=0):
    max = x.max(dim=dim, keepdim=True)[0]
    return ((x-max).exp().mean(dim, keepdim=True).log()+max).squeeze(dim)

def logsumexp(x, dim=0):
    max = x.max(dim=dim, keepdim=True)[0]
    return ((x-max).exp().sum(dim, keepdim=True).log()+max).squeeze(dim)

def logmmmeanexp(X, Y):
    x = X.max(dim=1, keepdim=True)[0]
    y = Y.max(dim=0, keepdim=True)[0]
    X = X - x
    Y = Y - y
    return x + y + t.mm(X.exp(), Y.exp()).log() - t.log(t.ones((), device=x.device)*X.size(1))


class P(nn.Module):
    def __init__(self, Pw, PzGw, PxGz):
        super().__init__()
        self.Pw = Pw
        self.PzGw = PzGw
        self.PxGz = PxGz

    def sample(self, N):
        w = self.Pw().sample()
        z = self.PzGw(w).sample(sample_shape=t.Size([N]))
        x = self.PxGz(z).sample()
        return (x, (w.unsqueeze(-1), z))

    def log_prob(self, xwz):
        x, (w, z) = xwz
        logPw   = self.Pw().log_prob(w)
        logPzGw = self.PzGw(w).log_prob(z)
        logPxGz = self.PxGz(z).log_prob(x)
        return logPw.sum(-1) + logPzGw.sum(-1) + logPxGz.sum(-1)

class Q(nn.Module):
    def __init__(self, Qw, Qz):
        super().__init__()
        self.Qw = Qw
        self.Qz = Qz

    def sample(self, N, sample_shape=t.Size([])):
        w = self.Qw().sample(sample_shape=sample_shape)
        z = self.Qz().sample(sample_shape=t.Size([*sample_shape, N]))
        return (w.unsqueeze(-1), z)

    def log_prob(self, wz):
        w, z = wz
        logQw = self.Qw().log_prob(w)
        logQz = self.Qz().log_prob(z)
        return logQw.sum(-1) + logQz.sum(-1)

class ParamNormal(nn.Module):
    def __init__(self, sample_shape, scale=1.):
        super().__init__()
        self.loc = nn.Parameter(t.zeros(sample_shape))
        self.log_scale = nn.Parameter(math.log(scale)*t.ones(sample_shape))

    def forward(self):
        return Normal(self.loc, self.log_scale.exp())

class LinearNormal(nn.Module):
    def __init__(self, sample_shape=t.Size([]), scale=1.):
        super().__init__()
        self.log_scale = nn.Parameter(math.log(scale)*t.ones(sample_shape))

    def forward(self, input):
        return Normal(input, self.log_scale.exp())

def pqx(N, sw, sz, sx):
    p = P(ParamNormal((), scale=sw), LinearNormal((), scale=sz), LinearNormal((), scale=sx))
    x, _ = p.sample(N)
    x = x.cuda()

    q = Q(ParamNormal((), scale=sw), ParamNormal((), scale=math.sqrt(sw**2+sz**2)))
    #(w, z) = q.sample(t.Size([3]))
    return (p, q, x)


class VAE(nn.Module):
    """
    Usual single/multi-sample VAE
    """
    def __init__(self, p, q, K):
        super().__init__()
        self.p = p
        self.q = q
        self.K = K

    def forward(self, x):
        wz = self.q.sample(x.size(0), sample_shape=t.Size([self.K]))
        elbo = self.p.log_prob((x, wz)) - self.q.log_prob(wz)
        lme = logmeanexp(elbo)
        return lme

    def train(self, x):
        opt = t.optim.Adam(q.parameters())
        for i in range(100):
            #opt.zero_grad()
            obj = self(x)
            #(-obj).backward()
            #opt.step()
            print(obj)

class TMC(nn.Module):
    def __init__(self, p, q, Kw, Kz=None):
        super().__init__()
        self.p = p
        self.q = q
        if Kz is None:
            Kz = Kw
        self.Kw = Kw
        self.Kz = Kz

    def train(self, x):
        opt = t.optim.Adam(q.parameters())
        for i in range(100):
            #opt.zero_grad()
            obj = self(x)
            #(-obj).backward()
            #opt.step()
            print(obj)

class TMC(TMC):
    def forward(self, x):
        w  = self.q.Qw().sample(sample_shape=t.Size([self.Kw, 1, 1]))
        z  = self.q.Qz().sample(sample_shape=t.Size([self.Kz, x.size(0)]))
        fw = self.p.Pw().log_prob(w) - self.q.Qw().log_prob(w)
        fz = self.p.PzGw(w).log_prob(z) - self.q.Qz().log_prob(z)
        fx = self.p.PxGz(z).log_prob(x)
        f_int_z = logmeanexp(fz + fx, -2)
        f_int_z = f_int_z.sum(-1) + fw.view(-1)
        f_int_w = logmeanexp(f_int_z)

        return f_int_w#.sum(0)

class TMC_Shared(TMC):
    def forward(self, x):
        w  = self.q.Qw().sample(sample_shape=t.Size([self.Kw]))
        z  = self.q.Qz().sample(sample_shape=t.Size([self.Kz]))
        fw = self.p.Pw().log_prob(w) - self.q.Qw().log_prob(w)

        fz = self.p.PzGw(w.unsqueeze(1)).log_prob(z) - self.q.Qz().log_prob(z)
        fx = self.p.PxGz(z.unsqueeze(1)).log_prob(x)
        #f_int_z = logmeanexp(fz + fx, -2)
        f_int_z = logmmmeanexp(fz, fx)
        f_int_z = f_int_z.sum(-1) + fw.view(-1)
        f_int_w = logmeanexp(f_int_z)

        return f_int_w#.sum(0)

class SMC(nn.Module):
    def __init__(self, p, q, K):
        super().__init__()
        self.p = p
        self.q = q
        self.K = K

    def forward(self, x):
        w = self.p.Pw().sample(sample_shape=t.Size([self.K]))
        z = self.q.Qz().sample(sample_shape=t.Size([self.K, x.size(0)]))
        #print(w[0:3])
        #print(z[0, 0:3])
        log_Qz = self.q.Qz().log_prob(z)
        log_PxGz = self.p.PxGz(z).log_prob(x)
        #Algo:
        #  Resample on each step.
        #  Just need to resample w, as we just use z's to compute the bootstrap marginal likelihood estimator.
        res = []
        for i in range(x.size(0)):
            log_ws = self.p.PzGw(w).log_prob(z[:, i]) + log_PxGz[:, i] - log_Qz[:, i]
            res.append(logmeanexp(log_ws))
            dist = Categorical(logits=log_ws)
            resample_idx = dist.sample(sample_shape=t.Size([self.K]))
            w = w[resample_idx]
        return sum(res)

class GRT(nn.Module):
    def __init__(self, p, q, K):
        super().__init__()
        self.p = p
        self.q = q
        self.K = K

    def forward(self, x):
        assert 1 == len(x.size())
        N = x.size(0)
        sw = self.p.Pw.log_scale.exp()
        sz = self.p.PzGw.log_scale.exp()
        sx = self.p.PxGz.log_scale.exp()
        dist = MVN(t.zeros(N, device=x.device), sw**2 * t.ones(N, N, device=x.device) + (sz**2+sx**2)*t.eye(N, device=x.device))
        return dist.log_prob(x)
