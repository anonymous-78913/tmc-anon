import torch as t
import torch.nn as nn
from torch.distributions import Normal, Categorical

def logmmmeanexp(X, Y):
    x = X.max(dim=1, keepdim=True)[0]
    y = Y.max(dim=0, keepdim=True)[0]
    X = X - x
    Y = Y - y
    return x + y + t.mm(X.exp(), Y.exp()).log() - t.log(t.ones((), device=x.device)*X.size(1))

class TMC(nn.Module):
    def __init__(self, T, res, log_std=0., like_std=1.):
        super().__init__()
        self.register_buffer('T', t.tensor(T))
        self.register_buffer('res', t.tensor(res))
        #self.res      = res
        self.means    = nn.Parameter(t.zeros(T))
        self.log_stds = nn.Parameter(t.sqrt((1.+t.arange(T).float())/T))#log_std*t.ones(T))
        self.like_std = like_std

    def logpqs(self, Ks):
        """
        Compute the series of tensors that we reduce over, by combining the universal generative model
        with the proposal propbabilities.
        """
        zs, log_qs = self.rsample_log_prob(Ks)

        log_ps = Normal(0., t.sqrt(1./self.T.float())).log_prob(zs[0])
        res = [(log_ps - log_qs[0]).unsqueeze(0)]

        for i in range(1, len(zs)):
            log_ps = Normal(zs[i-1].unsqueeze(1), t.sqrt(1/self.T.float())).log_prob(zs[i])
            res.append(log_ps - log_qs[i])

        res.append(Normal(zs[-1].unsqueeze(1), self.like_std).log_prob(self.res))

        return res

    def reduce(self, Ks):
        """
        Combine tensors
        """
        logpqs = self.logpqs(Ks)

        res = logpqs[0]
        for i in range(1, len(logpqs)):
            res = logmmmeanexp(res, logpqs[i])
        return res

class Fac(TMC):
    def rsample_log_prob(self, Ks):
        if isinstance(Ks, int):
            Ks = self.T.item() * [Ks]

        zs = []
        log_probs = []

        for i in range(len(Ks)):
            Q = Normal(self.means[i], self.log_stds[i].exp())
            z = Q.rsample(sample_shape=t.Size([Ks[i]]))
            zs.append(z)
            log_probs.append(Q.log_prob(z))

        return zs, log_probs

class NonFac(TMC):
    def rsample_log_prob(self, Ks):
        if isinstance(Ks, int):
            Ks = self.T.item() * [Ks]

        zs = []
        log_probs = []

        Q = Normal(0, t.sqrt(1/self.T.float()))
        z = Q.rsample(sample_shape=t.Size([Ks[0]]))
        zs.append(z)
        log_probs.append(Q.log_prob(z))

        for i in range(1, len(Ks)):
            z_prev = zs[-1]
            idx = Categorical(logits=t.zeros(Ks[i-1])).sample(sample_shape=t.Size([Ks[i]]))
            z = Normal(z_prev[idx], t.sqrt(1/self.T.float())).rsample()
            zs.append(z)

            Qs = Normal(z_prev, t.sqrt(1/self.T.float())).log_prob(z.unsqueeze(1)).exp().mean(1).log()
            log_probs.append(Qs)

        return zs, log_probs

