#Notes on numerical stability:
# Get rid of floor on softplus?
# BatchNorm

from collections import OrderedDict
import os
import sys
import copy
import math
import torch as t
import pandas as pd
from torch import nn, optim, distributions
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

output_filename = sys.argv[1]
#Filename format: iwa|tmc_std|stl|drg for standard | sticking the landing | DReGs
basename = os.path.basename(output_filename)
vae_typ = basename[:3]
obj_typ = basename[4:7]
train_typ = basename[8:11]
#print(obj_typ)
#print(train_typ)

def std_nonlin(x):
    return 0.01 + F.softplus(x)
def weight_norm(x):
    return nn.utils.weight_norm(x)


batch_size = 128
folder = os.path.basename(os.path.dirname(output_filename))
epochs = 20 if folder == "time" else 1200
cuda = True
seed = 1
log_interval = 10

t.manual_seed(seed)

device = t.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = t.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = t.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

def detach_module(input):
    output = copy.copy(input)
    # Detach parameters
    output._parameters = OrderedDict()
    for key in input._parameters:
        output._parameters[key] = input._parameters[key].detach()

    # Recursively detach submodules
    output._modules = OrderedDict()
    for key in input._modules:
        output._modules[key] = detach_module(input._modules[key])
    return output

    
        

def logmmmeanexp(X, Y):
    x = X.max(dim=-1, keepdim=True)[0]
    y = Y.max(dim=-2, keepdim=True)[0]
    X = (X - x)#.double()
    Y = (Y - y)#.double()
    return x + y + t.matmul(X.exp(), Y.exp()).log().float() - math.log(X.shape[-1])#t.log(t.ones((), device=x.device)*X.size(-1))

class Normal(distributions.Normal):
    def log_prob(self, x):
        return super().log_prob(x.view(x.shape[0], 1, x.shape[1], x.shape[3])).sum(-1)

    def rsample_log_prob(self):
        x = self.rsample()
        return x, self.log_prob(x)

class Bernoulli(distributions.Bernoulli):
    """
    Hack to ensure that sampling from the VAE outputs means
    """
    def rsample(self):
        return self.mean

    def log_prob(self, x):
        return super().log_prob(x).sum(-1)

class PNorm(nn.Module):
    """
    A parameterised Gaussian distribution used as the inital state
    for a VAE.
    
    Takes as input a batch-size, and returns a Normal distribution.
    """
    def __init__(self, features):
        super().__init__()
        self.mean    = nn.Parameter(t.zeros(features))
        self.trans_std = nn.Parameter(t.zeros(features))

    def forward(self, batch_size):
        return Normal(
                self.mean.expand(batch_size, -1), 
                #self.log_std.exp().expand(batch_size, -1)
                #F.softplus(self.log_std).expand(batch_size, -1)
                std_nonlin(self.trans_std).expand(batch_size, -1)
            )

class NormMLP(nn.Module):
    """
    Random MLP
    A deep neural network that produces a single, factorised 
    Gaussian distribution as output.
    """
    def __init__(self, in_features, out_features, inter_features):
        super().__init__()
        self.det = nn.Sequential(
            weight_norm(nn.Linear(in_features, inter_features)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Linear(inter_features, inter_features)),
            nn.LeakyReLU(0.1),
        )
        self.mean    = weight_norm(nn.Linear(inter_features, out_features))
        self.trans_std = weight_norm(nn.Linear(inter_features, out_features))
        self.det_reduction = weight_norm(nn.Linear(inter_features, out_features))
    
    def forward(self, input):
        x = self.det(input)
        mean = self.mean(x)
        #std  = F.softplus(self.log_std(x))#.exp()
        std  = std_nonlin(self.trans_std(x))
        return Normal(mean, std)

class DetNormMLP(NormMLP):
    def forward(self, input):
        x = self.det(input)
        mean = self.mean(x)
        #std  = F.softplus(self.log_std(x))#.exp()
        std  = std_nonlin(self.trans_std(x))
        return self.det_reduction(x), Normal(mean, std)

class BernMLP(nn.Module):
    """
    Random MLP
    A deep neural network that produces a single, factorised 
    Gaussian distribution as output.
    """
    def __init__(self, in_features, out_features, inter_features):
        super().__init__()
        self.det = nn.Sequential(
            weight_norm(nn.Linear(in_features, inter_features)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Linear(inter_features, inter_features)),
            nn.LeakyReLU(0.1),
            weight_norm(nn.Linear(inter_features, out_features)),
        )
    
    def forward(self, x):
        return Bernoulli(logits=self.det(x))

class CD(nn.Module):
    """
    CD stands for "conditional distribution"
    Consider RMLP, above.  Giving it an input returns a
    distribution that works in the usual fashion (rsample,
    log_prob and rsample_log_prob work as expected).

    CD allows you to do the same thing with more complicated
    models, by wrapping up e.g. an RandomSequential with its input
    """
    def __init__(self, dist, input):
        super().__init__()
        self.dist = dist
        self.input = input

    def rsample(self):
        return self.dist._rsample(self.input)

    def log_prob(self, zs):
        return self.dist._log_prob(self.input, zs)

    def rsample_log_prob(self):
        return self.dist._rsample_log_prob(self.input)


class FacRandomSequential(nn.Sequential):
    """
    Transforms input into a series of factorised distributions
    """
    def forward(self, input):
        mods = list(self._modules.values())
        z = input
        dists = []
        for i in range(len(mods)):
            z, dist = mods[i](z)
            dists.append(dist)
        return FactorisedDist(dists)

class FactorisedDist(nn.Module):
    """
    Takes a list of distributions and allows e.g. sampling from them 
    """
    def __init__(self, dists):
        super().__init__()
        self.dists = dists

    def rsample(self):
        return [d.rsample() for d in self.dists]

    def log_prob(self, zs):
        return [d.log_prob(z) for (d, z) in zip(self.dists, zs)]

    def rsample_log_prob(self):
        zs = self.rsample()
        lps = self.log_prob(zs)
        return zs, lps

class RandomSequential(nn.Sequential):
    """
    Has multiple layers
    Each layer returns a distriubtion (e.g. a Gaussian)
    Input to subsequent layers is a sample from the distribution
    """
    def _rsample(self, input):
        mods = list(self._modules.values())
        zs = [input]
        for i in range(len(mods)):
            zs.append(mods[i](zs[i]).rsample())
        return zs[1:]

    def _rsample_log_prob(self, input):
        mods = list(self._modules.values())
        zs = [input]
        lps = []
        for i in range(len(mods)):
            zs.append(mods[i](zs[i]).rsample())
            lps.append(mods[i](zs[i]).log_prob(zs[i+1]))
        return zs[1:], lps

    def _log_prob(self, input, zs):
        mods = list(self._modules.values())
        zs = [input, *zs]
        lps = []
        for i in range(len(mods)):
            lps.append(mods[i](zs[i]).log_prob(zs[i+1]))
        return lps

    def forward(self, input):
        return CD(self, input)


class VAE(nn.Module):

    def rsample(self, batch):
        return self.P(batch).rsample()[-1]

    def elbo(self, input):
        input = input.view(-1, 1, 1, 784)
        input = input.expand(-1, self.Nsamples, 1, -1)
        Q = self.Q(input)
        zs = Q.rsample()
        logQs = detach_module(Q).log_prob(zs)
        logPs = self.P(1).log_prob([*zs[::-1], input])

        logQs = [t.diagonal(lq, dim1=-1, dim2=-2) for lq in logQs]
        logPs = [
            logPs[0].view(logPs[0].shape[0], self.Nsamples),
            *[t.diagonal(lp, dim1=-1, dim2=-2) for lp in logPs[1:-1]],
            logPs[-1].view(logPs[-1].shape[0], self.Nsamples),
        ]
        logQ = sum(logQs)
        logP = sum(logPs)
        #obj = -logP.sum() / self.Nsamples
        obj = -(logP - logQ).sum() / self.Nsamples
        return obj, obj

    def iwae(self, input):
        input = input.view(-1, 1, 1, 784)
        input = input.expand(-1, self.Nsamples, 1, -1)
        #zs = self.Q(input).rsample()
        if train_typ=='std':
            zs, logQs = self.Q(input).rsample_log_prob()
        else:
            zs = self.Q(input).rsample()
            logQs = detach_module(self.Q)(input).log_prob(zs)
        logPs = self.P(1).log_prob([*zs[::-1], input])

        logQs = [t.diagonal(lq, dim1=-1, dim2=-2) for lq in logQs]
        logPs = [
            logPs[0].view(logPs[0].shape[0], self.Nsamples),
            *[t.diagonal(lp, dim1=-1, dim2=-2) for lp in logPs[1:-1]],
            logPs[-1].view(logPs[-1].shape[0], self.Nsamples),
        ]
        logQ = sum(logQs)
        logP = sum(logPs)

        logw = logP - logQ

        log_sum_w  = logw.logsumexp(1, keepdim=True)
        P_obj = -(log_sum_w - math.log(self.Nsamples)).sum()

        #Standard DReG
        if train_typ=="drg":
            log_sum_w2 = (2*logw).logsumexp(1, keepdim=True)
            norm_w2 = (2*(logw - log_sum_w)).exp().detach()
            Q_obj = -(norm_w2 * logw).sum()
        else:
            Q_obj = 0.

        # DReG (my version)
        #Q_obj = -0.5*(log_sum_w2 * (log_sum_w2 - 2*log_sum_w).exp().detach()).sum()
        return P_obj, Q_obj

    def tmc(self, input):
        input = input.view(-1, 1, 1, 784)
        input = input.expand(-1, self.Nsamples, 1, -1)
        #zs = self.Q(input).rsample()
        if train_typ=='std':
            zs, logQs = self.Q(input).rsample_log_prob()
        else:
            zs = self.Q(input).rsample()
            logQs = detach_module(self.Q)(input).log_prob(zs)
        logPs = self.P(1).log_prob([*zs[::-1], input])

        #[batch, dist, sample]
        logQs = [(lq.logsumexp(1, keepdim=True) - math.log(lq.shape[1])) for lq in logQs[::-1]]

        log_sum_w = logPs[0] - logQs[0]
        for i in range(1, len(logQs)):
            log_sum_w = logmmmeanexp(log_sum_w, logPs[i] - logQs[i])
        log_sum_w = logmmmeanexp(log_sum_w, logPs[-1])

        if train_typ=="drg":
            log_sum_w2 = 2*(logPs[0] - logQs[0])
            for i in range(1, len(logQs)):
                log_sum_w2 = logmmmeanexp(log_sum_w2, 2*(logPs[i] - logQs[i]))
            log_sum_w2 = logmmmeanexp(log_sum_w2, 2*logPs[-1])
            Q_obj = -0.5*(log_sum_w2 * (log_sum_w2 - 2*log_sum_w).exp().detach()).sum()
        else:
            Q_obj = 0.

        P_obj = -(log_sum_w).sum()

        return P_obj, Q_obj

    def rws(self):
        zs = self.P(batch_size).rsample()
        zs = [z.view(batch_size, 1, 1, -1).detach() for z in zs]
        #x = zs[-1]
        #zs = 
        logQs = self.Q(zs[-1]).log_prob(zs[:-1][::-1])
        return -sum(lq.sum() for lq in logQs)


class VAE_Fac(VAE):
    def __init__(self, Nsamples):
        super().__init__()
        self.Nsamples = Nsamples
        self.Q = FacRandomSequential(
            DetNormMLP(784, 64, 2*64),
            DetNormMLP( 64, 32, 2*32),
            DetNormMLP( 32, 16, 2*16),
            DetNormMLP( 16,  8,  2*8),
            DetNormMLP(  8,  4,  2*4),
        )

        self.P = RandomSequential(
            PNorm(4), 
            NormMLP( 4,   8,  2*4),
            NormMLP( 8,  16,  2*8),
            NormMLP(16,  32, 2*16),
            NormMLP(32,  64, 2*32),
            BernMLP(64, 784, 2*64)
        )

class VAE_NonFacSmall(VAE):
    def __init__(self, Nsamples):
        super().__init__()
        self.Nsamples = Nsamples
        self.Q = RandomSequential(
            NormMLP(784, 64, 2*64),
            NormMLP( 64, 32, 2*32),
            NormMLP( 32, 16, 2*16),
            NormMLP( 16,  8,  2*8),
            NormMLP(  8,  4,  2*4),
        )

        self.P = RandomSequential(
            PNorm(4), 
            NormMLP( 4,   8,  2*4),
            NormMLP( 8,  16,  2*8),
            NormMLP(16,  32, 2*16),
            NormMLP(32,  64, 2*32),
            BernMLP(64, 784, 2*64)
        )

class VAE_NonFacLarge(VAE):
    def __init__(self, Nsamples):
        super().__init__()
        self.Nsamples = Nsamples
        self.Q = RandomSequential(
            NormMLP(784, 64, 8*64),
            NormMLP( 64, 32, 8*32),
            NormMLP( 32, 16, 8*16),
            NormMLP( 16,  8,  8*8),
            NormMLP(  8,  4,  8*4),
        )

        self.P = RandomSequential(
            PNorm(4), 
            NormMLP( 4,   8,  2*4),
            NormMLP( 8,  16,  2*8),
            NormMLP(16,  32, 2*16),
            NormMLP(32,  64, 2*32),
            BernMLP(64, 784, 2*64)
        )

model = {
    'fac' : VAE_Fac,
    'nfs' : VAE_NonFacSmall,
    'nfl' : VAE_NonFacLarge,
}[vae_typ](20).to(device)
#VAE(40).to(device)
model.forward = {
    'iwa' : model.iwae,
    'tmc' : model.tmc,
}[obj_typ]

opt = optim.Adam(model.parameters(), lr=1e-3)
P_opt = optim.Adam(model.P.parameters(), lr=1e-3)
Q_opt = optim.Adam(model.Q.parameters(), lr=1e-3)

def train_unified(epoch, obj=model.__call__):
    model.train()
    train_loss = 0

    start = t.cuda.Event(enable_timing=True)
    end = t.cuda.Event(enable_timing=True)
    start.record()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        loss, _ = obj(data)

        opt.zero_grad()
        loss.backward()
        #t.nn.utils.clip_grad_value_(model.parameters(),10.)
        opt.step()

    end.record()
    t.cuda.synchronize()
    return start.elapsed_time(end)/1000

def train_separate(epoch, obj=model.__call__):
    model.train()
    train_loss = 0

    start = t.cuda.Event(enable_timing=True)
    end = t.cuda.Event(enable_timing=True)
    start.record()

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        P_loss, Q_loss = obj(data)
        loss = P_loss

        P_opt.zero_grad()
        P_loss.backward(retain_graph=True)
        #t.nn.utils.clip_grad_value_(model.parameters(),10.)
        P_opt.step()
        Q_opt.zero_grad()
        Q_loss.backward()
        #t.nn.utils.clip_grad_value_(model.parameters(),10.)
        Q_opt.step()

    end.record()
    t.cuda.synchronize()
    return start.elapsed_time(end)/1000

def train_rws(epoch, obj=model.__call__, mod=model):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        P_loss, _ = obj(data)

        P_opt.zero_grad()
        P_loss.backward()
        #t.nn.utils.clip_grad_value_(model.parameters(),10.)
        P_opt.step()

        Q_loss = model.rws()
        Q_opt.zero_grad()
        Q_loss.backward()
        #t.nn.utils.clip_grad_value_(model.parameters(),10.)
        Q_opt.step()


train = {
    'std' : train_unified,
    'stl' : train_unified,
    'drg' : train_separate,
    'rws' : train_rws,
}[train_typ]


def test(obj=model.__call__):
    model.eval()
    test_loss = 0
    with t.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            test_loss += obj(data)[0].item()

    test_loss /= len(test_loader.dataset)
    #print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

if __name__ == "__main__":
    _epochs = []
    iwas = []
    tmcs = []
    times = []
    for epoch in range(1, epochs + 1):
        times.append(train(epoch))
        test()
        _epochs.append(epoch)
        iwas.append(test(model.iwae))
        tmcs.append(test(model.tmc))
        #with t.no_grad():
        #    sample = model.rsample(64)
        #    save_image(sample.view(64, 1, 28, 28),
        #               'vae/' + basename[:-4] +'/sample_' + str(epoch) + '.png')
    pd.DataFrame({
        'epoch' : _epochs,
        'iwae' : iwas,
        'tmc' : tmcs,
        'time' : times
    }).to_csv(output_filename)
