from typing import Dict
import torch
from torch import nn
from torch.autograd import grad
from torch.nn.utils import spectral_norm
from torch import optim
from typing import List, Union, Dict
from .util import DotDict

import numpy as np

from tqdm import tqdm

def get_steps(a, b, N):
    return np.geomspace(a, b, num=N)

class Generator(nn.Module):
    def __init__(self, x_dim, n_h=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(x_dim, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1)
        )
        self.x_dim = x_dim
    def forward(self, x):
        return self.enc(x)

class CD_EBM:
    def __init__(self,
                 x_dim: int, 
                 gen_kwargs: Dict, 
                 stochastic: bool = True,
                 std: float = 1.0, # variance of noise injected before SGLD
                 sgld_n_steps: int = 100,
                 sgld_kwargs: Dict = dict(a=0.001, b=0.0001),
                 device: Union[str, int] = 'cpu'):
        """
        Energy-based model trained with contrastive divergence.
        
        Args:
        -----
         x_dim: dimensionality of input data x
         gen_kwargs: kwargs for Generator class
         std: standard deviation for Gaussian noise that gets injected into x for
           negative sampling loss (right before sgld gets invoked). This seems to
           be necessary for model convergence.
         sgld_kwargs: dictionary of kwargs for the SGLD function. These parameters
           influence both training and generation.
        """
        self.E = Generator(x_dim, **gen_kwargs).to(device)
        self.x_dim = x_dim
        self.std = std
        self.stochastic = stochastic
        self.sgld_n_steps = sgld_n_steps
        self.sgld_kwargs = sgld_kwargs
        self.device = device
    
    def sample_sgld(self, 
                    n_steps: int, 
                    x0: torch.Tensor, 
                    batch_size: int, 
                    *,
                    a: float, 
                    b: float, 
                    log_every: Union[int, None] = None, 
                    return_all_samples: bool = True,
                    verbose: bool = False) -> List[torch.Tensor]:
        """Returns a list of samples, from time steps 1...T."""
        if x0 is None and batch_size is None:
            raise ValueError("both x0 and batch_size cannot be None")
        elif x0 is not None and batch_size is not None:
            raise ValueError("batch size does not have any effect if x0 is set")
        else:
            pass
        if x0 is None:
            x0 = self.sample_x0(batch_size)
        
        eps_schedule = get_steps(a, b, n_steps)

        x = torch.clone(x0)
        x.requires_grad = True

        if verbose:
            pbar = tqdm(total=len(eps_schedule))

        if return_all_samples:
            all_samples = []
        for b, eps in enumerate(eps_schedule):

            grad_E = grad(self.E(x).sum(), x)[0]
            
            # sample noise
            if self.stochastic:
                z = torch.zeros_like(x).normal_(0, 1)
            else:
                z = torch.zeros_like(x)

            assert x.shape == grad_E.shape == z.shape
            
            x = x - (((eps**2)/2)*grad_E) + eps*z
            x = x.detach()
            x.requires_grad = True

            if verbose:
                pbar.update(1)
                pbar.set_description_str("eps={}".format(eps))
                
            if log_every is not None and b % log_every == 0:
                if len(all_samples) == 0:
                    auto_corr = np.nan
                else:
                    auto_corr = torch.sum((x-saved_samples[-1])**2).item()
                print("iter: {}, grad norm = {:.3f}, AC={:.3f}, min={:.3f}, max={:.3f}, mean={:.3f}".format(
                    b, (total_grad**2).mean(), auto_corr, x.min(), x.max(), x.mean() ))
                
            if return_all_samples:
                all_samples.append(x)

        if verbose:
            pbar.close()

        if return_all_samples:
            return all_samples
    
        return x
    
    def sample_x0(self, bs: int) -> torch.Tensor:
        return torch.zeros((bs, self.x_dim)).uniform_(-1.5,2).to(self.device)
    
    def ebm_loss(self, X_batch: torch.Tensor) -> Tuple[ torch.Tensor, Tuple[torch.Tensor, torch.Tensor] ]:

        E_pos = self.E(X_batch)
        perm = torch.randperm(X_batch.size(0))
        noise = torch.zeros_like(X_batch).normal_(0, self.std)
        x_sampled = self.sample_sgld(self.sgld_n_steps, 
                                     X_batch[perm]+noise, 
                                     batch_size=None, 
                                     return_all_samples=False,
                                     **self.sgld_kwargs).detach()
        E_neg = self.E(x_sampled)
        log_px = (-E_pos + E_neg.mean()).mean()
        return log_px, (E_pos, E_neg)
        
    def train(self, 
              loader: torch.utils.data.DataLoader, 
              n_epochs: int, 
              verbose: bool = False,
              use_tqdm: bool = False,
              update_tqdm_every: int = 100,
              log_every: int = 10,
              opt_class=optim.Adam,
              **opt_kwargs) -> Dict:
        """
        max p(x) = max -E(x) + E_{x ~ p_g(x)} E(x)
                 
        Where p_g(x) is our generative distribution, sampled
        from with SGLD.
        """
        
        opt_E = opt_class(self.E.parameters(), **opt_kwargs)
            
        buf = DotDict(log_px=[], E_neg=[], E_pos=[])
        for epoch in range(n_epochs):
            losses = []
            fxs = []
            if use_tqdm:
                pbar = tqdm(total=len(loader))
            buf_ = DotDict(log_px=[], E_neg=[], E_pos=[])
            for b, (X_batch, _) in enumerate(loader):
                
                opt_E.zero_grad()
                
                X_batch = X_batch.to(self.device)
                #y_batch = y_batch.to(self.device)
                
                log_px, (E_pos, E_neg) = self.ebm_loss(X_batch)
                with torch.no_grad():
                    buf_.E_neg.append(E_neg.mean().item())
                    buf_.E_pos.append(E_pos.mean().item())
                    buf_.log_px.append(log_px.item())

                (-log_px).backward()
                opt_E.step()
                
                if use_tqdm:
                    pbar.update(1)
                    if b % update_tqdm_every == 0 and b > 0:
                        pbar.set_postfix(
                            {k:np.mean(v) for k,v in buf_.items()}
                        )
            
            for k,v in buf_.items():
                buf[k].append(np.mean(v))
                
            if verbose and epoch % log_every == 0:
                print({ k:"{:.3f}".format(v[-1]) for k,v in buf.items()})
            
            if use_tqdm:
                pbar.close()
        return buf    
    
    def __repr__(self):
        return r"""
net = {}
beta = {}
sgld args = {}""".format(self.E, self.beta, self.sgld_kwargs).strip()
