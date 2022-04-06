import sys
#import needed manifolds
sys.path.insert(1, '../manifolds')
from poincareball import PoincareBall

#import needed distributions 
sys.path.insert(1,'distributions')
from wrapped_normal import WrappedNormal  

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
torch.set_default_dtype(torch.double)

class HVITEncoder(nn.Module):
    def __init__(
        self,
        manifold,
        input_dim: int, 
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        eta: float = 1e-4,
        nonlin = nn.LeakyReLU,
    ):
        super(HVITEncoder, self).__init__()
        self.eta = eta
        self.manifold = manifold
        self.fclayers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                    nonlin(),
                    nn.Dropout(p = dropout_rate),
        )
    
        #Encoder
        #Include Input Module
        modules = [nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p = dropout_rate),
                )]

        #Add hidden fully connected layers
        for i in range(n_hidden_layers-1):
                modules.append(self.fclayers)

        self.encoder = nn.Sequential(*modules)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, inputs):
        #encode
        inputs = inputs.double()
        results = self.encoder(inputs)
        mean = self.mean(results)
        mean = self.manifold.expmap0(mean)
        
        log_var = self.log_var(results)
        
        var = torch.exp(log_var).add(self.eta)
        
        #reparameterize
        latent_rep = WrappedNormal(mean, var.sqrt(), self.manifold).rsample()
        
        return mean, var, latent_rep

class HVITDecoder(nn.Module):
    def __init__(
        self,
        manifold,
        input_dim: int, 
        latent_dim: int,
        hidden_dim: int,
        n_hidden_layers: int,
        momentum: float = 0.01,
        eps: float = 0.001,
        dropout_rate: float = 0.2,
        nonlin = nn.LeakyReLU,
    ):
        super(HVITDecoder, self).__init__()
        self.manifold = manifold
        self.fclayers = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                    nonlin(),
                    nn.Dropout(p = dropout_rate),
        )
    
        #Encoder
        #Include Input Module
        modules = [nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=momentum, eps=eps),
                nonlin(),
                nn.Dropout(p = dropout_rate),
                )]

        #Add hidden fully connected layers
        for i in range(n_hidden_layers-1):
                modules.append(self.fclayers)
        
        modules.append(nn.Linear(hidden_dim, input_dim))
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, latent_rep):
        latent_rep = self.manifold.logmap0(latent_rep)
        #decode
        x_hat = self.decoder(latent_rep)
        return x_hat
