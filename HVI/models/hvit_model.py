import sys
sys.path.insert(1,'../distributions')
from wrapped_normal import WrappedNormal

import torch
from torch import nn
import torch.nn.functional as F 
from torch.distributions import kl_divergence as kl
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi import _CONSTANTS
import numpy as np
import math
from modules import *

# at beginning of the script
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.double)

class HVITModel(BaseModuleClass):
    """
    Variational AutoEncoder for latent space arithmetic for perutrbation prediction.
    
    Parameters
    ----------
    manifold
        Geoopt object that represents what kind of manifold you are on, with what curvature
    input_dim
        Number of input genes
    hidden_dim
        Number of nodes per hidden layer
    latent_dim
        Dimensionality of the latent space
    n_hidden_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    kl_weight
        Weight for kl divergence
    linear_decoder
        Boolean for whether or not to use linear decoder for a more interpretable model.
    """

    def __init__(
        self,
        manifold,
        input_dim: int,
        hidden_dim: int = 800,
        latent_dim: int = 2,
        n_hidden_layers: int = 2,
        dropout_rate: float = 0.1,
        kl_weight: float = 0.00005,
        linear_decoder: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.n_hidden_layers = n_hidden_layers
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.latent_distribution = "Normal"
        
        #         self.encoder = Encoder(
#             input_dim,
#             latent_dim,
#             n_layers=n_layers,
#             hidden_dim=hidden_dim,
#             dropout_rate=dropout_rate,
#             distribution=latent_distribution,
#             use_batch_norm= True,
#             use_layer_norm=False,
#             activation_fn=torch.nn.LeakyReLU,
#         )

        self.encoder = HVITEncoder(
            self.manifold,
            input_dim,
            latent_dim,
            hidden_dim,
            n_hidden_layers
        )
        
        
        #Decoder
        #Include Input Module
        
        self.nonlin_decoder = HVITDecoder(
            self.manifold,
            input_dim,
            latent_dim,
            hidden_dim,
            n_hidden_layers
        )
        
        #self.lin_decoder =  torch.nn.Sequential(self.manifold.logmap0,
         #               nn.Linear(latent_dim, input_dim),
          #              nn.BatchNorm1d(input_dim, momentum=0.01, eps=0.001)
           #             )
        
        self.decoder = self.lin_decoder if linear_decoder else self.nonlin_decoder
    
    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        input_dict = dict(
            x=x,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        input_dict = {
            "z": z,
        }
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        mean, var, latent_rep = self.encoder(x)

        outputs = dict(z=latent_rep, qz_m=mean, qz_v=var)
        return outputs

    @auto_move_data
    def generative(self, z):
        """Runs the generative model."""
        px = self.decoder(z)

        return dict(px=px)

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
    ):
        x = tensors[_CONSTANTS.X_KEY]
        mean = inference_outputs["qz_m"]
        var = inference_outputs["qz_v"]
        x_hat = generative_outputs["px"]
        z = inference_outputs["z"]
        std = var.sqrt()

        qz_x = WrappedNormal(mean, std, self.manifold)
        pz_mean = nn.Parameter(torch.zeros(1, self.latent_dim), requires_grad=False).to(device)
        pz_logvar = nn.Parameter(torch.ones(1, self.latent_dim), requires_grad=False).to(device)
        pz = WrappedNormal(pz_mean, pz_logvar, self.manifold)
        
        neg_elbo, kld = self.get_neg_elbo(x, x_hat, z, qz_x, pz, self.kl_weight)
        rl = self.get_reconstruction_loss(x, x_hat)
        
        loss = (0.5 * rl + 0.5 * (kld * self.kl_weight)).mean()
        return LossRecorder(loss, rl, kld, kl_global=0.0)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        inference_outputs, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )
        px = WrappedNormal(generative_outputs["px"], 1, self.manifold).sample()
        return px.cpu().numpy()

    def get_reconstruction_loss(self, x, x_hat) -> torch.Tensor:
        loss = ((x - x_hat) ** 2).sum(dim=1)
        return loss

    def get_neg_elbo(self, x, x_hat, z, qz_x, pz, beta) -> torch.Tensor:
        px_z = Normal(x_hat, torch.ones_like(x_hat).to(device))
        log_likelihood = px_z.log_prob(x).sum(-1)
        kl =  F.log_softmax(qz_x.log_prob(z).exp(), dim = -1).sum(-1) - F.log_softmax(pz.log_prob(z).exp(), dim = -1).sum(-1)
        neg_elbo = -log_likelihood.mean(0).sum() + beta * kl.mean(0).sum()
        return neg_elbo, kl
