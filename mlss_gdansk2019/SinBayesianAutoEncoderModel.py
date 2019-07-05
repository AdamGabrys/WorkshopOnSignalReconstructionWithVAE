from typing import NamedTuple

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd

from mlss_gdansk2019.SinDecoder import SinDecoder
from mlss_gdansk2019.SinEncoder import SinEncoder

"""
Bayesian Auto Encoder for sinus reconstruction.

The prior for the latent space = N(0,I)
Resources:
- https://arxiv.org/pdf/1312.6114.pdf
- https://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder
- https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf

sin(x) -> encoder -> 1D gaussian latent space -> decoder -> decoded sin(x)
"""


class SinBayesianAutoEncoder(gluon.Block):

    def __init__(self, n_latent=2, **kwargs):
        super(SinBayesianAutoEncoder, self).__init__(**kwargs)

        with self.name_scope():
            self.encoder = SinEncoder()
            self.latent_space = nn.Dense(n_latent * 2)  # mean and log variance of the latent space
            self.decoder = SinDecoder(samples_per_step=5)  # predict 5 samples at once
            self.l2loss = gluon.loss.L2Loss()

    def forward(self, signal: nd.NDArray, teacher_forcing_prob: float,
                latent_space_override: nd.NDArray = None):
        """

        Args:
            signal: Sin signal (m, signal_length), m - num of signals (batch_size)
            teacher_forcing_prob: The probability of activating the teacher forcing
            latent_space_override: The override value for the latent space.

        Returns:

        """
        sig_embedding = self.encoder(signal)  # (m,s), s - dim of the encoder embedding

        # Posterior of the latent space
        # Gaussian variance must be positive, therefore using log variance parametrization
        ls_mean, ls_log_var = self.latent_space(sig_embedding).split(axis=1, num_outputs=2)
        ls_std = nd.exp(ls_log_var * 0.5, axis=0)

        # Sampling from the unit gaussian instead of sampling from the latent space posterior
        # allow for gradient flow via latent_space_mean / latent_space_log_var parameters
        # z = (x-mu)/std, thus: x = mu + z*std
        normal_sample = nd.random_normal(0, 1, shape=ls_mean.shape)
        ls_val = ls_mean + ls_std * normal_sample

        if isinstance(latent_space_override, nd.NDArray):
            ls_val = latent_space_override

        length = signal.shape[1]
        reconstructed_sig = self.decoder(ls_val, length, signal, teacher_forcing_prob)  # (m,length)

        return SinBAEOutput(ls_mean, ls_log_var, ls_val, reconstructed_sig)

    def calc_loss(self, signal: nd.NDArray, teacher_forcing_prob: float) -> (float, float):
        """
        Compute gradients of the loss function with respect of the model parameters.

        Args:
            signal: Sin signal: (m, signal_length), m - num of signals (batch_size)
            teacher_forcing_prob: TODO

        Returns: L2 Loss between input and decoded signals, KLD loss

        """

        decoded_signal_output = self(signal, teacher_forcing_prob)

        latent_space_mean = decoded_signal_output.latent_space_mean
        latent_space_log_var = decoded_signal_output.latent_space_log_var

        l2_loss = self.l2loss(signal, decoded_signal_output.decoded_signal)
        negative_kld = 0.5 * nd.sum(
            1 + latent_space_log_var - latent_space_mean ** 2 - nd.exp(latent_space_log_var), axis=1)

        return l2_loss, -negative_kld


class SinBAEOutput(NamedTuple):
    """
    Args:
        latent_space_mean: array(m,1), The mean of the posterior of the latent space
        latent_space_log_var: array(m,1), Variance of the posterior of the latent space
        latent_space_val: array(m,1), Value sampled from the posterior of the latent space
        decoded_signal: array(m,signal_length)
    """
    latent_space_mean: nd.NDArray
    latent_space_log_var: nd.NDArray
    latent_space_val: nd.NDArray
    decoded_signal: nd.NDArray
