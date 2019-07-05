from mxnet import gluon
from mxnet.gluon import rnn, nn
from mxnet import ndarray as nd
import mxnet as mx
import numpy as np

from mlss_gdansk2019.StatefulGRUCell import StatefulGRUCell

"""
Sinus GRU auto regressive decoder

fixed sized decoder input -> decoded sinus signal

"""


class SinDecoder(gluon.Block):

    def __init__(self, samples_per_step: int = 5, **kwargs):
        """
        Args:
            samples_per_step: How many samples to generate at each decoding step.
            **kwargs:
        """
        super(SinDecoder, self).__init__(**kwargs)

        self._samples_per_step = samples_per_step

        with self.name_scope():
            self._decoder_rnn = rnn.GRUCell(hidden_size=20)
            self._projection_layer = nn.Dense(samples_per_step, flatten=False)

    def forward(self, gaussian_latent_space_sample: nd.NDArray,
                length: int, target_signal: nd.NDArray,
                teacher_forcing_prob: float):
        """
        Args:
            gaussian_latent_space_sample: (batch_size, 1)
            length: The total number of samples to generate
            target_signal: True sinus signal that is used during teacher-forcing training.
            teacher_forcing_prob: The probability of using the teacher forcing.

        Returns:
            Decoder sinus signal: (batch_size, length)
        """

        batch_size = gaussian_latent_space_sample.shape[0]
        stateful_rnn = StatefulGRUCell(self._decoder_rnn)
        predicted_frame = mx.nd.zeros(shape=(batch_size, self._samples_per_step))
        reconstructed_signal = []
        for i in range(length // self._samples_per_step):
            if target_signal is not None:
                predicted_frame = self._select_signal(predicted_frame, target_signal,
                                                      teacher_forcing_prob, i)  # teacher forcing
            rnn_input = nd.concat(gaussian_latent_space_sample,
                                  predicted_frame, dim=1)  # (batch_size, samples_per_step + 1)
            rnn_output = stateful_rnn(rnn_input)  # (batch_size, rnn_hidden_size)
            predicted_frame = self._projection_layer(rnn_output)  # (batch_size, samples_per_step)
            reconstructed_signal.append(predicted_frame)

        return nd.concat(*reconstructed_signal, dim=1)  # (batch_size, floor(length/samples_per_step) * samples_per_step)

    def _select_signal(self,
                       predicted_signal: nd.NDArray,
                       oracle_signal: nd.NDArray,
                       teacher_forcing_prob: float,
                       step: int):
        """
        Base on teacher_forcing_prob select either predicted_signal or aligned signal from
        Oracle signal.
        Args:
            predicted_signal: (batch_size, _samples_per_step)
            oracle_signal: (batch_size, #total number of samples)
            teacher_forcing_prob: probability that oracle signal will be selected
            step: step used to align between oracle and predicted signal
        Returns:
            signal for forward pass
        """
        if step > 0 and np.random.rand() < teacher_forcing_prob:
            start_sample = (step - 1) * self._samples_per_step
            stop_sample = start_sample + self._samples_per_step
            teacher_forcing_signal = oracle_signal[:, start_sample:stop_sample]
            return teacher_forcing_signal
        else:
            return predicted_signal
