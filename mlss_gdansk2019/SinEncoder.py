from mxnet import gluon
from mxnet.gluon import rnn
from mxnet import ndarray as nd


class SinEncoder(gluon.Block):
    """
    Sinus GRU encoder. Last GRU state is used as an encoded representation of the sinus signal.
    """

    def __init__(self, **kwargs):
        super(SinEncoder, self).__init__(**kwargs)

        with self.name_scope():
            self.rnn = rnn.GRU(hidden_size=10)

    def forward(self, signal: nd.ndarray):
        """
        Args:
            signal: Sin signal (m, signal_lengths), m - num of signals (batch_size)

        Returns: (last_gru_layer_output:Array(m,s) where m - batch size, s - dimensionality of the GRU latent space

        """
        output = nd.transpose(signal, axes=(1, 0))  # (length, m)
        output = output.expand_dims(axis=2)  # (length, m, 1)

        initial_hidden_state = self.rnn.begin_state(func=nd.zeros, batch_size=signal.shape[0])
        output, latent_space = self.rnn(output, initial_hidden_state)

        output = output[-1, :]  # (m,hidden_size)

        return output