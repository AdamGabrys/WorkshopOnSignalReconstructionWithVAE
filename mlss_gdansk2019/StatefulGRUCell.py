from mxnet.gluon import rnn
from mxnet import nd

"""
Gru maintaining its state.
"""


class StatefulGRUCell:
    """
    Wrapper over GRUCell caching hidden states.
    """
    def __init__(self, gru_cell: rnn.GRUCell):
        """
        Args:
            gru_cell: GRUCell which will be used to run forward pass.
            **kwargs:
        """
        self._gru_cell = gru_cell
        self._state = None

    def __call__(self, input_: nd.NDArray):
        """
        Run forward pass on GRUCell.
        First forward call will be initialized with GRUCell.begin_state, all further calls
        will use the latest state returned by GRUCell.
        Args:
            input_: input tensor with shape (batch_size, input_size).
        """
        if not self._state:
            self._state = self._gru_cell.begin_state(input_.shape[0])
        output, new_state = self._gru_cell(input_, self._state)
        self._state = new_state

        return output
