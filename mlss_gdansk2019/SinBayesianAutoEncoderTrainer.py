import logging
import time
from typing import List

from mxnet.gluon.data import DataLoader
from mxnet import gluon, autograd
import numpy as np
import mxnet.ndarray as nd

from mlss_gdansk2019.SinBayesianAutoEncoderModel import SinBayesianAutoEncoder


class SinBayesianAutoEncoderTrainer:
    """
    Train SinBayesianAutoEncoder model.
    """

    def __init__(self,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 model: SinBayesianAutoEncoder,
                 kld_weight: float = 0.01,
                 optimizer: str = "adam",
                 learning_rate: float = 0.001):
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._model = model
        self._kld_weight = kld_weight
        self._trainer = gluon.Trainer(self._model.collect_params(),
                                      optimizer, {'learning_rate': learning_rate})

    def train(self, n_epoch: int, decay_multiplier: int = 0.01) -> (List[float],
                                                                    List[float],
                                                                    List[float]):
        """
        Train the model for given number of epochs updating its weights,
        and exponentially decaying teacher forcing probability
        Args:
            n_epoch: Number of epochs for which model will be trained.
            decay_multiplier: multiplier of epoch in exponential decay of
                              teacher forcing.

        Returns: test_l2_loss_list, test_kld_list, teacher_forcing_prob_list
        """
        start = time.time()

        test_l2_loss_list = []
        test_kld_list = []
        teacher_forcing_prob_list = []
        for epoch in range(n_epoch):
            teacher_forcing_prob = np.exp(-decay_multiplier * epoch)

            epoch_l2_train_loss, epoch_kld_train_loss = self._forward_backward(teacher_forcing_prob)
            epoch_l2_test_loss, epoch_kld_test_loss = self._calc_test_accuracy(teacher_forcing_prob)

            epoch_total_train_loss = epoch_l2_train_loss + epoch_kld_train_loss
            epoch_total_test_loss = epoch_l2_test_loss + epoch_kld_test_loss
            logging.info(
                'Epoch %d, teacher_forcing_prob=%.2f, training loss (total/l2/kld): '
                '%.4f, %.4f, %.4f, test loss (total/l2/kld): %.4f, %.4f, %.4f'
                % (epoch,
                   teacher_forcing_prob,
                   epoch_total_train_loss, epoch_l2_train_loss, epoch_kld_train_loss,
                   epoch_total_test_loss, epoch_l2_test_loss, epoch_kld_test_loss))

            test_l2_loss_list.append(epoch_l2_test_loss)
            test_kld_list.append(epoch_kld_test_loss)
            teacher_forcing_prob_list.append(teacher_forcing_prob)

        end = time.time()
        logging.info('Time elapsed: {:.2f}s'.format(end - start))

        return test_l2_loss_list, test_kld_list, teacher_forcing_prob_list

    def _forward_backward(self, teacher_forcing_prob: float) -> (float, float):
        """
        Run single epoch forward and backward path with update of parameters on the model.
        Args:
            teacher_forcing_prob: probability of selecting oracle signal
        Returns:
            l2 loss, kld loss
        """
        epoch_l2_loss = 0
        epoch_kld_loss = 0

        for signal_batch in self._train_dataloader:
            with autograd.record():
                l2_loss, kld_loss = self._model.calc_loss(signal_batch, teacher_forcing_prob)
                loss = l2_loss + self._kld_weight * kld_loss  # weighting the kld loss

            loss.backward()
            self._trainer.step(signal_batch.shape[0])

            epoch_l2_loss += nd.mean(l2_loss).asscalar()
            epoch_kld_loss += nd.mean(kld_loss).asscalar()

        epoch_l2_loss /= len(self._train_dataloader)
        epoch_kld_loss /= len(self._train_dataloader)

        return epoch_l2_loss, epoch_kld_loss

    def _calc_test_accuracy(self, teacher_forcing_prob: float) -> (float, float):
        """
        Computes test set single epoch loss.
        Args:
            teacher_forcing_prob: probability of selecting oracle signal
        Returns:
            l2 loss, kld loss
        """
        epoch_l2_loss = 0
        epoch_kld_loss = 0

        for signal_batch in self._test_dataloader:
            l2_loss, kld_loss = self._model.calc_loss(signal_batch, teacher_forcing_prob)
            kld_loss = self._kld_weight * kld_loss

            epoch_l2_loss += nd.mean(l2_loss).asscalar()
            epoch_kld_loss += nd.mean(kld_loss).asscalar()

        epoch_l2_loss /= len(self._test_dataloader)
        epoch_kld_loss /= len(self._test_dataloader)

        return epoch_l2_loss, epoch_kld_loss
