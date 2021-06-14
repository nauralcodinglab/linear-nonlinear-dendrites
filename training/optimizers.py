import torch
import torch.nn as nn
from tqdm.notebook import trange


class DefaultOptimizer:
    def __init__(self, forward_fn, params):
        """Initialize the optimizer.

        Parameters
        ----------
        forward_fn
            Function used to compute the forward pass through the network.
            Should take the input to the network and return the output.
        params
            Parameters of the forward_fn to be optimized.

        """
        self._forward = forward_fn
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=2e-3, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        neg_log_lik_fn = nn.NLLLoss()

        def loss_fn(actual_output, desired_output):
            m, _ = torch.max(actual_output, 1)
            log_p_y = log_softmax_fn(m)
            loss_val = neg_log_lik_fn(log_p_y, desired_output.long())
            return loss_val

        self.loss_fn = loss_fn
        self.loss_history = []  # Will contain the loss at each epoch

    def optimize(self, input_, desired_output, epochs: int):
        """Run the optimizer.

        Parameters
        ----------
        input_
            Input to the network (passed to forward_fn).
        desired_output
            Desired output from the network (compared with the output of
            forward_fn to compute the loss).
        epochs
            Number of epochs to run optimization.

        """
        for e in trange(epochs):
            actual_output = self._forward(input_)

            self.optimizer.zero_grad()
            loss_val = self.loss_fn(actual_output, desired_output)
            loss_val.backward()
            self.optimizer.step()

            self.loss_history.append(loss_val.item())
