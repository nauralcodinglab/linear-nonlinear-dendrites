from typing import Callable

import torch
import numpy as np


class _SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements
    the surrogate gradient. By subclassing torch.autograd.Function,
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid
    as this was done in Zenke & Ganguli (2018).

    Implemented by Friedemann Zenke.

    """

    scale = 100.0  # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which
        we need to later backpropagate our error signals. To achieve this we use the
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the
        surrogate gradient of the loss with respect to the input.
        Here we use the normalized negative part of a fast sigmoid
        as this was done in Zenke & Ganguli (2018).
        """
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (
            grad_input / (_SurrGradSpike.scale * torch.abs(input) + 1.0) ** 2
        )
        return grad


def get_spike_fn(threshold: float) -> Callable:
    differentiable_heaviside = _SurrGradSpike.apply

    def spike_fn(x):
        x_over_thresh = x - threshold
        out = differentiable_heaviside(x_over_thresh)
        return out

    return spike_fn


def get_sigmoid_fn(
    threshold: float, sensitivity: float, gain: float
) -> Callable:
    plain_sigmoid = torch.nn.Sigmoid()

    def sigmoid_fn(x):
        return gain * plain_sigmoid(sensitivity * (x - threshold))

    return sigmoid_fn


class SpikingNetwork:
    _device = torch.device('cpu')
    _dtype = torch.float

    def __init__(self):

        self.nb_units_by_layer = (100, 4, 2)  # Input, hidden, output
        # Input -> hidden, hidden -> output
        self.weights_by_layer = [
            None for _ in range(len(self.nb_units_by_layer) - 1)
        ]
        self._initialize_weights()

        self.time_step = 1e-3
        self.nb_steps = 200
        self.batch_size = 256

    def set_neuron_parameters(
        self,
        tau_mem: float,
        tau_syn: float,
        somatic_spike_fn: Callable,
        dendritic_spike_fn: Callable,
    ):
        # Convert the synaptic and membrane time constants to discounting
        # factors. These are called 'alpha' and 'beta' in Friedemann's script.
        self._syn_discount = float(np.exp(-self.time_step / tau_syn))
        self._mem_discount = float(np.exp(-self.time_step / tau_mem))

        self._somatic_spike_fn = somatic_spike_fn
        self._dendritic_spike_fn = dendritic_spike_fn

    def _initialize_weights(self, weight_scale=7.0):
        adjusted_weight_scale = weight_scale * (1.0 - self._mem_discount)

        assert len(self.nb_units_by_layer) == len(self.weights_by_layer) + 1
        for l in range(len(self.weights_by_layer)):
            self.weights_by_layer[l] = torch.empty(
                self.nb_units_by_layer[l : l + 2],
                device=self._device,
                dtype=self._dtype,
                requires_grad=True,
            )
            torch.nn.init.normal_(
                self.weights_by_layer[l],
                mean=0.0,
                std=adjusted_weight_scale / np.sqrt(self.nb_units_by_layer[l]),
            )


"""
def run_snn(inputs):
    h1 = torch.einsum("abc,cd->abd", (inputs, w1))
    dendrite = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    soma = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    # Here we define two lists which we use to record the membrane potentials and output spikes
    dendrite_rec = []
    dendrite_nl_rec = []
    soma_rec = []
    spk_rec = []

    sig = torch.nn.Sigmoid()

    # Here we loop over time
    for t in range(nb_steps):
        soma_over_thresh = soma - soma_thresh
        out = spike_fn(soma_over_thresh)
        reset = out.detach()  # We do not want to backprop through the reset

        new_dendrite = alpha * dendrite + h1[:, t] + gamma * reset
        dendrite_nl = (
            sig_gain * sig(sig_sensitivity * (dendrite - dend_thresh))
            + dendrite
        )
        new_soma = (beta * soma + dendrite_nl) * (1.0 - reset)

        soma_rec.append(soma)
        dendrite_rec.append(dendrite)
        dendrite_nl_rec.append(dendrite_nl)
        spk_rec.append(out)

        dendrite = new_dendrite
        soma = new_soma

    dendrite_rec = torch.stack(dendrite_rec, dim=1)
    dendrite_nl_rec = torch.stack(dendrite_nl_rec, dim=1)
    soma_rec = torch.stack(soma_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    # Readout layer
    h2 = torch.einsum("abc,cd->abd", (spk_rec, w2))
    flt = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        new_flt = alpha * flt + h2[:, t]
        new_out = beta * out + flt

        flt = new_flt
        out = new_out

        out_rec.append(out)

    out_rec = torch.stack(out_rec, dim=1)
    other_recs = [soma_rec, spk_rec]
    return out_rec, other_recs
"""
