from typing import Callable, Tuple, Sequence
from dataclasses import dataclass

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


def get_default_dendritic_fn(
    threshold: float, sensitivity: float, gain: float
) -> Callable:
    sigmoid_fn = get_sigmoid_fn(threshold, sensitivity, gain)

    def dendritic_fn(x):
        return sigmoid_fn(x) + x

    return dendritic_fn


class Environment:
    device = torch.device('cpu')
    dtype = torch.float

    time_step = 1e-3
    nb_steps = 200
    batch_size = 256


@dataclass
class NeuronParameters:
    tau_mem: float = 10e-3
    tau_syn: float = 5e-3

    somatic_spike_fn: Callable


@dataclass
class TwoCompartmentNeuronParameters(NeuronParameters):
    """A spiking neuron with a dendritic compartment.

    No recurrent connection from somatic compartment back to dendritic
    compartment.

    """

    dendritic_spike_fn: Callable


@dataclass
class ParallelNeuronParameters(TwoCompartmentNeuronParameters):
    """Parameters of a two compartment neuron with parallel dendritic subunits.

    No recurrent connection from somatic compartment back to dendritic
    compartment.

    """

    NotImplemented


@dataclass
class RecurrentNeuronParameters(TwoCompartmentNeuronParameters):
    """Params of a two compartment neuron with recurrent connection.

    Has a feedback connection from soma to dendrite.

    """

    backprop_gain: float = 0.05


@dataclass
class PRCNeuronParameters(ParallelNeuronParameters, RecurrentNeuronParameters):
    # Union of ParallelNeuronParameters and RecurrentNeuronParameters and
    # nothing more.
    pass


@dataclass
class NetworkArchitecture:
    nb_units_by_layer: Sequence[int] = (100, 4, 2)


class _Synapses:
    def __init__(self, nb_units: int, discount_factor: float):
        self._discount_factor = discount_factor
        self.array = _zeros((Environment.batch_size, nb_units))

    def integrate_weighted_spikes(self, inputs):
        self.array = self.array * self._discount_factor + inputs


def _zeros(shape):
    """Get zeros tensor with device and dtype determined by Environment."""
    return torch.zeros(
        shape, device=Environment.device, dtype=Environment.dtype
    )


class SpikingNetwork:
    def __init__(
        self,
        neuron_parameters: NeuronParameters,
        network_architecture: NetworkArchitecture,
    ):
        # This is called 'alpha' in Friedemann's script
        self._syn_discount = self._timescale_to_discount(
            neuron_parameters.tau_syn
        )
        # This is called 'beta' in Friedemann's script
        self._mem_discount = self._timescale_to_discount(
            neuron_parameters.tau_mem
        )
        self._somatic_spike_fn = neuron_parameters.somatic_spike_fn

        # architecture stuff...
        self.nb_units_by_layer = network_architecture.nb_units_by_layer
        self.weights_by_layer = [
            None for _ in range(len(self.nb_units_by_layer) - 1)
        ]
        self._initialize_weights()

    @staticmethod
    def _timescale_to_discount(timescale: float) -> float:
        return float(np.exp(-Environment.time_step / timescale))

    def _initialize_weights(self, weight_scale=7.0):
        adjusted_weight_scale = weight_scale * (1.0 - self._mem_discount)

        assert len(self.nb_units_by_layer) == len(self.weights_by_layer) + 1

        # Initialize all weights from a normal distribution.
        for l in range(len(self.weights_by_layer)):
            self.weights_by_layer[l] = torch.empty(
                self.nb_units_by_layer[l : l + 2],
                device=Environment.device,
                dtype=Environment.dtype,
                requires_grad=True,
            )
            torch.nn.init.normal_(
                self.weights_by_layer[l],
                mean=0.0,
                std=adjusted_weight_scale / np.sqrt(self.nb_units_by_layer[l]),
            )

    def _compute_new_soma_state(
        self,
        current_state,
        external_inputs: _Synapses,
        somatic_spikes,
    ):
        return (
            self._mem_discount * current_state
            + external_inputs.array
        ) * (1.0 - somatic_spikes)

    def run_snn(self, inputs):
        pre_activation_l1 = torch.einsum(
            # [b]atches, [t]ime, [i]nput units, [h]idden units, [c]ompartments
            "bti,ih->bth",
            (inputs, self.weights_by_layer[0]),
        )
        soma_syn = _Synapses(self.nb_units_by_layer[1], self._syn_discount)
        soma = _zeros((Environment.batch_size, self.nb_units_by_layer[1]))

        # Here we define lists which we use to record the membrane potentials and output spikes
        soma_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(Environment.nb_steps):
            soma_syn.integrate_weighted_spikes(pre_activation_l1[:, t, :, 1])

            out = self._somatic_spike_fn(soma)

            reset = (
                out.detach()
            )  # We do not want to backprop through the reset
            new_soma = self._compute_new_soma_state(
                soma, soma_syn, reset
            )

            soma_rec.append(soma)
            spk_rec.append(out)

            soma = new_soma

        soma_rec = torch.stack(soma_rec, dim=1)
        spk_rec = torch.stack(spk_rec, dim=1)

        del soma, new_soma, reset, soma_syn

        # Readout layer
        pre_activation_l2 = torch.einsum(
            # [b]atch, [t]ime, [h]idden, [o]utput
            "bth,ho->bto",
            (spk_rec, self.weights_by_layer[1]),
        )
        output_syn = _Synapses(self.nb_units_by_layer[2], self._syn_discount)
        out = _zeros((Environment.batch_size, self.nb_units_by_layer[2]))
        out_rec = [out]
        for t in range(Environment.nb_steps):
            out = self._mem_discount * out + output_syn.array
            output_syn.integrate_weighted_spikes(pre_activation_l2[:, t, :])
            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = {
            'hidden_soma': soma_rec,
            'hidden_soma_spike': spk_rec,
        }
        return out_rec, other_recs


class TwoCompartmentSpikingNetwork(SpikingNetwork):
    def __init__(
        self,
        neuron_parameters: TwoCompartmentNeuronParameters,
        network_architecture: NetworkArchitecture,
    ):
        super().__init__(neuron_parameters, network_architecture)
        self._dendritic_spike_fn = neuron_parameters.dendritic_spike_fn

    def _initialize_weights(self, weight_scale=7.0):
        adjusted_weight_scale = weight_scale * (1.0 - self._mem_discount)

        assert len(self.nb_units_by_layer) == len(self.weights_by_layer) + 1

        # Initialize all weights from a normal distribution.
        for l in range(len(self.weights_by_layer)):
            if l < len(self.weights_by_layer) - 1:
                # All layers before output layer have two compartment neurons,
                # so we add an extra dim to the weight array for inputs to
                # somata and dendrites.
                # Shape is [units_in_prev_layer, units_this_layer, compartment]
                weight_tensor_shape: Tuple[int, int, int] = (
                    *self.nb_units_by_layer[l : l + 2],
                    2,
                )
            else:
                # The output layer has one compartment neurons, so no need for
                # an extra dimension.
                # Shape is [units_in_prev_layer, units_this_layer]
                weight_tensor_shape: Tuple[int, int] = self.nb_units_by_layer[
                    l : l + 2
                ]
            self.weights_by_layer[l] = torch.empty(
                weight_tensor_shape,
                device=Environment.device,
                dtype=Environment.dtype,
                requires_grad=True,
            )
            torch.nn.init.normal_(
                self.weights_by_layer[l],
                mean=0.0,
                std=adjusted_weight_scale / np.sqrt(self.nb_units_by_layer[l]),
            )

    def _compute_new_dendrite_state(
        self, current_state, external_inputs: _Synapses, somatic_spikes
    ):
        # Note: somatic_spikes is not used. Included for consistency with
        # RecurrentSpikingNetwork
        return self._mem_discount * current_state + external_inputs.array

    def _compute_new_soma_state(
        self,
        current_state,
        external_inputs: _Synapses,
        dendritic_nl_input,
        somatic_spikes,
    ):
        return (
            self._mem_discount * current_state
            + external_inputs.array
            + dendritic_nl_input
        ) * (1.0 - somatic_spikes)

    def run_snn(self, inputs):
        pre_activation_l1 = torch.einsum(
            # [b]atches, [t]ime, [i]nput units, [h]idden units, [c]ompartments
            "bti,ihc->bthc",
            (inputs, self.weights_by_layer[0]),
        )
        dendrite_syn = _Synapses(self.nb_units_by_layer[1], self._syn_discount)
        soma_syn = _Synapses(self.nb_units_by_layer[1], self._syn_discount)
        dendrite = _zeros((Environment.batch_size, self.nb_units_by_layer[1]))
        soma = _zeros((Environment.batch_size, self.nb_units_by_layer[1]))

        # Here we define lists which we use to record the membrane potentials and output spikes
        dendrite_rec = []
        dendrite_nl_rec = []
        soma_rec = []
        spk_rec = []

        # Here we loop over time
        for t in range(Environment.nb_steps):
            dendrite_syn.integrate_weighted_spikes(
                pre_activation_l1[:, t, :, 0]
            )
            soma_syn.integrate_weighted_spikes(pre_activation_l1[:, t, :, 1])

            out = self._somatic_spike_fn(soma)
            dendrite_nl = self._dendritic_spike_fn(dendrite)

            reset = (
                out.detach()
            )  # We do not want to backprop through the reset
            new_dendrite = self._compute_new_dendrite_state(
                dendrite, dendrite_syn, reset
            )
            new_soma = self._compute_new_soma_state(
                soma, soma_syn, dendrite_nl, reset
            )

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

        del (
            dendrite,
            soma,
            new_dendrite,
            new_soma,
            reset,
            dendrite_nl,
            dendrite_syn,
            soma_syn,
        )

        # Readout layer
        pre_activation_l2 = torch.einsum(
            # [b]atch, [t]ime, [h]idden, [o]utput
            "bth,ho->bto",
            (spk_rec, self.weights_by_layer[1]),
        )
        output_syn = _Synapses(self.nb_units_by_layer[2], self._syn_discount)
        out = _zeros((Environment.batch_size, self.nb_units_by_layer[2]))
        out_rec = [out]
        for t in range(Environment.nb_steps):
            out = self._mem_discount * out + output_syn.array
            output_syn.integrate_weighted_spikes(pre_activation_l2[:, t, :])
            out_rec.append(out)

        out_rec = torch.stack(out_rec, dim=1)
        other_recs = {
            'hidden_soma': soma_rec,
            'hidden_soma_spike': spk_rec,
            'hidden_dendrite': dendrite_rec,
            'hidden_dendrite_spike': dendrite_nl_rec,
        }
        return out_rec, other_recs


class RecurrentSpikingNetwork(TwoCompartmentSpikingNetwork):
    def __init__(
        self,
        neuron_parameters: RecurrentNeuronParameters,
        network_architecture: NetworkArchitecture,
    ):
        super().__init__(neuron_parameters, network_architecture)
        self._backprop_gain = neuron_parameters.backprop_gain

    def _compute_new_dendrite_state(
        self, current_state, external_inputs, somatic_spikes
    ):
        return (
            self._syn_discount * current_state
            + external_inputs
            + self._backprop_gain * somatic_spikes
        )

class ParallelSpikingNetwork(TwoCompartmentSpikingNetwork):
    NotImplemented

class PRCSpikingNetwork(ParallelSpikingNetwork, RecurrentSpikingNetwork):
    NotImplemented
