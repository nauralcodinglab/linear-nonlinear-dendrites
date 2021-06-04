from typing import (
    Callable,
    Tuple,
    Sequence,
    TypeVar,
    Optional,
    List,
    Iterable,
    Generic,
)
from dataclasses import dataclass
import warnings

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
    tau_mem: float
    tau_syn: float
    feedback_strength: float

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
    weight_scale: float = 7.0


class _TimeDependent:
    def __init__(self, initial_state):
        self.current_state = initial_state
        self.next_state = None

    def update(self):
        """Set the current_state to the next_state and clear next_state."""
        if self.next_state is None:
            raise RuntimeError('next_state is not set')
        self.current_state = self.next_state
        self.next_state = None


def _zeros(shape):
    """Get zeros tensor with device and dtype determined by Environment."""
    return torch.zeros(
        shape, device=Environment.device, dtype=Environment.dtype
    )


def _time_constant_to_discount_factor(time_constant: float) -> float:
    return float(np.exp(-Environment.time_step / time_constant))


class Subunit:
    def __init__(
        self,
        time_constant: float,
        initial_linear_state: torch.Tensor,
        initial_nonlinear_state: Optional[torch.Tensor] = None,
        feedforward_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        feedback_fn: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = lambda expected_linear, nonlinear: expected_linear,
    ):
        """Initialize the Subunit.

        Parameters
        ----------
        time_constant
            Time constant of the Subunit passive filter.
        feedforward_fn
            Nonlinear function applied to subunit input after linear filtering
            to generate subunit output. Default is no nonlinearity.
        feedback_fn
            Nonlinear function used to control the effect of the nonlinear
            output of the subunit on the linear state at the next time step.
            Must accept the expected linear state at the next time step and
            the nonlinear state at the current time step as inputs.

        """
        self._linear: _TimeDependent = _TimeDependent(initial_linear_state)
        if initial_nonlinear_state is None:
            initial_nonlinear_state = initial_linear_state.clone()
        self._nonlinear: _TimeDependent = _TimeDependent(
            initial_nonlinear_state
        )
        self._discount_factor: float = _time_constant_to_discount_factor(
            time_constant
        )
        self._feedforward_fn = feedforward_fn
        self._feedback_fn = feedback_fn

    @property
    def linear(self):
        return self._linear.current_state

    @property
    def nonlinear(self):
        return self._nonlinear.current_state

    def integrate_input(self, input_: torch.Tensor):
        self._linear.next_state = self._feedback_fn(
            self._linear.current_state * self._discount_factor + input_,
            self._nonlinear.current_state,
        )
        self._nonlinear.next_state = self._feedforward_fn(
            self._linear.current_state
        )

    def update(self):
        self._linear.update()
        self._nonlinear.update()


T = TypeVar('T')


class Recorder(Generic[T]):
    """Record a set of attributes over time."""

    def __init__(
        self,
        object_to_record: T,
        attributes_to_record: Iterable[str],
        record_initial_state: bool = True,
    ):
        """Initialize Recorder.

        Parameters
        ----------
        object_to_record
            The object whose attributes will be recorded when `record()` is
            called.
        attributes_to_record
            Iterable of strings giving the names of attributes to record.
            Nested attributes can be specified using dot notation. For example,
            use `['foo.bar', 'a.b.c']` to record `object_to_record.foo.bar` and
            `object_to_record.a.b.c`.
        record_initial_state
            Whether to record the state of `object_to_record` upon
            initialization of `Recorder`. Equivalent to calling `record()`
            immediately after initialization.

        """
        self._recorded_object = object_to_record
        self.recorded = {attr_name: [] for attr_name in attributes_to_record}
        self._finalized = False

        if record_initial_state:
            self.record()

    def record(self):
        """Save the current state of recorded attributes."""
        self._check_not_finalized()
        for attr in self.recorded.keys():
            self.recorded[attr].append(self._get_recorded_attr(attr))

    def update_and_record(self):
        """Update recorded attributes and record the new state."""
        self._check_not_finalized()
        self._recorded_object.update()
        self.record()

    def finalize(self):
        """Finalize Recording so it cannot be recorded to anymore.

        Converts values in recorded dict from lists to tensors.

        """
        for attr in self.recorded.keys():
            self.recorded[attr] = torch.stack(self.recorded[attr], dim=1)
        self._finalized = True

    def _get_recorded_attr(self, name: str):
        attribute_chain = name.split('.')
        return Recorder._Recorder__get_nested_attr(
            self._recorded_object, attribute_chain
        )

    @staticmethod
    def __get_nested_attr(obj, attribute_chain: List[str]):
        """Resolve nested attributes by name.

        Examples
        -------
        __get_nested_attr(obj, ['a', 'b', 'c']) is equivalent to obj.a.b.c

        """
        name = attribute_chain.pop(0)
        if len(attribute_chain) > 0:
            return Recorder._Recorder__get_nested_attr(
                getattr(obj, name), attribute_chain
            )
        return getattr(obj, name)

    def _check_not_finalized(self):
        if self._finalized:
            raise RuntimeError(
                'Recorder has been finalized and should be considered '
                'immutable.'
            )


class Synapse(Subunit):
    def __init__(self, nb_units: int, time_constant: float):
        initial_state = _zeros((Environment.batch_size, nb_units))
        super().__init__(
            time_constant,
            initial_state,
            initial_state,
            lambda linear: linear,
            lambda linear, nonlinear: linear,
        )

    @property
    def nonlinear(self):
        warnings.warn(
            'Synapse has no nonlinearity; use linear attribute instead to '
            'avoid one timestep delay'
        )
        return super().nonlinear


class Neuron:
    _attributes_to_record = (
        'somatic_synapse.linear',
        'somatic_subunit.linear',
        'output',
    )

    def __init__(self, parameters: NeuronParameters, nb_units: int):
        self.somatic_synapse = Synapse(nb_units, parameters.tau_syn)
        self.somatic_subunit = Subunit(
            parameters.tau_mem,
            _zeros((Environment.batch_size, nb_units)),
            _zeros((Environment.batch_size, nb_units)),
            parameters.somatic_spike_fn,
            lambda linear, nonlinear: linear
            - parameters.feedback_strength * nonlinear.detach(),
        )

    def integrate_input(self, incoming_spikes: torch.Tensor):
        self.somatic_synapse.integrate_input(incoming_spikes)
        self.somatic_subunit.integrate_input(self.somatic_synapse.linear)

    @property
    def output(self):
        return self.somatic_subunit.nonlinear

    def update(self):
        self.somatic_synapse.update()
        self.somatic_subunit.update()

    def get_recorder(self) -> Recorder:
        return Recorder(self, self._attributes_to_record)


class NonSpikingNeuron(Neuron):
    """Identical to a Neuron, but does not spike."""

    _attributes_to_record = ('somatic_synapse.linear', 'output')

    def __init__(self, parameters: NeuronParameters, nb_units: int):
        """Initialize the NonSpikingNeuron.

        Note: parameters.somatic_spike_fn is ignored.

        """
        self.somatic_synapse = Synapse(nb_units, parameters.tau_syn)
        self.somatic_subunit = Subunit(
            parameters.tau_mem,
            _zeros((Environment.batch_size, nb_units)),
            _zeros((Environment.batch_size, nb_units)),
            # Manually override spike and reset functions
            lambda linear: linear,
            lambda linear, nonlinear: linear,
        )

    @property
    def output(self):
        return self.somatic_subunit.linear


class TwoCompartmentNeuron(Neuron):
    _attributes_to_record = (
        'dendritic_synapse.linear',
        'dendritic_subunit.linear',
        'dendritic_subunit.nonlinear',
        'somatic_synapse.linear',
        'somatic_subunit.linear',
        'output',
    )

    def __init__(
        self, parameters: TwoCompartmentNeuronParameters, nb_units: int
    ):
        super().__init__(parameters, nb_units)
        self.dendritic_synapse = Synapse(nb_units, parameters.tau_syn)
        self.dendritic_subunit = Subunit(
            parameters.tau_mem,
            _zeros((Environment.batch_size, nb_units)),
            _zeros((Environment.batch_size, nb_units)),
            parameters.dendritic_spike_fn,
            lambda linear, nonlinear: linear,  # No feedback
        )

    def integrate_input(self, weighted_incoming_spikes: torch.Tensor):
        """Integrate weighted input from previous layer.

        Parameters
        ----------
        weighted_incoming_spikes
            Last axis of tensor should have a size of 2. First index is passed
            to dendrites, second index is passed to soma.

        """
        if weighted_incoming_spikes.shape[-1] != 2:
            raise ValueError(
                'Expected last axis of weighted_incoming_spikes to have size 2'
            )

        self.dendritic_synapse.integrate_input(
            weighted_incoming_spikes[..., 0]
        )
        self.dendritic_subunit.integrate_input(self.dendritic_synapse.linear)
        self.somatic_synapse.integrate_input(weighted_incoming_spikes[..., 1])
        self.somatic_subunit.integrate_input(
            self.somatic_synapse.linear + self.dendritic_subunit.nonlinear
        )

    def update(self):
        super().update()
        self.dendritic_synapse.update()
        self.dendritic_subunit.update()


class RecurrentNeuron(TwoCompartmentNeuron):
    def __init__(self, parameters: RecurrentNeuronParameters, nb_units: int):
        super().__init__(parameters, nb_units)
        self._backprop_gain = parameters.backprop_gain

    def integrate_input(self, weighted_incoming_spikes: torch.Tensor):
        """Integrate weighted input from previous layer.

        Parameters
        ----------
        weighted_incoming_spikes
            Last axis of tensor should have a size of 2. First index is passed
            to dendrites, second index is passed to soma.

        """
        if weighted_incoming_spikes.shape[-1] != 2:
            raise ValueError(
                'Expected last axis of weighted_incoming_spikes to have size 2'
            )

        self.dendritic_synapse.integrate_input(
            weighted_incoming_spikes[..., 0]
        )
        self.dendritic_subunit.integrate_input(
            self.dendritic_synapse.linear
            + self._backprop_gain * self.somatic_subunit.nonlinear
        )
        self.somatic_synapse.integrate_input(weighted_incoming_spikes[..., 1])
        self.somatic_subunit.integrate_input(
            self.somatic_synapse.linear + self.dendritic_subunit.nonlinear
        )


class ParallelNeuron:
    NotImplemented


class PRCNeuron:
    NotImplemented


class SpikingNetwork:
    _hidden_neuron_cls = Neuron
    _output_neuron_cls = NonSpikingNeuron

    def __init__(
        self,
        neuron_parameters: NeuronParameters,
        network_architecture: NetworkArchitecture,
    ):
        self.nb_units_by_layer = network_architecture.nb_units_by_layer

        self.weights_by_layer = [
            None for _ in range(len(self.nb_units_by_layer) - 1)
        ]
        self._initialize_weights(
            network_architecture.weight_scale, neuron_parameters.tau_mem
        )

        self.units_by_layer = []
        self._initialize_units(neuron_parameters)

    def _initialize_units(self, neuron_parameters: NeuronParameters):
        for l in range(1, len(self.nb_units_by_layer)):
            if l < len(self.nb_units_by_layer) - 1:
                # If this is not the last layer, use spiking subunits.
                self.units_by_layer.append(
                    self._hidden_neuron_cls(
                        neuron_parameters, self.nb_units_by_layer[l]
                    )
                )
            else:
                # If this is the last layer, remove spiking nonlinearity.
                self.units_by_layer.append(
                    self._output_neuron_cls(
                        neuron_parameters, self.nb_units_by_layer[l]
                    )
                )

    def _initialize_weights(self, weight_scale, membrane_time_constant):
        discount = _time_constant_to_discount_factor(membrane_time_constant)
        adjusted_weight_scale = weight_scale * (1.0 - discount)

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

    def run_snn(self, inputs):
        weighted_spikes_l1 = torch.einsum(
            # [b]atches, [t]ime, [i]nput units, [h]idden units
            "bti,ih->bth",
            (inputs, self.weights_by_layer[0]),
        )
        recorder_l1 = self._run_layer(
            weighted_spikes_l1, self.units_by_layer[0]
        )

        # Readout layer
        weighted_spikes_l2 = torch.einsum(
            # [b]atch, [t]ime, [h]idden, [o]utput
            "bth,ho->bto",
            (recorder_l1.recorded['output'], self.weights_by_layer[1]),
        )
        recorder_l2 = self._run_layer(
            weighted_spikes_l2, self.units_by_layer[1]
        )

        out_rec = recorder_l2.recorded['output']
        other_recs = {
            'l1': recorder_l1,
            'l2': recorder_l2,
        }

        return out_rec, other_recs

    @staticmethod
    def _run_layer(input_: torch.Tensor, units: Neuron) -> Recorder[Neuron]:
        recorder: Recorder[Neuron] = units.get_recorder()
        for t in range(Environment.nb_steps):
            units.integrate_input(input_[:, t, ...])
            recorder.update_and_record()

        recorder.finalize()

        return recorder


class TwoCompartmentSpikingNetwork(SpikingNetwork):
    _hidden_neuron_cls = TwoCompartmentNeuron

    def _initialize_weights(self, weight_scale, membrane_time_constant):
        adjusted_weight_scale = weight_scale * (1.0 - membrane_time_constant)

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

    def run_snn(self, inputs):
        weighted_spikes_l1 = torch.einsum(
            # [b]atches, [t]ime, [i]nput units, [h]idden units, [c]ompartments
            "bti,ihc->bthc",
            (inputs, self.weights_by_layer[0]),
        )
        recorder_l1 = self._run_layer(
            weighted_spikes_l1, self.units_by_layer[0]
        )

        # Readout layer
        weighted_spikes_l2 = torch.einsum(
            # [b]atch, [t]ime, [h]idden, [o]utput
            "bth,ho->bto",
            (recorder_l1.recorded['output'], self.weights_by_layer[1]),
        )
        recorder_l2 = self._run_layer(
            weighted_spikes_l2, self.units_by_layer[1]
        )

        out_rec = recorder_l2.recorded['output']
        other_recs = {
            'l1': recorder_l1,
            'l2': recorder_l2,
        }

        return out_rec, other_recs


class RecurrentSpikingNetwork(TwoCompartmentSpikingNetwork):
    # RecurrentSpikingNetwork inherits all its implementation from
    # TwoCompartmentSpikingNetwork
    _hidden_neuron_cls = RecurrentNeuron


class ParallelSpikingNetwork(TwoCompartmentSpikingNetwork):
    # ParallelSpikingNetwork inherits all its implementation from
    # TwoCompartmentSpikingNetwork
    _hidden_neuron_cls = ParallelNeuron


class PRCSpikingNetwork(ParallelSpikingNetwork, RecurrentSpikingNetwork):
    # PRCSpikingNetwork inherits all its implementation from its parents.
    _hidden_neuron_cls = PRCNeuron
