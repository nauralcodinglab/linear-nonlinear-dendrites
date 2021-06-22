#!/usr/bin/env python
"""Train spiking neural nets to memorize random inputs.

This script runs the task from SpyTorch tutorial 1 over multiple random
datasets and initial weights.

"""
from typing import Tuple, Dict
from copy import deepcopy
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch

from model_components import (
    get_spike_fn,
    get_default_dendritic_fn,
    get_sigmoid_fn,
    SpikingNetwork,
    RecurrentSpikingNetwork,
    TwoCompartmentSpikingNetwork,
    RecurrentNeuronParameters,
    ParallelSpikingNetwork,
    PRCSpikingNetwork,
    PRCNeuronParameters,
    NetworkArchitecture,
    Environment,
)
from optimizers import DefaultOptimizer

NETWORK_ARCHITECTURE = NetworkArchitecture()
NUM_SEEDS = 10
EPOCHS = 600


def main():
    """Run training loop across multiple random seeds in parallel."""
    with mp.Pool(3) as pool:
        pool.map(worker, range(NUM_SEEDS))


def worker(rep_num: int):
    nets, optimizers = train_networks(rep_num)
    save_loss_history(
        optimizers, f'../data/memorization_training_results_{rep_num}.csv'
    )


def train_networks(
    rep_num: int, epochs: int = EPOCHS, set_seed: bool = True
) -> Tuple[Dict[str, SpikingNetwork], Dict[str, DefaultOptimizer]]:
    """Train a set of PRC models to memorize a random dataset."""
    if set_seed:
        torch.manual_seed(rep_num)

    nets = get_networks()
    optimizers = get_optimizers(nets)
    x_data, y_data = generate_data()

    for label in nets:
        print(f'Training \"{label}\" - {rep_num}')
        initial_accuracy = classification_accuracy(x_data, y_data, nets[label])
        optimizers[label].optimize(x_data, y_data, epochs, progress_bar='none')
        final_accuracy = classification_accuracy(x_data, y_data, nets[label])
        print(
            f'Finished training \"{label}\" - {rep_num}; '
            f'initial acc. {100 * initial_accuracy:.1f}%, '
            f'final acc. {100 * final_accuracy:.1f}%.'
        )

    return nets, optimizers


def save_loss_history(
    optimizers: Dict[str, DefaultOptimizer], fname: str
) -> None:
    """Save loss and accuracy during training to CSV file."""
    data = {'model_name': [], 'epoch': [], 'loss': [], 'accuracy': []}

    for label, optimizer in optimizers.items():
        assert len(optimizer.loss_history) == len(optimizer.accuracy_history)
        num_epochs = len(optimizer.loss_history)

        data['model_name'].extend([label] * num_epochs)
        data['epoch'].extend(range(num_epochs))
        data['loss'].extend(optimizer.loss_history)
        data['accuracy'].extend(optimizer.accuracy_history)

    data_df = pd.DataFrame(data)
    data_df.to_csv(fname, index=False)


def generate_data(
    input_freq_hz: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a random dataset to memorize."""
    # Generate random input spiketrains
    prob = input_freq_hz * Environment.time_step
    mask = torch.rand(
        (
            Environment.batch_size,
            Environment.nb_steps,
            NETWORK_ARCHITECTURE.nb_units_by_layer[0],
        ),
        device=Environment.device,
        dtype=Environment.dtype,
    )
    x_data = torch.zeros(
        (
            Environment.batch_size,
            Environment.nb_steps,
            NETWORK_ARCHITECTURE.nb_units_by_layer[0],
        ),
        device=Environment.device,
        dtype=Environment.dtype,
        requires_grad=False,
    )
    x_data[mask < prob] = 1.0

    # Generate random labels
    y_data = torch.tensor(
        1 * (np.random.rand(Environment.batch_size) < 0.5),
        device=Environment.device,
    )

    return x_data, y_data


def get_networks() -> Dict[str, SpikingNetwork]:
    """Get a set of spiking networks to train."""
    somatic_spike_fn = get_spike_fn(threshold=15)
    dendritic_nl_fn = get_default_dendritic_fn(
        threshold=2, sensitivity=10, gain=1
    )
    neuron_params = RecurrentNeuronParameters(
        tau_mem=10e-3,
        tau_syn=5e-3,
        backprop_gain=0.5,
        feedback_strength=15,
        somatic_spike_fn=somatic_spike_fn,
        dendritic_spike_fn=dendritic_nl_fn,
    )

    parallel_params = PRCNeuronParameters(
        tau_mem=10e-3,
        tau_syn=5e-3,
        backprop_gain=0.05,
        feedback_strength=15,
        somatic_spike_fn=somatic_spike_fn,
        dend_na_fn=dendritic_nl_fn,
        dend_ca_fn=get_sigmoid_fn(threshold=4, sensitivity=10, gain=1),
        dend_nmda_fn=dendritic_nl_fn,
        tau_dend_na=5e-3,
        tau_dend_ca=40e-3,
        tau_dend_nmda=80e-3,
    )

    simple_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    simple_network_architecture.weight_scale_by_layer = (140, 7)

    two_compartment_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    two_compartment_network_architecture.weight_scale_by_layer = (15, 7)

    parallel_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    parallel_network_architecture.weight_scale_by_layer = (1, 7)

    nets = {
        'One compartment': SpikingNetwork(
            neuron_params, simple_network_architecture
        ),
        'No BAP': TwoCompartmentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        'BAP': RecurrentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        'Parallel subunits, no BAP': ParallelSpikingNetwork(
            parallel_params, parallel_network_architecture
        ),
        'Parallel subunits + BAP (full PRC model)': PRCSpikingNetwork(
            parallel_params, parallel_network_architecture
        ),
    }

    return nets


def classification_accuracy(x_data, y_data, net: SpikingNetwork) -> float:
    """Compute the classification accuracy."""
    output, _ = net.run_snn(x_data)
    m, _ = torch.max(output, 1)  # max over time
    _, am = torch.max(m, 1)  # argmax over output units
    acc = np.mean((y_data == am).detach().cpu().numpy())  # compare to labels
    return acc


def get_optimizers(
    nets: Dict[str, SpikingNetwork]
) -> Dict[str, DefaultOptimizer]:
    return {
        key: DefaultOptimizer(
            _get_feedforward_func(nets[key]), nets[key].weights_by_layer
        )
        for key in nets
    }


def _get_feedforward_func(net):
    def feedforward(x):
        return net.run_snn(x, reset=True)[0]

    return feedforward


if __name__ == '__main__':
    main()
