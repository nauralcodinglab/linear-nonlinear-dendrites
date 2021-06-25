import os
from copy import deepcopy
from typing import Tuple, Dict
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
import gzip
import shutil
import hashlib
import urllib.request
from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve
from tqdm import trange

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


class DefaultOptimizer:
    def __init__(self, forward_fn, params):
        self._forward = forward_fn
        self.params = params
        self.optimizer = torch.optim.Adamax(params, lr=2e-3, betas=(0.9, 0.999))

        log_softmax_fn = nn.LogSoftmax(dim=1)
        neg_log_lik_fn = nn.NLLLoss()

        def loss_fn(epochs):
            for e in trange(epochs):
                for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_train, y_train, Environment_batch_size,
                                                                               Environment_nb_steps, nb_inputs,
                                                                               max_time):
                    actual_output = self._forward(x_local.to_dense())
                    m, _ = torch.max(actual_output, 1)
                    log_p_y = log_softmax_fn(m)

                    loss_val = neg_log_lik_fn(log_p_y, y_local.long())
                    return loss_val

        self.loss_fn = loss_fn
        self.loss_history = []

    def optimize(self, epochs):
        for e in trange(epochs):
            loss_val = self.loss_fn(epochs)
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            self.loss_history.append(loss_val.item())

network_architecture = NetworkArchitecture()
Environment.nb_steps = 100
network_architecture.nb_units_by_layer = (700, 200, 20)
nb_inputs = network_architecture.nb_units_by_layer[0]
NUM_SEEDS = 10
EPOCHS = 300

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def get_shd_dataset(cache_dir, cache_subdir):

    # The remote directory with the data files
    base_url = "https://compneuro.net/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}
    # Download the Spiking Heidelberg Digits (SHD) dataset
    files = ["shd_train.h5.gz",
            "shd_test.h5.gz",
            ]
    for fn in files:
        origin = "%s/%s" % (base_url, fn)
        hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn], cache_dir=cache_dir,
                                            cache_subdir=cache_subdir)
        print("File %s decompressed to:" % (fn))
        print(hdf5_file_path)

def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None, cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s" % gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path

def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False

def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()

def get_file(fname,
                origin,
                md5_hash=None,
                file_hash=None,
                cache_subdir='datasets',
                hash_algorithm='auto',
                extract=False,
                archive_format='auto',
                cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                        'incomplete or outdated because the ' + hash_algorithm +
                        ' file hash does not match the original value of ' + file_hash +
                        ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
        try:
            urlretrieve(origin, fpath)
        except HTTPError as e:
            raise Exception(error_msg.format(origin, e.code, e.msg))
        except URLError as e:
            raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(fpath):
            os.remove(fpath)

    return fpath

cache_dir = os.path.expanduser("~/data")
cache_subdir = "hdspikes"
get_shd_dataset(cache_dir, cache_subdir)

train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')

x_train = train_file['spikes']
y_train = train_file['labels']
x_test = test_file['spikes']
y_test = test_file['labels']

Environment_batch_size = Environment.batch_size
Environment_nb_steps = Environment.nb_steps
network_architecture_nb_units_by_layer= network_architecture.nb_units_by_layer
max_time = 1.4


def sparse_data_generator_from_hdf5_spikes(X, y, Environment_batch_size, Environment_nb_steps,
                                           network_architecture_nb_units_by_layer, max_time, shuffle=True):
    """ This generator takes a spike dataset and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(labels_) // Environment.batch_size
    sample_index = np.arange(len(labels_))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(0, max_time, num=Environment.nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[Environment.batch_size * counter:Environment.batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size(
            [Environment.batch_size, Environment.nb_steps, network_architecture_nb_units_by_layer])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1

nb_inputs = network_architecture.nb_units_by_layer[0]

def get_mini_batch(x_data, y_data, shuffle=False):
    for ret in sparse_data_generator_from_hdf5_spikes(x_data, y_data, Environment_batch_size, Environment_nb_steps, nb_inputs, max_time, shuffle=shuffle):
        return ret

x_batch, y_batch = get_mini_batch(x_test, y_test)

def main():
    """Run training loop across multiple random seeds in parallel."""
    with mp.Pool(3) as pool:
        pool.map(worker, range(NUM_SEEDS))

def worker(rep_num: int):
    nets, optimizers = train_networks(rep_num)
    save_loss_history(optimizers, r'C:\Users\Anish Goel\Downloads\lnl_project\memorization_training_results_{rep_num}.csv')

def train_networks(
    rep_num: int, epochs: int = EPOCHS, set_seed: bool = True
) -> Tuple[Dict[str, SpikingNetwork], Dict[str, DefaultOptimizer]]:
    """Train a set of PRC models to memorize a random dataset."""
    if set_seed:
        torch.manual_seed(rep_num)

    nets = get_networks()
    optimizers = get_optimizers(nets)

    for label in nets:
        print(f'Training \"{label}\" - {rep_num}')
        initial_train_accuracy = compute_classification_accuracy(x_train, y_train, nets[label])
        initial_test_accuracy = compute_classification_accuracy(x_test, y_test, nets[label])
        optimizers[label].optimize(epochs)
        final_train_accuracy = compute_classification_accuracy(x_train, y_train, nets[label])
        final_test_accuracy = compute_classification_accuracy(x_test, y_test, nets[label])
        print(
            f'Finished training \"{label}\" - {rep_num}; '
            f'Initial Train Acc. {100 * initial_train_accuracy:.1f}%, '
            f'Initial Test Acc. {100 * initial_test_accuracy:.1f}%.'
            f'Final Train Acc. {100 * final_train_accuracy:.1f}%.'
            f'Final Test Acc. {100 *  final_test_accuracy:.1f}%.'
        )

    return nets, optimizers


def save_loss_history(
    optimizers: Dict[str, DefaultOptimizer], fname: str
) -> None:
    """Save loss during training to CSV file."""
    data = {'model_name': [], 'epoch': [], 'loss': []}

    for label, optimizer in optimizers.items():
        num_epochs = len(optimizer.loss_history)

        data['model_name'].extend([label] * num_epochs)
        data['epoch'].extend(range(num_epochs))
        data['loss'].extend(optimizer.loss_history)


    data_df = pd.DataFrame(data)
    data_df.to_csv(fname, index=False)

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

    simple_network_architecture = deepcopy(network_architecture)
    simple_network_architecture.weight_scale_by_layer = (3, 7)

    two_compartment_network_architecture = deepcopy(network_architecture)
    two_compartment_network_architecture.weight_scale_by_layer = (0.5, 7)

    parallel_network_architecture = deepcopy(network_architecture)
    parallel_network_architecture.weight_scale_by_layer = (0.02, 7)

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

def compute_classification_accuracy(x_data, y_data, net: SpikingNetwork) -> float:
    """ Computing classification accuracy on supplied data for each of the networks. """
    accs = []
    for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data, y_data, Environment_batch_size, Environment_nb_steps, nb_inputs, max_time, shuffle=False):
        output,_ = net.run_snn(x_local.to_dense())
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)


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
