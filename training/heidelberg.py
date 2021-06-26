import os
from copy import deepcopy
from typing import Tuple, Dict, Optional
import multiprocessing as mp
import itertools

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
from memorize import get_optimizers
from memorize import (
    classification_accuracy as _minibatch_classification_accuracy,
)


NETWORK_ARCHITECTURE = NetworkArchitecture((700, 200, 20))
Environment.nb_steps = 100
NUM_SEEDS = 10
EPOCHS = 300
SWEEP_DURATION = 1.4

CACHE_DIR = os.path.expanduser("~/lnl-dendrite-data")
CACHE_SUBDIR = "hdspikes"

class Data:
    def __init__(self, path_to_train_data: str, path_to_test_data: str):
        self._path_to_train = path_to_train_data
        self._path_to_test = path_to_test_data

        self._train_file: Optional[h5py.File] = None
        self._test_file: Optional[h5py.File] = None

        self.x_train: Optional[h5py.Dataset] = None
        self.y_train: Optional[h5py.Dataset] = None
        self.x_test: Optional[h5py.Dataset] = None
        self.y_test: Optional[h5py.Dataset] = None

    def __enter__(self):
        self._train_file = h5py.File(self._path_to_train, 'r')
        self.x_train = self._train_file['spikes']
        self.y_train = self._train_file['labels']

        self._test_file = h5py.File(self._path_to_test, 'r')
        self.x_test = self._test_file['spikes']
        self.y_test = self._test_file['labels']

    def __exit__(self, *err_args):
        self._train_file.close()
        self._test_file.close()
        for a, b in itertools.product(['x', 'y'], ['train', 'test']):
            setattr(self, '_'.join((a, b)), None)


class DefaultOptimizer:
    def __init__(self, forward_fn, params):
        self._forward = forward_fn
        self.params = params
        self.optimizer = torch.optim.Adamax(
            params, lr=2e-3, betas=(0.9, 0.999)
        )

        log_softmax_fn = nn.LogSoftmax(dim=1)
        neg_log_lik_fn = nn.NLLLoss()

        def loss_fn(actual_output, desired_output):
            m, _ = torch.max(actual_output, 1)
            log_p_y = log_softmax_fn(m)
            loss_val = neg_log_lik_fn(log_p_y, desired_output.long())
            return loss_val

        self.loss_fn = loss_fn
        self.loss_history = []

    def optimize(self, input_, desired_output, epochs):
        for e in trange(epochs):
            batch_loss = []
            for batch_x, batch_y in sparse_data_generator_from_hdf5_spikes(
                input_, desired_output, SWEEP_DURATION, shuffle=True
            ):
                actual_output = self._forward(batch_x)

                self.optimizer.zero_grad()
                loss_val = self.loss_fn(actual_output, batch_y)
                loss_val.backward()
                self.optimizer.step()

                batch_loss.append(loss_val.item())

            self.loss_history.append(np.mean(batch_loss))


def get_shd_dataset(cache_dir, cache_subdir):

    # The remote directory with the data files
    base_url = "https://compneuro.net/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {
        line.split()[1]: line.split()[0]
        for line in lines
        if len(line.split()) == 2
    }
    # Download the Spiking Heidelberg Digits (SHD) dataset
    files = [
        "shd_train.h5.gz",
        "shd_test.h5.gz",
    ]
    for fn in files:
        origin = "%s/%s" % (base_url, fn)
        hdf5_file_path = get_and_gunzip(
            origin,
            fn,
            md5hash=file_hashes[fn],
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
        )
        print("File %s decompressed to:" % (fn))
        print(hdf5_file_path)


def get_and_gunzip(
    origin, filename, md5hash=None, cache_dir=None, cache_subdir=None
):
    gz_file_path = get_file(
        filename,
        origin,
        md5_hash=md5hash,
        cache_dir=cache_dir,
        cache_subdir=cache_subdir,
    )
    hdf5_file_path = gz_file_path
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(
        gz_file_path
    ) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s" % gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(
            hdf5_file_path, 'wb'
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (
        algorithm == 'auto' and len(file_hash) == 64
    ):
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


def get_file(
    fname,
    origin,
    md5_hash=None,
    file_hash=None,
    cache_subdir='datasets',
    hash_algorithm='auto',
    extract=False,
    archive_format='auto',
    cache_dir=None,
):
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
                print(
                    'A local file was found, but it seems to be '
                    'incomplete or outdated because the '
                    + hash_algorithm
                    + ' file hash does not match the original value of '
                    + file_hash
                    + ' so we will re-download the data.'
                )
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



def sparse_data_generator_from_hdf5_spikes(
    X, y, sweep_duration: float, shuffle=True,
):
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

    time_bins = np.linspace(0, sweep_duration, num=Environment.nb_steps)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[
            Environment.batch_size
            * counter : Environment.batch_size
            * (counter + 1)
        ]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(Environment.device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(Environment.device)

        X_batch = torch.sparse.FloatTensor(
            i,
            v,
            torch.Size(
                [
                    Environment.batch_size,
                    Environment.nb_steps,
                    NETWORK_ARCHITECTURE.nb_units_by_layer[0],
                ]
            ),
        ).to(Environment.device)
        y_batch = torch.tensor(labels_[batch_index], device=Environment.device)

        yield X_batch.to(device=Environment.device), y_batch.to(
            device=Environment.device
        )

        counter += 1


def main():
    """Run training loop across multiple random seeds in parallel."""
    get_shd_dataset(CACHE_DIR, CACHE_SUBDIR)
    with mp.Pool(3) as pool:
        pool.map(worker, range(NUM_SEEDS))


def worker(rep_num: int):
    nets, optimizers = train_networks(rep_num)
    save_loss_history(
        optimizers,
        r'C:\Users\Anish Goel\Downloads\lnl_project\memorization_training_results_{rep_num}.csv',
    )


def train_networks(
    rep_num: int, epochs: int = EPOCHS, set_seed: bool = True
) -> Tuple[Dict[str, SpikingNetwork], Dict[str, DefaultOptimizer]]:
    """Train a set of PRC models to memorize a random dataset."""
    if set_seed:
        torch.manual_seed(rep_num)

    nets = get_networks()
    optimizers = get_optimizers(nets)

    path_to_train_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_train.h5')
    path_to_test_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_test.h5')

    with Data(path_to_train_data, path_to_test_data) as data:
        for label in nets:
            print(f'Training \"{label}\" - {rep_num}')
            initial_train_accuracy = classification_accuracy(
                data.x_train, data.y_train, nets[label]
            )
            initial_test_accuracy = classification_accuracy(
                data.x_test, data.y_test, nets[label]
            )
            optimizers[label].optimize(epochs)
            final_train_accuracy = classification_accuracy(
                data.x_train, data.y_train, nets[label]
            )
            final_test_accuracy = classification_accuracy(
                data.x_test, data.y_test, nets[label]
            )

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

    simple_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    simple_network_architecture.weight_scale_by_layer = (3, 7)

    two_compartment_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    two_compartment_network_architecture.weight_scale_by_layer = (0.5, 7)

    parallel_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
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


def classification_accuracy(x_data, y_data, net: SpikingNetwork) -> float:
    """ Computing classification accuracy on supplied data for each of the networks. """
    accuracies = []
    for x_local, y_local in sparse_data_generator_from_hdf5_spikes(
        x_data,
        y_data,
        SWEEP_DURATION,
        shuffle=False,
    ):
        accuracies.append(
            _minibatch_classification_accuracy(
                x_local.to_dense(), y_local, net
            )
        )
    return np.mean(accuracies)


if __name__ == '__main__':
    main()
