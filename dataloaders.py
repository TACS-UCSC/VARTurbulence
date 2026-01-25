import numpy as np
import os
from scipy.io import loadmat
import jax.numpy as jnp
import jax
import jax.random as random
import matplotlib.pyplot as plt
import threading
import queue


class PrefetchIterator:
    """Wraps an iterator to prefetch batches in a background thread."""

    def __init__(self, iterator, prefetch_count=2):
        self.iterator = iterator
        self.prefetch_count = prefetch_count
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self):
        try:
            for item in self.iterator:
                if self.stop_event.is_set():
                    break
                # Stage data on device asynchronously
                inputs, targets = item
                inputs = jax.device_put(inputs)
                targets = jax.device_put(targets)
                self.queue.put((inputs, targets))
        except Exception as e:
            self.queue.put(e)
        finally:
            self.queue.put(None)  # Sentinel to signal end

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item

    def __del__(self):
        self.stop_event.set()

# Load all turbulence data into memory
def load_turbulence_data(data_dir, start_idx=10000, stop_idx=19999, normalize=False):
    data = {}
    print(f"Loading {stop_idx - start_idx + 1} files...")
    for idx in range(start_idx, stop_idx + 1):
        file_path = os.path.join(data_dir, f"{idx}.mat")
        data[idx] = loadmat(file_path)['Omega']

    if normalize:
        all_data = np.stack(list(data.values()))
        mean, std = np.mean(all_data), np.std(all_data)
        for idx in data:
            data[idx] = (data[idx] - mean) / std

    return data

# Generator that yields batches (internal, not prefetched)
def _batch_generator(data, batch_size, dt=5, shuffle=True, seed=0):
    # Create valid pairs of indices (input, target) separated by dt
    indices = sorted(list(data.keys()))
    pairs = [(indices[i], indices[i+dt]) for i in range(len(indices)-dt)
             if indices[i+dt] - indices[i] == dt]

    num_samples = len(pairs)
    num_batches = num_samples // batch_size

    # Initialize random key
    key = random.PRNGKey(seed)

    # Shuffle all indices if needed
    if shuffle:
        key, subkey = random.split(key)
        perm = random.permutation(subkey, num_samples)
        shuffled_pairs = [pairs[i] for i in perm]
    else:
        shuffled_pairs = pairs

    # Yield batches
    for i in range(num_batches):
        batch_pairs = shuffled_pairs[i * batch_size:(i + 1) * batch_size]

        inputs = []
        targets = []
        for input_idx, target_idx in batch_pairs:
            inputs.append(data[input_idx])
            targets.append(data[target_idx])

        yield jnp.array(np.expand_dims(inputs, axis=1)), jnp.array(np.expand_dims(targets, axis=1))


# Create a dataloader that yields random batches with prefetching
def create_turbulence_dataloader(data, batch_size, dt=5, shuffle=True, seed=0, prefetch=2):
    """Create a dataloader with optional prefetching.

    Args:
        prefetch: Number of batches to prefetch. Set to 0 to disable prefetching.
    """
    gen = _batch_generator(data, batch_size, dt, shuffle, seed)
    if prefetch > 0:
        return PrefetchIterator(gen, prefetch_count=prefetch)
    return gen
