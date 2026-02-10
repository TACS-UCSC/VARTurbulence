import numpy as np
import os
from scipy.io import loadmat
import jax.numpy as jnp
import jax
import jax.random as random
import matplotlib.pyplot as plt
import threading
import queue
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

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

# Load all turbulence data into memory as a pre-stacked numpy array
def load_turbulence_data_mat(data_dir, start_idx=10000, stop_idx=19999, normalize=False):
    samples = []
    print(f"Loading {stop_idx - start_idx + 1} files...")
    for idx in range(start_idx, stop_idx + 1):
        file_path = os.path.join(data_dir, f"{idx}.mat")
        samples.append(loadmat(file_path)['Omega'])

    data = np.stack(samples).astype(np.float32)  # (N, H, W)
    data = data[:, np.newaxis, :, :]              # (N, 1, H, W)

    if normalize:
        mean, std = data.mean(), data.std()
        data = (data - mean) / std

    return data

# Load all turbulence data into memory
def load_turbulence_data_rb_convection(data_loc, 
                                       start_idx=None, 
                                       stop_idx=None, 
                                       normalize=False,
                                       field = "buoyancy",
                                       field_idx = 0):
    ## loading the data as individual idx's into the data
    f = h5py.File(data_loc, 'r')

    # bcs_xperiodic = f["boundary_conditions"]["x_periodic"]
    # bcs_y_wall_dirichlet = f["boundary_conditions"]["y_wall_dirichlet"]

    # loading buoyancy, pressure, and velocity fields

    if field == "buoyancy":
        data = f["t0_fields"]["buoyancy"][field_idx, start_idx:stop_idx, :, :]
    elif field == "pressure":
        data = f["t0_fields"]["pressure"][field_idx, start_idx:stop_idx, :, :]
    elif field == "velocity_x":
        data = f["t1_fields"]["velocity"][field_idx, start_idx:stop_idx, :, :, 0]
    elif field == "velocity_y":
        data = f["t1_fields"]["velocity"][field_idx, start_idx:stop_idx, :, :, 1]
    else:
        raise ValueError(f"Invalid field: {field}")

    f.close()

    data_dict = {}
    for i in range(data.shape[0]):
        data_dict[i] = data[i, :, :]

    return data_dict


class well_dataset_loader(Dataset):
    """
    PyTorch Dataset for turbulence data (supports 'rb_convection' and potentially others).
    
    Supports:
    - Multiple HDF5 data files
    - Flexible start/end indexing per file
    - Multiple field types
    - Optional normalization
    - Temporal sequences with configurable time delta
    - Internal batching and shuffling
    """
    
    def __init__(self, 
                 data_locations,
                 dataset_name="rb_convection",
                 start_idx=None,
                 stop_idx=None,
                 fields=["buoyancy"],
                 normalize=False,
                 dt=None,
                 return_indices=False,
                 batch_size=1,
                 shuffle=False,
                 seed=None):
        """
        Args:
            data_locations (list or str): Path(s) to HDF5 file(s).
            dataset_name (str): Name of the dataset type.
            start_idx (int, list, or None): Starting index for data loading.
            stop_idx (int, list, or None): Ending index for data loading.
            fields (list or str): Field(s) to load. 
            normalize (bool): Whether to normalize data.
            dt (int or None): Time delta.
            return_indices (bool): If True, return indices.
            batch_size (int): Batch size for iteration. If None, yields single samples.
            shuffle (bool): Whether to shuffle data during iteration.
            seed (int): Random seed for shuffling.
        """
        super().__init__()
        
        if isinstance(data_locations, str):
            data_locations = [data_locations]
        self.data_locations = data_locations
        self.dataset_name = dataset_name
        
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        
        # Handle start/stop indices
        num_files = len(data_locations)
        if start_idx is None:
            self.start_indices = [None] * num_files
        elif isinstance(start_idx, int):
            self.start_indices = [start_idx] * num_files
        else:
            assert len(start_idx) == num_files, "start_idx list must match number of files"
            self.start_indices = start_idx
            
        if stop_idx is None:
            self.stop_indices = [None] * num_files
        elif isinstance(stop_idx, int):
            self.stop_indices = [stop_idx] * num_files
        else:
            assert len(stop_idx) == num_files, "stop_idx list must match number of files"
            self.stop_indices = stop_idx
        
        self.normalize = normalize
        self.dt = dt
        self.return_indices = return_indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Load metadata and build index mapping
        self._build_index_mapping()
        
        # Compute normalization statistics if needed
        if self.normalize:
            self._compute_normalization_stats()
            
    def _get_field_shape(self, f, field):
        """Get shape of a field in the HDF5 file based on dataset name."""
        if self.dataset_name == "rb_convection":
            if field == "buoyancy":
                return f["t0_fields"]["buoyancy"].shape
            elif field == "pressure":
                return f["t0_fields"]["pressure"].shape
            elif field in ["velocity", "velocity_x", "velocity_y"]:
                return f["t1_fields"]["velocity"].shape
            else:
                raise ValueError(f"Unknown field '{field}' for dataset '{self.dataset_name}'")
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _get_field_data(self, f, field, sim_slice, time_slice):
        """Get data slice for a field based on dataset name."""
        if self.dataset_name == "rb_convection":
            if field == "buoyancy":
                return f["t0_fields"]["buoyancy"][sim_slice, time_slice]
            elif field == "pressure":
                return f["t0_fields"]["pressure"][sim_slice, time_slice]
            elif field == "velocity":
                return f["t1_fields"]["velocity"][sim_slice, time_slice]
            elif field == "velocity_x":
                return f["t1_fields"]["velocity"][sim_slice, time_slice, ..., 0]
            elif field == "velocity_y":
                return f["t1_fields"]["velocity"][sim_slice, time_slice, ..., 1]
            else:
                raise ValueError(f"Unknown field '{field}' for dataset '{self.dataset_name}'")
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")
    
    def _build_index_mapping(self):
        """Build a mapping from global index to (file_idx, sim_idx, local_idx)"""
        self.index_map = []
        self.file_lengths = []
        
        for file_idx, (data_loc, start, stop) in enumerate(
            zip(self.data_locations, self.start_indices, self.stop_indices)
        ):
            with h5py.File(data_loc, 'r') as f:
                shape = self._get_field_shape(f, self.fields[0])
                num_sims = shape[0]
                num_samples = shape[1]
                print(f"File {file_idx} ({self.dataset_name}): {num_sims} simulations, {num_samples} time steps")
                
                start_eff = start if start is not None else 0
                if stop is None or stop == -1:
                    stop_eff = num_samples
                elif stop < 0:
                    stop_eff = num_samples + stop
                else:
                    stop_eff = stop
                
                stop_eff = min(stop_eff, num_samples)
                length_per_sim = max(0, stop_eff - start_eff)
                
                self.file_lengths.append(length_per_sim * num_sims)
                
                for sim_idx in range(num_sims):
                    for local_idx in range(length_per_sim):
                        actual_idx = start_eff + local_idx
                        self.index_map.append((file_idx, sim_idx, actual_idx))
        
        self.total_length = len(self.index_map)
        
        if self.dt is not None:
            valid_indices = []
            for i in range(len(self.index_map)):
                file_idx, sim_idx, local_idx = self.index_map[i]
                if i + self.dt < len(self.index_map):
                    target_file_idx, target_sim_idx, target_local_idx = self.index_map[i + self.dt]
                    if (target_file_idx == file_idx and 
                        target_sim_idx == sim_idx and 
                        target_local_idx == local_idx + self.dt):
                        valid_indices.append(i)
            
            self.valid_indices = valid_indices
            self.total_length = len(valid_indices)
    
    def _compute_normalization_stats(self):
        """Compute mean and std across all data for normalization per channel"""
        print(f"Computing normalization stats for {self.dataset_name}...")
        
        total_channels = 0
        channel_map = [] 
        
        for field in self.fields:
            if field == "velocity" and self.dataset_name == "rb_convection":
                total_channels += 2
                channel_map.append((field, 2))
            else:
                total_channels += 1
                channel_map.append((field, 1))
                
        all_values = [[] for _ in range(total_channels)]
        
        for file_idx, data_loc in enumerate(self.data_locations):
            start = self.start_indices[file_idx]
            stop = self.stop_indices[file_idx]
            
            with h5py.File(data_loc, 'r') as f:
                shape = self._get_field_shape(f, self.fields[0])
                num_samples = shape[1]
                
                start_eff = start if start is not None else 0
                if stop is None or stop == -1:
                    stop_eff = num_samples
                elif stop < 0:
                    stop_eff = num_samples + stop
                else:
                    stop_eff = stop
                stop_eff = min(stop_eff, num_samples)
                
                if stop_eff <= start_eff:
                    continue

                for field in self.fields:
                    curr_channel_offset = 0
                    for fname, fch in channel_map:
                        if fname == field:
                            break
                        curr_channel_offset += fch
                    
                    data = self._get_field_data(f, field, slice(None), slice(start_eff, stop_eff))
                    
                    if field == "velocity" and self.dataset_name == "rb_convection":
                        all_values[curr_channel_offset].append(data[..., 0].flatten())
                        all_values[curr_channel_offset+1].append(data[..., 1].flatten())
                    else:
                        all_values[curr_channel_offset].append(data.flatten())
        
        self.mean = np.zeros(total_channels, dtype=np.float32)
        self.std = np.zeros(total_channels, dtype=np.float32)
        
        for c in range(total_channels):
            vals = np.concatenate(all_values[c])
            self.mean[c] = np.mean(vals)
            self.std[c] = np.std(vals)
            print(f"Channel {c} stats: mean={self.mean[c]:.4f}, std={self.std[c]:.4f}")
            
    def _load_sample(self, file_idx, sim_idx, sample_idx):
        """Load a single sample from a specific file"""
        data_loc = self.data_locations[file_idx]
        components = []
        
        with h5py.File(data_loc, 'r') as f:
            for field in self.fields:
                data = self._get_field_data(f, field, sim_idx, sample_idx)
                
                if field == "velocity" and self.dataset_name == "rb_convection":
                    components.append(np.transpose(data, (2, 0, 1)))
                else:
                    components.append(data[np.newaxis, ...])
        
        data = np.concatenate(components, axis=0) # (C, H, W)
        data = data.astype(np.float32)
        
        if self.normalize:
            data = (data - self.mean[:, None, None]) / self.std[:, None, None]
        
        return torch.from_numpy(data)
    
    def __len__(self):
        # Return batch count if batching is used, or sample count?
        # Usually len(dataset) is sample count.
        # But for iteration, len(loader) is usually batch count.
        # Here we are mixing Dataset and Loader.
        # To avoid confusion, let's keep __len__ as sample count, 
        # but usage in training script should know if it's iterating batches.
        # In train_vqvae_well.py, steps_per_epoch = len(data_loader).
        # If I change iteration to batches, len(data_loader) should reflect batches.
        if self.batch_size is not None and self.batch_size > 0:
            return (self.total_length + self.batch_size - 1) // self.batch_size
        return self.total_length
    
    def __getitem__(self, idx):
        if self.dt is not None:
            actual_idx = self.valid_indices[idx]
            file_idx, sim_idx, sample_idx = self.index_map[actual_idx]
            
            input_data = self._load_sample(file_idx, sim_idx, sample_idx)
            target_data = self._load_sample(file_idx, sim_idx, sample_idx + self.dt)
            
            if self.return_indices:
                return (file_idx, sim_idx, sample_idx), input_data, target_data
            else:
                return input_data, target_data
        else:
            file_idx, sim_idx, sample_idx = self.index_map[idx]
            data = self._load_sample(file_idx, sim_idx, sample_idx)
            
            if self.return_indices:
                return (file_idx, sim_idx, sample_idx), data
            else:
                return data

    def __iter__(self):
        """Iterate over the dataset with batching."""
        indices = np.arange(self.total_length)
        if self.shuffle:
            if self.seed is not None:
                np.random.seed(self.seed)
            np.random.shuffle(indices)
            
        if self.batch_size is None or self.batch_size <= 0:
            for idx in indices:
                yield self[idx]
        else:
            for start_idx in range(0, self.total_length, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.total_length)
                batch_indices = indices[start_idx:end_idx]
                
                batch_items = [self[i] for i in batch_indices]
                
                if not batch_items:
                    continue
                    
                first_item = batch_items[0]
                
                if isinstance(first_item, torch.Tensor):
                    yield torch.stack(batch_items)
                elif isinstance(first_item, tuple):
                    transposed = zip(*batch_items)
                    yield tuple(torch.stack(items) if isinstance(items[0], torch.Tensor) else items 
                                for items in transposed)
                else:
                    yield batch_items


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
