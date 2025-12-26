import h5py
import numpy as np
import torch
from helper import features_by_attribute
from torch.utils.data import TensorDataset, random_split
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn_graph
from tqdm import tqdm

# ----------- Preprocessing Pipeline for the ATLAS Jet Tagging Dataset --------------


def list_contents(group, prefix=""):
    """
    Recursively lists the contents of an HDF5 group.
    """
    for key in group.keys():
        item = group[key]
        path = f"{prefix}/{key}"
        print(path)

        # If the item is a Group, recurse into it
        if isinstance(item, h5py.Group):
            list_contents(item, path)
        else:  # Item is a Dataset
            print(f" - Dataset shape: {item.shape}, Dataset type: {item.dtype}")


def validate_user(who):
    """Validate the user role."""
    valid_roles = ["Teacher", "Student"]
    if who not in valid_roles:
        raise ValueError(f"Valid keys for 'who' are {valid_roles}.")


def get_slice_indices(max_items, who):
    """Calculate slice indices based on the user role."""
    if who == "Teacher":
        return slice(None, max_items)
    else:
        return slice(max_items, int(max_items * 2) if max_items is not None else None)


def fetch_dataset(f, key, idx_slice):
    """Fetch and return dataset slices."""
    return np.asarray(f[key][idx_slice])


def get_data(filename, attribute, max_items=None, verbose=False, who="Teacher"):
    """
    Extract data from an ATLAS h5 file based on a specified attribute.

    Parameters:
    - filename (str): Path to the h5 file.
    - attribute (str): One of 'jet', 'constituents', or 'high_level'.
    - max_items (int, optional): Maximum number of items to return. Defaults to None (entire dataset).
    - verbose (bool, optional): If True, prints file contents. Defaults to False.
    - who (str, optional): Role of the user, either 'Teacher' or 'Student'. Defaults to 'Teacher'.

    Returns:
    - tuple: A tuple containing data, labels, weights, and feature names.
    """

    validate_user(who)
    idx_slice = get_slice_indices(max_items, who)

    data = []
    ordered_keys = []  # maintain a specific order of keys
    with h5py.File(filename, "r") as f:
        if verbose:
            print("File contents:", list(f.keys()))

        use_keys = features_by_attribute(attribute)

        for key in use_keys:
            if key in f:
                data.append(fetch_dataset(f, key, idx_slice))
                ordered_keys.append(key)

        # Handle inconsistent naming for weights
        weights_key = "training_weights" if "training_weights" in f else "weights"
        weights = fetch_dataset(f, weights_key, idx_slice)
        labels = fetch_dataset(f, "labels", idx_slice)

    # Adjust the data shape based on the attribute
    if attribute == "jet":
        data = np.stack(data).T  # transposed for 'jet'
    elif attribute == "constituents":
        data = np.stack(data).transpose(1, 2, 0)  # reshaped for 'constituents'

    return data, labels, weights, np.array(ordered_keys)


def constituent_preprocess(data, features, max_constits=None):
    """constituent - This function applies a standard preprocessing to the
    jet data contained in data. It will operate on the raw constituent
    level quantities and return 7 constituent level quantities which can be
    used for tagger training.

    Arguments:
    data (np array) - Array of constituent-level data
    features (list) - List of constituent-level feature names
    # max_constits (int) - The maximum number of constituents to consider in
    # preprocessing. Cut jet constituents at this number.

    Returns:
    (np array) - The seven constituent level quantities, stacked along the last
    axis.

    ADAPTED FROM ATLAS-top-tagging-open-data/preprocessing.py ON GITHUB:
    https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/preprocessing.py?ref_type=heads
    """

    ############################## Load Data ###################################

    # Pull data from data dict
    pt = data[:, :max_constits, np.where(features == "fjet_clus_pt")[0][0]]
    eta = data[:, :max_constits, np.where(features == "fjet_clus_eta")[0][0]]
    phi = data[:, :max_constits, np.where(features == "fjet_clus_phi")[0][0]]
    energy = data[:, :max_constits, np.where(features == "fjet_clus_E")[0][0]]

    # Find location of zero pt entries in each jet. This will be used as a
    # mask to re-zero out entries after all preprocessing steps
    mask = np.asarray(pt == 0).nonzero()

    ########################## Angular Coordinates #############################

    # 1. Center hardest constituent in eta/phi plane. First find eta and
    # phi shifts to be applied
    eta_shift = eta[:, 0]
    phi_shift = phi[:, 0]

    # Apply them using np.newaxis
    eta_center = eta - eta_shift[:, np.newaxis]
    phi_center = phi - phi_shift[:, np.newaxis]

    # Fix discontinuity in phi at +/- pi using np.where
    phi_center = np.where(phi_center > np.pi, phi_center - 2 * np.pi, phi_center)
    phi_center = np.where(phi_center < -np.pi, phi_center + 2 * np.pi, phi_center)

    # 2. Rotate such that 2nd hardest constituent sits on negative phi axis
    second_eta = eta_center[:, 1]
    second_phi = phi_center[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi / 2
    eta_rot = eta_center * np.cos(alpha[:, np.newaxis]) + phi_center * np.sin(alpha[:, np.newaxis])
    phi_rot = -eta_center * np.sin(alpha[:, np.newaxis]) + phi_center * np.cos(alpha[:, np.newaxis])

    # 3. If needed, reflect so 3rd hardest constituent is in positive eta
    third_eta = eta_rot[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta_flip = (eta_rot * parity[:, np.newaxis]).astype(np.float32)
    # Cast to float32 needed to keep numpy from turning eta to double precision

    # 4. Calculate R with pre-processed eta/phi
    radius = np.sqrt(eta_flip**2 + phi_rot**2)

    ############################# pT and Energy ################################

    # Take the logarithm, ignoring -infs which will be set to zero later
    log_pt = np.log(pt)
    log_energy = np.log(energy)

    # Sum pt and energy in each jet
    sum_pt = np.sum(pt, axis=1)
    sum_energy = np.sum(energy, axis=1)

    # Normalize pt and energy and again take logarithm
    lognorm_pt = np.log(pt / sum_pt[:, np.newaxis])
    lognorm_energy = np.log(energy / sum_energy[:, np.newaxis])

    ########################### Finalize and Return ############################

    # Reset all of the original zero entries to zero
    eta_flip[mask] = 0
    phi_rot[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis
    features = [eta_flip, phi_rot, log_pt, log_energy, lognorm_pt, lognorm_energy, radius]
    stacked_data = np.stack(features, axis=-1)

    # Also return feature names
    feature_names = ["delta_eta", "delta_phi", "log_pt", "log_E", "lognorm_pt", "lognorm_E", "R"]

    return stacked_data, np.array(feature_names)


# --------------------- FCNN-specific functions -------------------------------------------


def standardize_split(data, labels, weights, train_split, val_split, seed=None):
    """
    Given data of shape [INPUT_SIZE, NUM_FEATURES, NUM_CONSTITUENTS], labels of shape [INPUT_SIZE], and weights of shape [INPUT_SIZE]

    - cast to PyTorch TensorDataset
    - split into training, validation, and testing

    Returns:
    - training_dataset, val_dataset, test_dataset: PyTorch TensorSubset objects
    """
    # Cast to tensor
    data = torch.tensor(data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.float32)  # Ensure weights are also a tensor

    # Standardization (mean=0, std=1)
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    data = (data - mean) / std

    # get rid of NaNs and infs --> cast to zeros
    data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Wrap into a TensorDataset with weights
    dataset = TensorDataset(data, labels, weights)

    # Split into train, validation, and test sets
    total_length = len(dataset)
    train_length = int(train_split * total_length)
    val_length = int(val_split * total_length)
    test_length = total_length - train_length - val_length  # To handle any rounding issues

    if seed:
        training_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_length, val_length, test_length],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        training_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_length, val_length, test_length]
        )

    return training_dataset, val_dataset, test_dataset


# ------------------------ GNN-specific functions -------------------------------------


class WeightedData(Data):
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight


def prepare_graphs(data, labels, weights, k, device, batch_size=64):
    """
    Given our input data, labels, and training weights, construct graphs with
    the KNN algorithm
    """

    # Convert data to PyTorch tensors and move to the appropriate device
    data_tensor = torch.tensor(data, dtype=torch.float).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.long).to(device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    # Standardize the data (mean=0, std=1)
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)
    data_tensor = (data_tensor - mean) / std

    # get rid of NaNs and infs --> cast to zeros
    data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a dataset and dataloader for batch processing
    dataset = TensorDataset(data_tensor, labels_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    graphs = []
    for data, label, weight in tqdm(loader):
        # Construct graphs for each sample in the batch
        for i in range(data.size(0)):
            edge_index = knn_graph(data[i], k=k, loop=False)
            graph = WeightedData(
                weight=weight[i], x=data[i], edge_index=edge_index, y=label[i].unsqueeze(0)
            )
            graphs.append(graph.to("cpu"))  # for memory management, don't store these on GPU

    return graphs


def split_graphs(graph_list, training_split, val_split):
    """
    Given a GraphDataset object, split it into training, validation, and test sets
    """

    # Calculate sizes for each split
    total_count = len(graph_list)
    train_count = int(training_split * total_count)
    valid_count = int(val_split * total_count)
    test_count = total_count - train_count - valid_count

    # Perform the split using random_split
    train_dataset, valid_dataset, test_dataset = random_split(
        graph_list, [train_count, valid_count, test_count]
    )

    return train_dataset, valid_dataset, test_dataset
