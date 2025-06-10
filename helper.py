import h5py
import numpy as np


def load_h5_data(filename, verbose=True):
    """
    Opens an h5py data file and creates variables named after the keys in the data.

    Parameters:
    -----------
    filename : str
        Path to the h5py data file
    verbose : bool, optional
        If True, prints the keys and data shape of each key. Default is False.

    Returns:
    --------
    data_dict : dict
        Dictionary containing all the data from the h5py file, with keys matching the original file keys
    """

    data_file = h5py.File(filename, "r")

    data_dict = {}

    if verbose:
        print(f"File: {filename}")
        print("Keys:")

    for key in data_file.keys():
        # flatten h5py
        data_dict[key] = data_file[key][...]

        if verbose:
            print(
                f"  - {key}: shape {data_dict[key].shape}, dtype {data_dict[key].dtype}"
            )

    data_file.close()

    return data_dict
