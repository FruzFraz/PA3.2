from typing import Any, Dict, List, Optional, Tuple, Union

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from plotid.publish import publish
from plotid.tagplot import tagplot


def generate_group_name(
    controller: Union[str, List[str]],
    topology: Union[str, List[str]],
    disruption: Union[str, List[str]],
) -> List[str]:
    pass
    if isinstance(controller, str):
        controller = [controller]
    if isinstance(topology, str):
        topology = [topology]
    if isinstance(disruption, str):
        disruption = [disruption]

    return[f"{c}_{t}_{d}" for c in controller for t in topology for d in disruption]     


def read_metadata(file: str, path: str, attr_key: str) -> Any:
    try:
        with h5.File(file, "r") as hdf:
            if path not in hdf:
                print(f"Warning: Path '{path}' not found.")
                return None
            if attr_key not in hdf[path].attrs:
                print(f"Warning: Attribute '{attr_key}' not found at '{path}'.")
                return None
            return hdf[path].attrs[attr_key]
    except OSError as e:
        print(f"Warning: {e}")
        return None


def read_data(file: str, path: str) -> Optional[NDArray]:
    """Read data from HDF5 dataset and return as 1D numpy array."""
    try:
        with h5.File(file, 'r') as f:
            if path not in f:
                print(f"Warning: Path '{path}' does not exist in HDF5 file.")
                return None
            
            dataset = f[path]
            
            if isinstance(dataset, h5.Group):
                print(f"Warning: Path '{path}' points to a HDF5 group, not a dataset.")
                return None
            
            
            data = dataset[:]
            return np.atleast_1d(data)
    except Exception as e:
        print(f"Warning: Error reading data from '{path}': {str(e)}")
        return None



def cap_service_data(service_data: NDArray, setpoint: float) -> NDArray:
    return np.array([
        min(max(value, 0.0), setpoint) for value in service_data
    ])


def check_negative_values(array: NDArray) -> bool:
    return np.all(array >= 0)
    pass



def integral_with_time_step(data: NDArray, time_steps: NDArray) -> float:
    if len(data) != len(time_steps):
        print("Warning: Length mismatch.")
        return None
    integral = 0.0
    for i in range(len(data) - 1):
        integral += (data[i] + data[i + 1]) / 2 * (time_steps[i + 1] - time_steps[i])
    return float(integral)
    pass


def calculate_service_loss(service_fill: float, service_target: float) -> float:
    return 100.0 * (1.0 - service_fill / service_target)
pass


def convert_Ws_to_Wh(energy_in_Ws: float) -> float:
    return energy_in_Ws / 3600.0
    pass


def calculate_mean_and_std(data: List[float]) -> Tuple[float, float]:
    arr = np.array(data)
    return float(np.mean(arr)), float(np.std(arr))
pass


def save_dataframe_in_hdf5_with_metadata(
    df: pd.DataFrame,
    hdf5_path: str,
    group_name: str,
    metadata: Dict[str, Any],
) -> None:
    pass


def read_plot_data(
    file_path: str, group_path: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    pass


def plot_service_loss_vs_power(
    processed_data: pd.DataFrame, plot_format_data: Dict[str, str]
) -> Figure:
    pass


def publish_plot(
    fig: Figure, source_paths: Union[str, List[str]], destination_path: str
) -> None:
    pass