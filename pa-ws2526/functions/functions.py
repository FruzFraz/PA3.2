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
    return np.clip(service_data, 0.0, setpoint)


def check_negative_values(array: NDArray) -> bool:
    return np.all(array >= 0)




def integral_with_time_step(data: NDArray, time_steps: NDArray) -> float:
    if len(data) != len(time_steps):
        print("Warning: Length mismatch.")
        return None
    integral = 0.0
    for i in range(len(data) - 1):
        integral += (data[i] + data[i + 1]) / 2 * (time_steps[i + 1] - time_steps[i])
    return float(integral)



def calculate_service_loss(service_fill: float, service_target: float) -> float:
    if service_target == 0:
        return 0.0
    return 100.0 * (1.0 - service_fill / service_target)



def convert_Ws_to_Wh(energy_in_Ws: float) -> float:
    return energy_in_Ws / 3600.0
    


def calculate_mean_and_std(
    data: List[float]
) -> Tuple[float, float]:
    if len(data) == 0:  # >>> ADDED
        print("Warning: Empty data list.")
        return 0.0, 0.0
    
    arr = np.array(data)
    return float(np.mean(arr)), float(np.std(arr))



def save_dataframe_in_hdf5_with_metadata(
    df: pd.DataFrame,
    hdf5_path: str,
    group_name: str,
    metadata: Dict[str, Any],
) -> None:
    try:
        with pd.HDFStore(hdf5_path, "a") as store:
            store.put(group_name, df)
            group = store.get_storer(group_name).group
            for k, v in metadata.items():
                group._v_attrs[k] = v
    except Exception as e:
        print(f"Warning: Error saving DataFrame to HDF5: {str(e)}")

    


def read_plot_data(
    file_path: str, group_path: str
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    with pd.HDFStore(file_path, "r") as store:
             df = store[group_path]
             attrs = store.get_storer(group_path).group._v_attrs
             meta = {
             "legend_title": attrs.legend_title,
             "x_label": attrs.x_label,
             "x_unit": attrs.x_unit,
             "y_label": attrs.y_label,
             "y_unit": attrs.y_unit,
        }
    return df, meta



def plot_service_loss_vs_power(
    processed_data: pd.DataFrame, plot_format_data: Dict[str, str]
) -> Figure:
    fig, ax = plt.subplots()

    for label in processed_data.index:
        row = processed_data.loc[label]
        ax.errorbar(
            row["service_loss_mean"],
            row["power_mean"],
            xerr=row["service_loss_std"],
            yerr=row["power_std"],
            fmt="o",
            label=label,
        )

    ax.set_xlabel(f'{plot_format_data["x_label"]} [{plot_format_data["x_unit"]}]')
    ax.set_ylabel(f'{plot_format_data["y_label"]} [{plot_format_data["y_unit"]}]')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(title=plot_format_data["legend_title"])
    ax.grid(True)

    return fig
    


def publish_plot(
    fig: Figure, source_paths: Union[str, List[str]], destination_path: str
) -> None:
    fig = tagplot(
        fig,
        id_method="time",
        prefix="GdD_WS_2526_<3695111>_",
    )
    publish(fig, source_paths, destination_path)
    