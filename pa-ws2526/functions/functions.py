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
    """Generate group names from cartesian product of parameters.
    
    Creates all possible combinations of controller, topology, and disruption
    parameters in the format: <Controller>_<Topology>_<Disruption>
    
    Args:
        controller: Control strategy (e.g., "ARIMA", "PID", "DTW") or list of strategies
        topology: Network topology (e.g., "Coupled", "Decentral", "Central") or list
        disruption: Disruption type (e.g., "BlockageConstant", "PumpOutage") or list
    
    Returns:
        List of group name strings in format "Controller_Topology_Disruption"
    
    Example:
        >>> generate_group_name(["ARIMA", "PID"], "Coupled", "NoDisruption")
        ["ARIMA_Coupled_NoDisruption", "PID_Coupled_NoDisruption"]
    """
    if isinstance(controller, str):
        controller = [controller]
    if isinstance(topology, str):
        topology = [topology]
    if isinstance(disruption, str):
        disruption = [disruption]

    return[f"{c}_{t}_{d}" for c in controller for t in topology for d in disruption]    



def read_metadata(file: str, path: str, attr_key: str) -> Any:
    """Read a metadata attribute from an HDF5 file.
    
    Opens an HDF5 file and retrieves a specific attribute from a group or dataset.
    Returns None and prints a warning if the path or attribute does not exist.
    
    Args:
        file: Path to the HDF5 file
        path: Path to the group or dataset within the HDF5 file
        attr_key: Name of the metadata attribute to retrieve
    
    Returns:
        The attribute value, or None if path/attribute doesn't exist or on error
    
    Example:
        >>> setpoint = read_metadata("data.h5", "ARIMA_Coupled_NoDisruption", "setpoint")
    """
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
    """Read data from HDF5 dataset and return as 1D numpy array.
    
    Reads a dataset from an HDF5 file and ensures it is returned as at least
    a 1D array. Returns None and prints a warning if the path doesn't exist
    or points to a group instead of a dataset.
    
    Args:
        file: Path to the HDF5 file
        path: Path to the dataset within the HDF5 file
    
    Returns:
        1D numpy array containing the data, or None on error
    
    Example:
        >>> pressure = read_data("data.h5", "group/run_01/tank_1_pressure")
"""
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
    """Limit service data to the range [0, setpoint].
    
    Clips all values in the service data array to be within the valid range
    from 0 to the setpoint value. Values below 0 are set to 0, values above
    setpoint are set to setpoint.
    
    Args:
        service_data: Array of service values (e.g., tank pressure)
        setpoint: Maximum allowed value (target setpoint)
    
    Returns:
        Array with values clipped to [0, setpoint]
    
    Example:
        >>> capped = cap_service_data(np.array([-1, 10, 25, 15]), 20.0)
        >>> # Returns: array([0, 10, 20, 15])
    """
    return np.clip(service_data, 0.0, setpoint)


def check_negative_values(array: NDArray) -> bool:
    """Check if all values in array are non-negative.
    
    Validates that all elements in the array are greater than or equal to zero.
    Used for data validation to detect physically impossible negative values
    in measurements like power or pressure.
    
    Args:
        array: NumPy array to check
    
    Returns:
        True if all values >= 0, False if any negative value exists
    
    Example:
        >>> check_negative_values(np.array([0, 1, 2]))
        True
        >>> check_negative_values(np.array([0, -1, 2]))
        False
    """   
    return np.all(array >= 0)




def integral_with_time_step(data: NDArray, time_steps: NDArray) -> Optional[float]:
    """Calculate integral over non-uniform time steps using trapezoidal rule.
    
    Computes the definite integral of a time series using the trapezoidal rule,
    which approximates the area under the curve by summing trapezoid areas
    between consecutive points.
    
    Formula: Integral ≈ Σ [(y[i] + y[i+1])/2 * (t[i+1] - t[i])]
    
    Args:
        data: Array of measurement values
        time_steps: Array of corresponding time points
    
    Returns:
        The computed integral value, or None if inputs are invalid
    
    Raises:
        Returns None and prints warning if:
            - Either input is None
            - Arrays have different lengths
    
    Example:
        >>> y = np.array([0, 1, 2])
        >>> t = np.array([0, 1, 3])
        >>> integral_with_time_step(y, t)
        3.0  # Area: (0+1)/2*1 + (1+2)/2*2 = 0.5 + 3.0
    """
    if data is None or time_steps is None:
        print("Warning: Data or time_steps is None.")
        return None

    if len(data) != len(time_steps):
        print("Warning: Length mismatch.")
        return None
    integral = 0.0
    for i in range(len(data) - 1):
        integral += (data[i] + data[i + 1]) / 2 * (time_steps[i + 1] - time_steps[i])
    return float(integral)



def calculate_service_loss(service_fill: float, service_target: float) -> float:
    """Calculate percentage service loss.
    
    Computes how much the actual service delivery falls short of the target
    service level, expressed as a percentage.
    
    Formula: SERVICE_LOSS (%) = 100 * (1 - service_fill / service_target)
    
    Args:
        service_fill: Actual service delivered (integral of actual performance)
        service_target: Ideal service target (integral of target performance)
    
    Returns:
        Service loss percentage (0-100), or 0.0 on error
    
    Example:
        >>> calculate_service_loss(80.0, 100.0)
        20.0  # 20% service loss
        """
    if service_fill is None or service_target is None:
        print("Warning: Service values are None.")
        return 0.0

    if service_target == 0:
        return 0.0
    return 100.0 * (1.0 - service_fill / service_target)



def convert_Ws_to_Wh(energy_in_Ws: float) -> float:
    """Convert energy from Watt-seconds (Ws) to Watt-hours (Wh).
    
    Converts energy units using the relationship: 1 Wh = 3600 Ws
    
    Args:
        energy_in_Ws: Energy in Watt-seconds
    
    Returns:
        Energy in Watt-hours, or 0.0 if input is None
    
    Example:
        >>> convert_Ws_to_Wh(7200.0)
        2.0  # 7200 Ws = 2 Wh
    """
    if energy_in_Ws is None:  # >>> ADDED ChatGPT
        print("Warning: Energy value is None.")
        return 0.0
    return energy_in_Ws / 3600.0
    


def calculate_mean_and_std(
    data: List[float]
) -> Tuple[float, float]:
    
    if len(data) == 0:  # >>> ADDED CahtGPT
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
    """Save a DataFrame to HDF5 file with additional metadata attributes.
    
    Stores a pandas DataFrame in an HDF5 file and attaches metadata as
    attributes to the group. Opens file in append mode to preserve existing data.
    
    Args:
        df: DataFrame to save
        hdf5_path: Path to the HDF5 file
        group_name: Name of the group within the HDF5 file
        metadata: Dictionary of metadata to attach as attributes
    
    Example:
        >>> df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> meta = {'x_label': 'Time', 'x_unit': 's'}
        >>> save_dataframe_in_hdf5_with_metadata(df, 'data.h5', 'results', meta)
    """
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
    """Read plot data and associated metadata from HDF5 file.
    
    Loads a DataFrame and its metadata attributes from an HDF5 file,
    extracting plot formatting information like labels and units.
    
    Args:
        file_path: Path to the HDF5 file
        group_path: Path to the group within the HDF5 file
    
    Returns:
        Tuple of (DataFrame with data, Dictionary with plot metadata)
        
    Example:
        >>> plot_data, plot_meta = read_plot_data('archive.h5', 'plot_data')
        >>> print(plot_meta['x_label'])
        'Service Loss'
    """
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
    """Create scatter plot of service loss vs. power with error bars.
    
    Generates a publication-quality plot showing the relationship between
    service loss and power consumption for different system configurations.
    Each data point includes error bars for both dimensions.
    
    Args:
        processed_data: DataFrame with columns:
            - service_loss_mean, service_loss_std
            - power_mean, power_std
            Index contains configuration labels
        plot_format_data: Dictionary with plot formatting:
            - legend_title: Title for the legend
            - x_label, x_unit: X-axis label and unit
            - y_label, y_unit: Y-axis label and unit
    
    Returns:
        Matplotlib Figure object containing the plot
    
    Example:
        >>> fig = plot_service_loss_vs_power(df, format_dict)
        >>> fig.savefig('results.png')
    """
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
    """Publish a plot with PlotID tagging and source file references.
    
    Tags the plot with a unique identifier and publishes it along with all
    source files (code, data) for reproducibility. Uses the plotid package
    to ensure traceability.
    
    Args:
        fig: Matplotlib Figure to publish
        source_paths: Path(s) to source files (code, data) - single string or list
        destination_path: Target directory for plot output
    
    Note:
        - Adds timestamp-based unique ID to the plot
        - Links plot to all source files for reproducibility
        - Prefix should include course code and student ID
    
    Example:
        >>> publish_plot(fig, ['data.h5', 'script.py'], './output')
    """
    #source_paths = [source_paths]
    fig = tagplot(
        fig,
        id_method="time",
        prefix="GdD_WS_2526_3695111_",
        engine= "matplotlib"
    )
    if isinstance(source_paths, str):
       source_paths = [source_paths]
    publish(fig, source_paths, destination_path)
    