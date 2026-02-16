
import numpy as np
import pandas as pd
from functions import functions as fn
import os



def main(): 
    # Determine the directory where this main.py file is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
     # Path to the input HDF5 file containing simulation data
    file_path = "./data_GdD_WiSe2526.h5"
    print(file_path)
     # Define all possible parameter combinations
    controler = ("ARIMA", "DTW", "PID")
    topologies = ("Coupled", "Decentral", "Central")
    disruptions = ["BlockageConstant",
                    "BlockageCosine",
                    "PumpOutage","NoDisruption"]

    # Generate all possible group names
    # Example: ARIMA_Coupled_BlockageConstant, etc.
    grupe_names = fn.generate_group_name(controler, topologies, disruptions)
    # Only these groups will actually be evaluated and plotted
    considerd_groups = [
        "ARIMA_Coupled_BlockageConstant",
        "ARIMA_Coupled_BlockageCosine", 
        "ARIMA_Decentral_BlockageCosine", 
        "ARIMA_Decentral_NoDisruption"]
     # DataFrame to store mean values and standard deviations
    processed_data = pd.DataFrame(
    columns=[
        "power_mean",
        "power_std",
        "service_loss_mean",
        "service_loss_std",
    ]
    )

    # Output file for processed plot data
    data_archive_path = "./data_GdD_plot_data_WiSe2526.h5"
    metadata_dict = {
            "legend_title": "Controller_Topology_Disruption",
            "x_label": "Service Loss",
            "x_unit": "%",
            "y_label": "Power",
            "y_unit": "Wh",
            }
    # Metadata used later for plot labeling
    for group in grupe_names: 
        # Iterate over all generated group names   
        if group not in considerd_groups:
            continue
        # Read the setpoint value from metadata
        setpoint = fn.read_metadata(file_path, group, "setpoint")
        # Lists to collect results from simulation runs
        group_service_loss = []
        group_power = []

        for run in range(1, 11):
           
            run_id = f"run_{run:02d}"
            base = f"{group}/{run_id}"
            
             # Read index where analysis should start
            # (e.g., ignore initial transient phase)
            start_time_index = fn.read_metadata(
                file_path, base, "analyse_start_time_index"
            )
            # Read required time series data
            tank_pressure = fn.read_data(file_path, f"{base}/tank_1_pressure")
            power_1 = fn.read_data(file_path, f"{base}/pump_1_power")
            power_2 = fn.read_data(file_path, f"{base}/pump_2_power")
            time = fn.read_data(file_path, f"{base}/time")

    

            # Limit service data to physically meaningful range [0, setpoint]
            service_fill = fn.cap_service_data(tank_pressure, setpoint)
            # Check for negative power values
            if not fn.check_negative_values(power_1) or not fn.check_negative_values(power_2):
                print(f"Warning: Negative power values in {group} {run_id}")
            # Integrate actual service from analysis start onward
            sf_int = fn.integral_with_time_step(
                service_fill[start_time_index:], time[start_time_index:]
            )
             # Create ideal target curve (constant at setpoint)
            target = np.full_like(service_fill[start_time_index:], setpoint)
            # Integrate ideal service
            st_int = fn.integral_with_time_step(target, time[start_time_index:])
            # Calculate relative service loss in %
            service_loss = fn.calculate_service_loss(sf_int, st_int)
            # Integrate total pump energy consumption
            power_int = (
            fn.integral_with_time_step(power_1, time)
            + fn.integral_with_time_step(power_2, time)
            )
            # Convert energy from Ws to Wh
            power_Wh = fn.convert_Ws_to_Wh(power_int)
            # Store run results
            group_service_loss.append(service_loss)
            group_power.append(power_Wh)
        # Compute mean and standard deviation over all 10 runs    
        sl_mean, sl_std = fn.calculate_mean_and_std(group_service_loss)
        p_mean, p_std = fn.calculate_mean_and_std(group_power)
        # Store aggregated results in DataFrame
        processed_data.loc[group] = [p_mean, p_std, sl_mean, sl_std]   
        # Print intermediate results for debugging
        print(processed_data)
     # Save processed data with metadata to HDF5
    fn.save_dataframe_in_hdf5_with_metadata(
            processed_data,
            data_archive_path,
            "plot_data",
            metadata_dict,
    )
    # Read plot data again (including metadata)
    plot_data, plot_format_data = fn.read_plot_data(
    data_archive_path, 
    "plot_data"
    )

    # Create plot (Service Loss vs Power) 
    fig = fn.plot_service_loss_vs_power(plot_data, plot_format_data)
     # Publish plot using plotid
    fn.publish_plot(
        fig,
        source_paths=[
            data_archive_path,  
            os.path.join(base_dir, "functions", "functions.py"),
            os.path.join(base_dir, "main.py")
        ],
        destination_path=os.path.join(base_dir, "plotid"),
    )  
# Standard Python pattern: execute main only if script is run directly      
if __name__ == "__main__":
    main()
