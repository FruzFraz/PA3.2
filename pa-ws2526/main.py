
import numpy as np
import pandas as pd
from functions import functions as fn




def main(): 
    file_path = "C:\PA3.2\data_GdD_WiSe2526 (1).h5"
    controler = ("ARIMA", "DTW", "PID")
    topologies = ("Coupled", "Decentral", "Central")
    disruptions = ["BlockageConstant",
                    "BlochageCosine",
                    "PumpOutage","NoDisruption"]

    grupe_names = fn.generate_group_name(controler, topologies, disruptions)
    considerd_groups = [
        'ARIMA_Coupled_BlockageConstant',
        'ARIMA_Coupled_BlockageCosine', 
        'ARIMA_Decentral_BlockageCosine', 
        'ARIMA_Decentral_NoDisruption']
    processed_data = pd.DataFrame(
    columns=[
        "power_mean",
        "power_std",
        "service_loss_mean",
        "service_loss_std",
    ]
)
    
    
    metadata_dict = {
            "legend_title": "",
            "x_label": "Power (Wh)",
            "x_unit": "Wh",
            "y_label": "Service Loss (m³)",
            "y_unit": "m³",
    }
    
    for group in grupe_names:    
        if group not in considerd_groups:
                    continue
    pass
    

    setpoint = fn.read_metadata(file_path, group, "setpoint")

    group_service_loss = []
    group_power = []

    for run in range(1, 11):
        run_id = f"run_{run:02d}"
        base = f"{group}/{run_id}"

        start_time_index = fn.read_metadata(
            file_path, base, "analyse_start_time_index"
        )

        tank_pressure = fn.read_data(file_path, f"{base}/tank_1_pressure")
        power_1 = fn.read_data(file_path, f"{base}/pump_1_power")
        power_2 = fn.read_data(file_path, f"{base}/pump_2_power")
        time = fn.read_data(file_path, f"{base}/time")

    


    service_fill = fn.cap_service_data(tank_pressure, setpoint)
    if not fn.check_negative_values(power_1) or not fn.check_negative_values(power_2):
        print(f"Warning: Negative power values in {group} {run_id}")
    sf_int = fn.integral_with_time_step(
        service_fill[start_time_index:], time[start_time_index:]
    )

    target = np.full_like(service_fill[start_time_index:], setpoint)
    st_int = fn.integral_with_time_step(target, time[start_time_index:])

    service_loss = fn.calculate_service_loss(sf_int, st_int)

    power_int = (
    fn.integral_with_time_step(power_1, time)
    + fn.integral_with_time_step(power_2, time)
    )

    power_Wh = fn.convert_Ws_to_Wh(power_int)

    group_service_loss.append(service_loss)
    group_power.append(power_Wh)
    sl_mean, sl_std = fn.calculate_mean_and_std(group_service_loss)
    p_mean, p_std = fn.calculate_mean_and_std(group_power)
   
   fn.save_dataframe_in_hdf5_with_metadata(
        prcoessed_data,
        data_archve_path,
        Matadata_path,
   )
    processed_data.loc[group] = [p_mean, p_std, sl_mean, sl_std]
    fig = fn.plot_service_loss_vs_power(
        plot_data,
        plot_format_data,


    
if __name__ == "__main__":
    main()
