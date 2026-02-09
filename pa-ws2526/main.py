from email.mime import base
from time import time
from tokenize import group
from typing import Union
import numpy as np
import pandas as pd
import os
from functions import functions as fn

from functions import functions as fn
file_path =os.path.join("data_GdD_WiSe2526", "PA_WS2526_data.csv")
controler = ("ARIMA", "DTWk", "PID")
topologies = ("Coupled", "Decentral", "Central")
disruptions = ("BlockageConstant", "BlochageCosine", "PumpOutage","NoDisruption")


def main(): 
    grupe_names = fn.generate_group_name(controler, topologies, disruptions)
    considerd_groups = fn.get_considered_groups('ARIMA_Coupled_BlockageConstant', 'ARIMA_Coupled_BlockageCosine', 'ARIMA_Decentral_BlockageCosine', 'ARIMA_Decentral_NoDisruption')
    processed_data = fn.dataframe(
        power_mean = [],
        power_std =[],
        servise_loss_mean = [],
        servise_loss_std = [],
        )
    
    pass
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
            file_path, base, "analysis_start_time_index"
        )

        tank_pressure = fn.read_data(file_path, f"{base}/tank_1_pressure")
        power_1 = fn.read_data(file_path, f"{base}/pump_1_power")
        power_2 = fn.read_data(file_path, f"{base}/pump_2_power")
        time = fn.read_data(file_path, f"{base}/time")

    pass


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
pass
    
if __name__ == "__main__":
    main()
