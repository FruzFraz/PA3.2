from email.mime import base
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
    
pass


if __name__ == "__main__":
    main()
