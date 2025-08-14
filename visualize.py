import os
from typing import Dict, Tuple

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.interpolate import interp1d

from loader import *
from tools import *

N_MARKERS = 5
FS_MOCAP  = 250.0
FS_RADAR  = 256.0
RF        = 5.8e9         # Hz
NPERSEG   = 256
NOVERLAP  = 224
SMO_POLY  = 3
CLIM      = None            
XLIM_SEC  = None            

def compute_mean_radar_pos(radar_positions):
    
    radar_pos_reshaped = radar_pos.reshape(-1, 3)
    mean_xyz = np.mean(radar_pos_reshaped, axis=0)
    return mean_xyz

def reshape_mocap(mocap_dict, N_MARKERS):
    print(mocap_dict)
    return 0



if __name__ == "__main__":
    file_list = ["20250715_pen1.csv"]
    _, radar_nps = process_radar_data(file_list)
    _, mocap_nps, radar_positions = process_mocap_data(file_list)

    reshape_mocap(mocap_nps, N_MARKERS)

