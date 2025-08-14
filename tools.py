import numpy as np
def compute_mean_position(radar_pos):

  radar_pos_reshaped = radar_pos.reshape(-1, 3)
  mean_xyz = np.mean(radar_pos_reshaped, axis=0)
  return mean_xyz

