
import os
import numpy as np
import pandas as pd

def process_radar_data(file_names):
  radar_dfs = {}
  radar_nps = {}
  for file_name in file_names:
      csv_path = os.path.join("radar", file_name)
      df = pd.read_csv(csv_path,
                      delimiter=r'[\t,]+', engine='python',
                      skiprows=range(1,129))
      df.columns = df.columns.str.strip() # make col no tab
      first_beacon = df.iloc[0]['Beacon']  # note the first beacon
      change_idx   = df.index[df['Beacon'] != first_beacon] # data that are not assined with the first beacon
      # extract radar data
      start_pos  = change_idx[0]
      df_final = df.loc[start_pos:,['Beacon', 'I_payload', 'Q_payload']].reset_index(drop=True)
      #print(df_final.head())
      radar_dfs[file_name] = df_final
      iq_values = df_final[['I_payload', 'Q_payload']].apply(pd.to_numeric, errors='coerce')
      iq_np = iq_values.to_numpy()
      radar_nps[file_name] = iq_np
  return radar_dfs, radar_nps

#----------------------------------------------------------
# load MoCap data
#----------------------------------------------------------
def process_mocap_data(file_names):
  mocap_dfs = {}
  mocap_nps = {}
  radar_positions = {}  # Dictionary to store radar positions
  STOP_READ = 'Radar2'
  COL_NAME = 'Frame'

  for file_name in file_names:
    csv_path = os.path.join("mocap", file_name)
    df = pd.read_csv(csv_path, low_memory=False, header=1)

    radar_pos = None # Initialize radar position for this file

    if STOP_READ in df[COL_NAME].values:
      index = df.index[df[COL_NAME] == STOP_READ][0]
      # Extract the row after STOP_READ for radar position
      if index + 2 < len(df):
          radar_row = df.iloc[index + 2]
          # Assuming the radar position is in the columns after 'Frame'
          # You might need to adjust the column indices based on your data
          radar_pos_values = radar_row.iloc[1:-1].apply(pd.to_numeric, errors='coerce')
          radar_pos = radar_pos_values.to_numpy()
          radar_positions[file_name] = radar_pos


      df = df.loc[:index-1] # Trim the dataframe before STOP_READ

    df.columns = df.columns.str.strip() # make col no tab
    mocap_dfs[file_name] = df
    mocap_values = df.iloc[:, 1:-1] #skip frist col (frames) and last col (NaN)
    mocap_values = mocap_values.apply(pd.to_numeric, errors='coerce')
    mocap_np = mocap_values.to_numpy()

    mocap_nps[file_name] = mocap_np

  return mocap_dfs, mocap_nps, radar_positions


if __name__ == "__main__":
    file_list = ["20250715_pen1.csv"]
    radar_dfs, radar_nps = process_radar_data(file_list)
    mocap_dfs, mocap_nps, radar_positions = process_mocap_data(file_list)
    print("\nShape of mocap_nps:")
    for key, value in mocap_nps.items():
        print(f"  Key: {key}, Shape: {value.shape}")
    print("\nShape of radar_nps:")
    for key, value in radar_nps.items():
        print(f"  Key: {key}, Shape: {value.shape}")
    print("\nShape of radar_positions:")
    for key, value in radar_positions.items():
        print(f"  Key: {key}, Shape: {value.shape}")
        
