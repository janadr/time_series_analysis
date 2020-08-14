import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd

datadir = "../data/"
#df = pd.read_pickle(datadir + "floats_new.pkl")
#df.to_csv(datadir + "floats.csv")
"""
reader = pd.read_csv(datadir + "floats.csv",
                index_col=[0, 1],
                infer_datetime_format=True,
                chunksize=10000
                )
datadir = "../data/floats/"

count = 1
for chunk in reader:
    chunk.to_xarray().to_netcdf(datadir + f"chunk{count}.nc")
    count += 1
"""
datadir = "../data/floats/"
ds = xr.open_mfdataset(datadir + "*.nc",
                        #combine="nested",
                        concat_dim="time"
                        )
#ds["lat"].sel(drifter=1).plot()

print(ds)
