import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


datadir = "../data/"
ds = xr.open_dataset(datadir + "drifters.nc")
ds["id"] = ds.id.astype("uint8")
ds["buoy"] = ds.buoy.astype("uint8")
ds = ds.chunk(100000)
df = ds.to_dataframe()
print(df)




data_vars = []
for var in list(ds.data_vars)[2:]:
    data_var = []
    j = 0
    for i in np.where(np.isnan(ds[var]))[0]:
        data_var.append(ds[var].values[j:i])
        j = i + 1
    data_vars.append(data_var)

data_vars = list(map(list, zip(*data_vars)))

print("hello")

index1 = []
index2 = []
buoy = []
for i in range(len(data_vars)):
    buoy += list(np.zeros_like(data_vars[i][0], dtype="uint8") + ds["buoy"].values[i])
    index1 += list(np.ones_like(data_vars[i][0], dtype="uint8") + i)
    index2 += [dt.datetime.fromordinal(int(j)) + dt.timedelta(days=j%1) - dt.timedelta(days = 366) for j in data_vars[i][0]]
index = [index1, index2]
index = pd.MultiIndex.from_arrays(index, names=["drifter", "time"])

print("hello")
data_vars = []
for var in list(ds.data_vars)[2:]:
    data_var = []
    j = 0
    for i in np.where(np.isnan(ds[var]))[0]:
        data_var += list(ds[var].values[j:i])
        j = i + 1
    data_vars.append(data_var)
data_vars = list(map(list, zip(*data_vars)))

print("hello")

datadir = "../data/"
df = pd.DataFrame(data_vars, columns=list(ds.data_vars)[2:], index=index)
df.drop(columns=["num"], inplace=True)
df.filled = df.filled.astype(int)
df["buoy"] = buoy
df.expid = df.expid.astype(int)
df.typeid = df.typeid.astype(int)
df.to_pickle(datadir + "drifters_new.pkl")
print(df)
