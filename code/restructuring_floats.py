import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


datadir = "../data/"
ds = xr.open_dataset(datadir + "floats.nc")

data_vars = []
for var in list(ds.data_vars)[3:]:
    data_var = []
    j = 0
    for i in np.where(np.isnan(ds[var]))[0]:
        data_var.append(ds[var].values[j:i])
        j = i + 1
    data_vars.append(data_var)

data_vars = list(map(list, zip(*data_vars)))


index1 = []
index2 = []
expid = []
typeid = []
for i in range(len(data_vars)):
    expid += list(np.zeros_like(data_vars[i][0]) + ds["expid"].values[i])
    typeid += list(np.zeros_like(data_vars[i][0]) + ds["typeid"].values[i])
    index1 += list(np.ones_like(data_vars[i][0], dtype=int) + i)
    index2 += [dt.datetime.fromordinal(int(j)) + dt.timedelta(days=j%1) - dt.timedelta(days = 366) for j in data_vars[i][0]]
index = [index1, index2]
index = pd.MultiIndex.from_arrays(index, names=["drifter", "time"])


data_vars = []
for var in list(ds.data_vars)[3:]:
    data_var = []
    j = 0
    for i in np.where(np.isnan(ds[var]))[0]:
        data_var += list(ds[var].values[j:i])
        j = i + 1
    data_vars.append(data_var)
data_vars = list(map(list, zip(*data_vars)))


df = pd.DataFrame(data_vars, columns=list(ds.data_vars)[3:], index=index)
df.drop(columns=["num"], inplace=True)
df["expid"] = expid
df["typeid"] = typeid
df.expid = df.expid.astype(int)
df.typeid = df.typeid.astype(int)
#df.to_pickle(datadir + "floats_new.pkl")
print(df)
ds = xr.Dataset.from_dataframe(df, sparse=True)
print(ds)
