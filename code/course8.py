from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from scipy import stats


# creating a dataframe using xarray
datadir = "../data/"
floats = xr.open_dataset(datadir + "floats.nc")
print(floats.head())

# converts to numpy arrays
lon = floats.lon.values
lat = floats.lat.values

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
print(np.nanmin(lon))
map = Basemap(  ax=ax,
                llcrnrlon=np.nanmin(lon)+1, urcrnrlon=np.nanmax(lon)-1,
                llcrnrlat=np.nanmin(lat)+1, urcrnrlat=np.nanmax(lat)-1,
                projection="cyl",
                fix_aspect=False,
                resolution='l'
                )
map.fillcontinents(color="black", lake_color="black")
map.drawparallels(np.arange(-90, 90, 20), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(-180, 180, 20), labels=[1, 0, 0, 1])

LON, LAT = map.shiftdata(lon, lat)
map.plot(LON, LAT, latlon=True)


u = floats.u.to_masked_array()
v = floats.v.to_masked_array()
U = np.sqrt(u**2 + v**2)

xbins = np.arange(-80, 0, 0.5)
ybins = np.arange(15, 65, 0.5)

H = stats.binned_statistic_2d(lon, lat, None, bins=[xbins, ybins], statistic="count")
M = stats.binned_statistic_2d(lon, lat, U, bins=[xbins, ybins], statistic="mean")
std = stats.binned_statistic_2d(lon, lat, U, bins=[xbins, ybins], statistic="std")
xcenter = 0.5*(H.x_edge[1:] + H.x_edge[:-1])
ycenter = 0.5*(H.y_edge[1:] + H.y_edge[:-1])
X, Y = np.meshgrid(xcenter, ycenter, indexing="ij")

# creating a figure and ax object and setting the aspect ratio to equal
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_aspect("equal")

# creating a Basemap object which is used to draw continents
map = Basemap(  ax=ax,
                llcrnrlon=xbins[0], urcrnrlon=xbins[-1],
                llcrnrlat=ybins[0], urcrnrlat=ybins[-1],
                projection="cyl",
                fix_aspect=False,
                resolution='l'
                )

# Sets all values of 0 to nan as log10(0) = -inf
# Ideally I could use a mask here
H.statistic[H.statistic == 0] = np.nan
im = map.pcolormesh(X, Y, np.log10(H.statistic),
                    vmin=0.5,
                    vmax=2.5,
                    cmap="jet",
                    shading="flat",
                    )
map.fillcontinents(color="black", lake_color="black")
map.drawparallels(np.arange(-90, 90, 20), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(-180, 180, 20), labels=[1, 0, 0, 1])

# creating a colorbar attached to the ax
fig.colorbar(im, ax=ax)
# removing whitespace in the figure
fig.tight_layout()
fig.savefig("test.pdf")


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_aspect("equal")

map = Basemap(  ax=ax,
                llcrnrlon=xbins[0], urcrnrlon=xbins[-1],
                llcrnrlat=ybins[0], urcrnrlat=ybins[-1],
                projection="cyl",
                fix_aspect=False,
                resolution='l'
                )

# Sets all values of 0 to nan as log10(0) = -inf
# Ideally I could use a mask here
im = map.pcolormesh(X, Y, M.statistic,
                vmin=0,
                vmax=60,
                cmap="jet",
                shading="flat",
                )
map.fillcontinents(color="black", lake_color="black")
map.drawparallels(np.arange(-90, 90, 20), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(-180, 180, 20), labels=[1, 0, 0, 1])

# creating a colorbar attached to the ax
fig.colorbar(im, ax=ax)
# removing whitespace in the figure
fig.tight_layout()
fig.savefig("test0.pdf")


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_aspect("equal")

map = Basemap(  ax=ax,
                llcrnrlon=xbins[0], urcrnrlon=xbins[-1],
                llcrnrlat=ybins[0], urcrnrlat=ybins[-1],
                projection="cyl",
                fix_aspect=False,
                resolution='l'
                )

# Sets all values of 0 to nan as log10(0) = -inf
# Ideally I could use a mask here
std.statistic[std.statistic == 0] = np.nan
im = map.pcolormesh(X, Y, std.statistic,
                vmin=0,
                vmax=60,
                cmap="jet",
                shading="flat",
                )
map.fillcontinents(color="black", lake_color="black")
map.drawparallels(np.arange(-90, 90, 20), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(-180, 180, 20), labels=[1, 0, 0, 1])

# creating a colorbar attached to the ax
fig.colorbar(im, ax=ax)
# removing whitespace in the figure
fig.tight_layout()
fig.savefig("test1.pdf")
plt.show()
