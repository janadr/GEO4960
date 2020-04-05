import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


plt.rc("text", usetex=True)
plt.rc("font", family="serif")

datadir = "../data/"
vars2D = "hw1_p1_modelfields_2dvariables.nc"
midbasinTemps = "hw1_p1_modelfields_temp_midbasin.nc"
midbasinV = "hw1_p1_modelfields_v_midbasin.nc"


ds = xr.open_dataset(datadir + vars2D)

param = "zeta"

start_time = "1970-01-01"
end_time = "1973-03-01"
x_start = 0
x_end = 200
y0 = 100

time_slice = slice(start_time, end_time)
x_slice = slice(x_start, x_end)

data = ds[param].sel(ocean_time=time_slice)

coordnames = list(data.coords.keys())

time = data.ocean_time.values
x = data[coordnames[1]].values[0, x_slice]

X, Y = np.meshgrid(x, time)

figdir = "../figurar/"

fig, ax = plt.subplots()

c = ax.contourf(X*1e-3, Y, data.values[:, y0, x_slice]*1e2,
                levels=25,
                cmap="PRGn"
                )
ax.contour(X*1e-3, Y, data.values[:, y0, x_slice]*1e2,
                levels=25,
                colors="black")
ax.invert_yaxis()
plt.setp(ax.get_xticklabels(), ha="right", rotation=30)
cbar = fig.colorbar(c)
cbar.set_label("Sea surface height [cm]", fontsize=14)
ax.set_xlabel("Longitude [km]", fontsize=14)
ax.set_ylabel("Time", fontsize=14)
fig.tight_layout()
fig.savefig(figdir + "oppg1_1.pdf")

plt.show()
