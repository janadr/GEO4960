from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import gsw


def plotquivers(lon, lat, u, v, scale=10, latspacing=10, lonspacing=20, rotate=True, **kwargs):
    map = Basemap(**kwargs)
    map.drawcoastlines()
    map.etopo()
    map.fillcontinents(color="white", lake_color="white")
    if rotate:
        u, v, lon, lat = map.rotate_vector(u, v, lon, lat, returnxy=True)

    map.quiver(lon, lat, u, v,
            color="k",
            scale=scale
            )
    map.drawparallels(np.arange(-90, 90, latspacing), labels=[0, 1, 1, 0])
    map.drawmeridians(np.arange(-180, 180, lonspacing), labels=[1, 0, 0, 1])

    return map


plt.rc("text", usetex=True)
plt.rc("font", family="serif")


datadir = "../data/"
figdir = "../figurar/"

zetafile = datadir + "mdt-cnes-cls18.nc"       # 1/8 deg res
temperaturefile = datadir + "woa18_decav_t00_01.nc"   # 1 deg res
salinityfile = datadir + "woa18_decav_s00_01.nc"      # 1 deg res
zeta_ds = xr.open_dataset(zetafile)
res = 8
u = zeta_ds.u[0, ::res, ::res].to_masked_array()
v = zeta_ds.v[0, ::res, ::res].to_masked_array()
lon = zeta_ds.longitude.values[::res]
lat = zeta_ds.latitude.values[::res]
LON, LAT = np.meshgrid(lon, lat)



temperature_ds = xr.open_dataset(temperaturefile, decode_times=False)
lon = temperature_ds.lon.values
lat = temperature_ds.lat.values
depth = temperature_ds.depth.values
temperature = temperature_ds.t_an[0].to_masked_array()

salinity_ds = xr.open_dataset(salinityfile, decode_times=False)
salinity = salinity_ds.s_an[0].to_masked_array()

DEPTH, LAT, LON = np.meshgrid(-depth, lat, lon, indexing="ij")

pressure = gsw.conversions.p_from_z(DEPTH, LAT)
density = gsw.density.rho_t_exact(salinity, temperature, pressure)

g = 9.81
rho0 = 1027
f = 2*2*np.pi/86400*np.sin(LAT*np.pi/180)
grad = np.gradient(density, lat, lon, axis=[1, 2], edge_order=1)
du_dz = g/(rho0*f)*grad[0]/(4e7/360)
dv_dz = -g/(rho0*f)*grad[1]/(4e7/360*np.cos(LAT*np.pi/180))

dz = -np.diff(DEPTH, axis=0)
D = depth[-1] - depth[0]
U = np.sum(du_dz[:-1]*dz, axis=0)
V = np.sum(dv_dz[:-1]*dz, axis=0)

ubottom = u - U
vbottom = v - V
ucrossUnorm = u*vbottom - ubottom*v
udotU = u*ubottom + v*vbottom
theta = np.arctan(ucrossUnorm/udotU)*180/np.pi

with np.errstate(divide='ignore',invalid='ignore'):
    theta = np.arccos(udotU/(np.sqrt(u**2 + v**2)*np.sqrt(ubottom**2 + vbottom**2)))*np.sign(ucrossUnorm)*180/np.pi

fig, ax = plt.subplots(1, 2, figsize=(10, 8))
map = plotquivers(lon, lat, u, v, ax=ax[0], latspacing=10, lonspacing=20, scale=5,
                    width=5e6, height=5e6, projection="gnom",
                    lat_0=0.0, lon_0=80.0, resolution="i"
                    )
map = plotquivers(lon, lat, ubottom, vbottom, ax=ax[1], latspacing=10, lonspacing=20,
                    scale=5, width=5e6, height=5e6, projection="gnom",
                    lat_0=0.0, lon_0=80.0, resolution="i"
                    )
fig.tight_layout()
fig.savefig(figdir + "oppg2.4_2.pdf")


fig, ax = plt.subplots(1, 2, figsize=(10, 8))
map = plotquivers(lon, lat, u, v, ax=ax[0], latspacing=10, lonspacing=20, scale=2,
                    width=1.5e6, height=1.5e6, projection="gnom",
                    lat_0=-60.0, lon_0=80.0, resolution="i")
map = plotquivers(lon, lat, ubottom, vbottom, ax=ax[1], latspacing=10, lonspacing=20,
                    scale=2, width=1.5e6, height=1.5e6, projection="gnom",
                    lat_0=-60.0, lon_0=80.0, resolution="i"
                    )
fig.tight_layout()
fig.savefig(figdir + "oppg2.4_1.pdf")


fig, ax = plt.subplots(figsize=(10, 8))
map = plotquivers(lon, lat, U, V, ax=ax, latspacing=20, lonspacing=40, rotate=False,
                    projection="cyl", resolution="i", scale=None
                    )
fig.tight_layout()
fig.savefig(figdir + "oppg2.4_4.pdf")


fig, ax = plt.subplots(figsize=(10, 8))
map = Basemap(ax=ax)
map.drawcoastlines()
#map.etopo()
map.fillcontinents(color="white", lake_color="white")
map.drawparallels(np.arange(-90, 90, 20), labels=[0, 1, 1, 0])
map.drawmeridians(np.arange(-180, 180, 40), labels=[1, 0, 0, 1])
c = ax.contourf(LON[0], LAT[0], theta, levels=20, cmap="twilight")
cbar = fig.colorbar(c, shrink=0.5)
cbar.set_label(r"Degrees [$^\circ$]", fontsize=14)
fig.tight_layout()
fig.savefig(figdir + "oppg3.3_1.pdf")

plt.show()
