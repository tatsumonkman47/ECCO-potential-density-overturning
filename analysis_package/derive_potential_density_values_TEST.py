import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os 

from xmitgcm import open_mdsdataset
import xmitgcm
import ecco_v4_py as ecco

from netCDF4 import Dataset

import seawater
from seawater.library import T90conv

from analysis_package import open_datasets




ATMOSPHERIC_PRESSURE = 101325. # pascals: N/ms^2
PA_TO_DBAR = 1e-4 # dbar/pascal
G_ACCELERATION = 9.81 # m/s^2
REFERENCE_DENSITY = 1029. # kg/m^3

# String identifiers for various ECCO dataset values
# RHOAnoma: insitu density anomaly
RHOAnoma_var_str = "RHOAnoma"
# PHIHYD: insitu pressure anomaly with respect to the depth integral of gravity and reference density (g*rho_reference)
PHIHYD_var_str = "PHIHYD"
# SALT: insitu salinity (psu)
SALT_var_str = "SALT"
# THETA: potential pressure (C)
THETA_var_str = "THETA"




def make_potential_density_dataset(PHIHYD_ds_raw, SALT_ds_raw, THETA_ds_raw, time_slice=slice(0,288),data_dir="./old_nctiles_monthly/",save=False,ref_pressure=2000.):
	"""
	function to derive potential density 

	Variables
	---------
	time_slice: numpy array 
	data_dir: str, location of reanalysis datasets 
	save: bool, True values 
	ref_pressure: float, reference pressure with which to calculate potential density with respect to

	Returns
	-------
	PDENS_ds: xarray DataSet, DataSet with dimensions (tile,time,k,i,j) containing potential density values at 
		grid center points 
	fname: string, filename underwhich PDENS_ds is saved (file format: netCDF4)

	"""


	#RHOAnoma_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, RHOAnoma_var_str, time_slice)
	#PHIHYD_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, PHIHYD_var_str, time_slice)
	#SALT_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, SALT_var_str, time_slice)
	#THETA_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, THETA_var_str, time_slice)

	#RHOAnoma_ds_raw = xr.open_dataset("RHOAnoma_ds_raw.nc")
	#PHIHYD_ds_raw = xr.open_dataset("PHIHYD_ds_raw.nc")
	#SALT_ds_raw = xr.open_dataset("SALT_ds_raw.nc")
	#THETA_ds_raw = xr.open_dataset("THETA_ds_raw.nc")

	# resize data arrays to get rid of nan padding on the final index
	#RHOAnoma_ds = RHOAnoma_ds_raw.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	#RHOAnoma_ds.load()	
	PHIHYD_ds = PHIHYD_ds_raw#.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	PHIHYD_ds.load()
	SALT_ds = SALT_ds_raw#.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	SALT_ds.load()
	THETA_ds = THETA_ds_raw#.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	THETA_ds.load()


	# create arrays for intermediate variables
	P_INSITU_ds = PHIHYD_ds.copy(deep=True)*0
	P_INSITU_ds.load()
	P_INSITU_ds = P_INSITU_ds.rename(name_dict={"PHIHYD":"P_INSITU"})

	
	P_INSITU_ds["P_INSITU"] = (PHIHYD_ds["PHIHYD"]*(REFERENCE_DENSITY) + G_ACCELERATION*REFERENCE_DENSITY*PHIHYD_ds["dep"])*PA_TO_DBAR

	# print values to check pressure
	#print("Max insitu pressure: ", P_INSITU_ds["P_INSITU"].max(), "decibars")
	#print("Min insitu pressure: ", P_INSITU_ds["P_INSITU"].min(), "decibars")
	#print("Mean insitu pressure: ", P_INSITU_ds["P_INSITU"].mean(), "decibars")

	TEMP_INSITU_ds = PHIHYD_ds.copy(deep=True)
	TEMP_INSITU_ds.load()
	TEMP_INSITU_ds = TEMP_INSITU_ds.rename(name_dict={"PHIHYD":"TEMP_INSITU"})

	PDENS_ds = PHIHYD_ds.copy(deep=True)
	PDENS_ds.load()
	PDENS_ds = PDENS_ds.rename(name_dict={"PHIHYD":"PDENS"})

	# derive insitu temperature at the insitu pressure level
	insitu_temperature = seawater.eos80.temp(SALT_ds.SALT,
                                			 T90conv(THETA_ds.THETA),
                                			 P_INSITU_ds.P_INSITU)

	if "time" and "tile" in PHIHYD_ds.dims:
		TEMP_INSITU_ds["TEMP_INSITU"] = (["tile","time","k","j","i"], insitu_temperature)
	elif "tile" in PHIHYD_ds.dims:
		TEMP_INSITU_ds["TEMP_INSITU"] = (["tile","k","j","i"], insitu_temperature)
	elif "time" in PHIHYD_ds.dims:
		TEMP_INSITU_ds["TEMP_INSITU"] = (["time","k","j","i"], insitu_temperature)
	else:
		TEMP_INSITU_ds["TEMP_INSITU"] = (["k","j","i"], insitu_temperature)
	#print("TEMP_INSITU_ds.max(): ", TEMP_INSITU_ds.max(), "degrees C")
	#print("TEMP_INSITU_ds.min(): ", TEMP_INSITU_ds.min(), "degrees C")
	# derive potential density
	insitu_potential_density = seawater.eos80.pden(SALT_ds.SALT,
	                                				TEMP_INSITU_ds.TEMP_INSITU,
	                                				P_INSITU_ds["P_INSITU"],
	                                				pr=ref_pressure)
	# should probably change some of the units and attributes and stuff..
	# later...
	#print("insitu_potential_density.max(): ", insitu_potential_density.max(), " decibars")
	#print("insitu_potential_density.min(): ", insitu_potential_density.min(), " decibars")
	if "time" and "tile" in PHIHYD_ds.dims:
		PDENS_ds["PDENS"] = (["tile","time","k","j","i"], insitu_potential_density)
	elif "tile" in PHIHYD_ds.dims:
		PDENS_ds["PDENS"] = (["tile","k","j","i"], insitu_potential_density)
	elif "time" in PHIHYD_ds.dims:
		PDENS_ds["PDENS"] = (["time","k","j","i"], insitu_potential_density)
	else:
		PDENS_ds["PDENS"] = (["k","j","i"], insitu_potential_density)

	fname = None
	if save == True:
		fname = "Potential_Density_ts" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + ".nc"
		i = 0
		while os.path.isfile(fname) == True:
			i += 1
			fname = "Potential_Density_ts" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + "_copy" + str(i) + ".nc"
		PDENS_ds.to_netcdf(fname)
		print("Saved potential density data for timesteps " + str(time_slice[0]) 
			  + " to " + str(time_slice[-1]) + " to file " + str(fname))

	return PDENS_ds, P_INSITU_ds

