import xarray as xr 
import numpy as np
import os

data_dir = "./old_nctiles_monthly/"
uvvel_data_dir = "./nctiles_monthly/"

UVELMASS_var = "UVELMASS"
VVELMASS_var = "VVELMASS"
GM_PSIX_var = "GM_PsiX"
GM_PSIY_var = "GM_PsiY"


def open_combine_raw_ECCO_tile_files(path,VAR,time_slice,grid_path="./ecco_grid/ECCOv4r3_grid.nc",rename_indices=True):
	""" Open and combine individual tile files for an xmitgcm dataset and return a complete dataset 
	will probably remove grid assignment to make this function more general..
	I am not adding in a grid for now since merging datasets is pretty computationally intensive.

	Parameters
	----------
	path: string
		path to datafiles
	VAR: string 
		name of variable to extract, eg 'UVELMASS'
	grid_path: string
		path to grid file, 

	Returns
	_______
	variable_all_tiles: xarray dataset
		Xarray datset with tile files stacked along 'tile' dimension
	"""
	variable_dir = path+VAR+"/"
	variable_dict = {}
	variable_nc_dict = {}

	for i in range(1,14):
	    if i < 10:
	        variable_dict["tile_"+str(i)] = xr.open_dataset(variable_dir+VAR+".000"+str(i)+".nc").load()
	    else:
	        variable_dict["tile_"+str(i)] = xr.open_dataset(variable_dir+VAR+".00"+str(i)+".nc").load()

	# rename dimension indicies to match grid dims..
	# will need to change dimension index names if you load variables that aren't in the middle 
	# of each grid tile, eg UVELMASS is on the western face of each grid tile wherease 
	if VAR == "UVELMASS" and rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j","i4":"i_g"})).isel(time=time_slice)
	elif VAR == "VVELMASS" and rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j_g","i4":"i"})).isel(time=time_slice)
	elif rename_indices==True:
		for tile in variable_dict:
		    variable_dict[tile] = (variable_dict[tile].rename({"i1":"time", "i2":"k","i3":"j","i4":"i"})).isel(time=time_slice)
	else:
		for tile in variable_dict:
		    variable_dict[tile] = variable_dict[tile].isel(time=time_slice)
	# combine tiles along new dimension "tile", which is the last argument in this function 
	variable_all_tiles = xr.concat([variable_dict["tile_1"],variable_dict["tile_2"],variable_dict["tile_3"],
	                               variable_dict["tile_4"],variable_dict["tile_5"],variable_dict["tile_6"],
	                               variable_dict["tile_7"],variable_dict["tile_8"],variable_dict["tile_9"],
	                               variable_dict["tile_10"],variable_dict["tile_11"],variable_dict["tile_12"],
	                             variable_dict["tile_13"]],'tile')

	# assign tile coordinates..
	tile_coords = np.arange(0,13)

	#print(variable_all_tiles)

	variable_all_tiles.assign_coords(tile = tile_coords)

	# not merging with grid for now...
	print("Loaded " + VAR + " over time slice  \n")
	# turn into dask array for performance purposes!

	#dask_variable_all_tiles = variable_all_tiles.chunk((12, len(time_slice), 50, 90, 90))
	return variable_all_tiles


def merge_uvel_vvel_into_dataset(UVELMASS_ds_raw, VVELMASS_ds_raw, time_slice,save=False,include_grid=True):
	""" Merge UVELMASS and VVELMASS into single dataset along with an xmitgcm grid object. 
	This assumes that we are using the raw ECCO data files and as such there are some extra steps that
	verify that things like dimensions and coordinates are set up correctly


	Parameters
	----------
	path: string
		path to datafiles
	VAR: string 
		name of variable to extract, eg 'UVELMASS'
	grid_path: string
		path to grid file, 

	Returns
	_______
	variable_all_tiles: xarray dataset
		Xarray datset with tile files stacked along 'tile' dimension
	fname: string
		name of saved file
	"""
	# move the load grid command outside of this function for final implementation
	grid_path = "./ecco_grid/ECCOv4r3_grid.nc"
	grid = xr.open_dataset(grid_path)

	#UVELMASS_ds_raw = open_combine_raw_ECCO_tile_files(uvvel_data_dir, 
	#												   UVELMASS_var, 
	#												   time_slice)
	#VVELMASS_ds_raw = open_combine_raw_ECCO_tile_files(uvvel_data_dir, 
	#													VVELMASS_var, 
	#													time_slice)

	# shift coordinate system to align with mitgcm grid indexing 
	# (when I load the raw files xarray starts indexing at 1 for i,i_g,j,j_g,k, not sure why but this won't affect
	# correctly loaded files anyway)

	# trim datasets if final nan padding value is present.. otherwise this won't change anything
	UVELMASS_ds = UVELMASS_ds_raw.isel(i_g=slice(0,90),j=slice(0,90),k=slice(0,50)).drop("lon").drop("lat")
	VVELMASS_ds = VVELMASS_ds_raw.isel(i=slice(0,90),j_g=slice(0,90),k=slice(0,50)).drop("lon").drop("lat")
	UVELMASS_ds.load()
	VVELMASS_ds.load()

	# trim extra stuff that we don't care about to speed up computation.
	UVELMASS_ds = UVELMASS_ds.drop("timstep").drop("land").drop("area").drop("thic")
	VVELMASS_ds = VVELMASS_ds.drop("timstep").drop("land").drop("area").drop("thic")

	# convert to dask array for computation speed.	
	UVELMASS_ds["UVELMASS"] = UVELMASS_ds.UVELMASS.chunk(UVELMASS_ds.UVELMASS.size)
	VVELMASS_ds["VVELMASS"] = VVELMASS_ds.VVELMASS.chunk(VVELMASS_ds.VVELMASS.size)

	# I'm including trivial masks since these are needed for the ecco_v4_py package functions..
	uvel_copy = UVELMASS_ds.copy(deep=True)
	maskW_ds = uvel_copy.rename(name_dict={"UVELMASS":"maskW"})
	maskW_ds = (maskW_ds["maskW"]*0+1.)

	vvel_copy = VVELMASS_ds.copy(deep=True)
	maskS_ds = vvel_copy.rename(name_dict={"VVELMASS":"maskS"})
	maskS_ds = (maskS_ds["maskS"]*0+1.)



	# merge everything into one dataset for future reference
	if include_grid == True:
		dataset = xr.merge([grid,UVELMASS_ds,VVELMASS_ds,maskW_ds,maskS_ds]).set_coords(["maskW","maskS"])
	else:
		dataset = xr.merge([UVELMASS_ds,VVELMASS_ds,maskW_ds,maskS_ds]).set_coords(["maskW","maskS"])

	if save == True:
		fname = "xy_velmass_" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + ".nc"
		i = 0
		while os.path.isfile(fname) == True:
			i += 1
			fname = "xy_velmass_" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + str(i) + ".nc"
		dataset.to_netcdf(fname)
		print("Saved UVELMASS and VVELMASS data for timesteps " + str(time_slice[0]) + " to " + str(time_slice[-1]) + " to file " + str(fname))

		return dataset, fname
	else:
		return dataset



def merge_bolus_u_bolus_v_into_dataset(time_slice,lat_vals,save=False):
	"""Merge UVELMASS and VVELMASS into single dataset along with an xmitgcm grid object. 
	This assumes that we are using the raw ECCO data files and as such has some extra steps that
	verify things like dimensions and coordinates are set up correctly


	Parameters
	----------
	path: string
		path to datafiles
	VAR: string 
		name of variable to extract, eg 'UVELMASS'
	grid_path: string
		path to grid file, 

	Returns
	_______
	variable_all_tiles: xarray dataset
		Xarray datset with tile files stacked along 'tile' dimension
	fname: string
		name of saved file
	"""
	# move the load grid command outside of this function for final implementation
	grid_path = "./ecco_grid/ECCOv4r3_grid.nc"
	grid = xr.open_dataset(grid_path)
	GM_PSIX_var = "GM_PsiX"
	GM_PSIY_var = "GM_PsiY"
	tile_data_dir = "./nctiles_monthly/"

	GM_PSIX_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(tile_data_dir, 
                                                  					GM_PSIX_var, 
                                                  					time_slice)
	GM_PSIY_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(tile_data_dir, 
                                                  					GM_PSIY_var, 
                                                  					time_slice)
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.drop("lon").drop("lat")
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.drop("lon").drop("lat")

	# do some post-processing..
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90))
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90))

	# trim datasets if final nan padding value is present.. otherwise this won't change anything
	GM_PSIX_ds_raw = GM_PSIX_ds_raw.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	GM_PSIY_ds_raw = GM_PSIY_ds_raw.isel(i=slice(0,90),j=slice(0,90),k=slice(0,50))
	GM_PSIX_ds_raw.load()
	GM_PSIY_ds_raw.load()


	# convert to dask array for computation speed..?	
	GM_PSIX_ds_raw["GM_PsiX"] = GM_PSIX_ds_raw.GM_PsiX.chunk(GM_PsiX.size)
	GM_PSIY_ds_raw["GM_PsiY"] = GM_PSIY_ds_raw.GM_PsiY.chunk(GM_PsiY.size)

	k_new_coords = np.arange(0,51)
	GM_PSIX_ds_plus_extra_k.coords.update({'k':k_new_coords})
	GM_PSIY_ds_plus_extra_k.coords.update({'k':k_new_coords})

	bolus_u = GM_PSIX_ds_plus_extra_k.copy(deep=True)
	bolus_v = GM_PSIY_ds_plus_extra_k.copy(deep=True)
	bolus_u = bolus_u.rename({'GM_PsiX':'bolus_uvel'})
	bolus_v = bolus_v.rename({'GM_PsiY':'bolus_vvel'})
	bolus_u.load()
	bolus_v.load()

	# perform finite difference step
	for k in range(0,50):
	    bolus_u.bolus_uvel[:,:,k,:,:] = (GM_PSIX_ds_plus_extra_k.GM_PsiX[:,:,k,:,:] - GM_PSIX_ds_plus_extra_k.GM_PsiX[:,:,k+1,:,:])/grid.drF[k]
	    bolus_v.bolus_vvel[:,:,k,:,:] = (GM_PSIY_ds_plus_extra_k.GM_PsiY[:,:,k,:,:] - GM_PSIY_ds_plus_extra_k.GM_PsiY[:,:,k+1,:,:])/grid.drF[k]


	bolus_u_copy = bolus_u.copy(deep=True)
	maskW_ds = bolus_u_copy.rename(name_dict={"UVELMASS":"maskW"})
	maskW_ds = (maskW_ds["maskW"]*0+1.)

	bolus_v_copy = bolus_v.copy(deep=True)
	maskS_ds = bolus_v_copy.rename(name_dict={"VVELMASS":"maskS"})
	maskS_ds = (maskS_ds["maskS"]*0+1.)

	bolus_u_final = bolus_u.isel(k=slice(0,50)).copy(deep=True).rename({'i':'i_g'})
	bolus_v_final = bolus_v.isel(k=slice(0,50)).copy(deep=True).rename({'j':'j_g'})

	# merge everything into one dataset for future reference
	dataset = xr.merge([grid,bolus_u_final,bolus_v_final,maskW_ds,maskS_ds]).set_coords(["maskW","maskS"])

	if save == True:
		fname = "xy_bolus_" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + ".nc"
		i = 0
		while os.path.isfile(fname) == True:
			i += 1
			fname = "xy_velmass_" + str(time_slice[0]) + "_to_" + str(time_slice[-1]) + str(i) + ".nc"
		dataset.to_netcdf(fname)
		print("Saved UVELMASS and VVELMASS data for timesteps " + str(time_slice[0]) + " to " + str(time_slice[-1]) + " to file " + str(fname))

	return dataset, fname