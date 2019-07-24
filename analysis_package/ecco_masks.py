import ecco_v4_py as ecco 
import numpy as np
import xarray as xr 

def get_basin_masks(maskW, maskS, maskC):
	
	mexico_mask_W, mexico_mask_S = ecco.get_basin_mask("mexico",maskW), ecco.get_basin_mask("mexico",maskS)
	baffin_mask_W, baffin_mask_S = ecco.get_basin_mask("baffin",maskW), ecco.get_basin_mask("baffin",maskS)
	north_mask_W, north_mask_S = ecco.get_basin_mask("north",maskW), ecco.get_basin_mask("north",maskS)
	hudson_mask_W, hudson_mask_S = ecco.get_basin_mask("hudson",maskW), ecco.get_basin_mask("hudson",maskS)
	gin_mask_W, gin_mask_S = ecco.get_basin_mask("gin",maskW), ecco.get_basin_mask("gin",maskS)
	atl_mask_W, atl_mask_S = ecco.get_basin_mask("atl",maskW), ecco.get_basin_mask("atl",maskS)

	full_atl_basin_mask_W = atl_mask_W + baffin_mask_W + north_mask_W + gin_mask_W + mexico_mask_W + hudson_mask_W
	full_atl_basin_mask_S = atl_mask_S + baffin_mask_S + north_mask_S + gin_mask_S + mexico_mask_S + hudson_mask_S


	ind_mask_W, ind_mask_S = ecco.get_basin_mask("ind",maskW), ecco.get_basin_mask("ind",maskS)
	pac_mask_W, pac_mask_S = ecco.get_basin_mask("pac",maskW), ecco.get_basin_mask("pac",maskS)
	southChina_mask_W, southChina_mask_S = ecco.get_basin_mask("southChina",maskW), ecco.get_basin_mask("southChina",maskS)
	japan_W, japan_S = ecco.get_basin_mask("japan",maskW), ecco.get_basin_mask("japan",maskS)
	eastChina_W, eastChina_S = ecco.get_basin_mask("eastChina",maskW), ecco.get_basin_mask("eastChina",maskS)
	timor_W, timor_S = ecco.get_basin_mask("timor",maskW), ecco.get_basin_mask("timor",maskS)
	java_W, java_S = ecco.get_basin_mask("java",maskW), ecco.get_basin_mask("java",maskS)
	bering_mask_W, bering_mask_S = ecco.get_basin_mask("bering",maskW), ecco.get_basin_mask("bering",maskS)
	okhotsk_mask_W, okhotsk_mask_S = ecco.get_basin_mask("okhotsk",maskW), ecco.get_basin_mask("okhotsk",maskS)

	full_indpac_basin_mask_W = (ind_mask_W + pac_mask_W + southChina_mask_W + japan_W + eastChina_W 
	                            + timor_W + java_W + okhotsk_mask_W + bering_mask_W)
	full_indpac_basin_mask_S = (ind_mask_S + pac_mask_S + southChina_mask_S + japan_S + eastChina_S 
	                            + timor_S + java_S + okhotsk_mask_S + bering_mask_S)

	j_transport_coords = np.arange(0,90)
	i_transport_coords = np.arange(0,90)
	# WATCH OUT FOR THIS INDEXING!
	tile_coords = np.arange(0,13)

	southern_ocean_mask_C = (maskC.where(maskC["lat"] < -33)*0 + 1.)
	southern_ocean_mask_C = southern_ocean_mask_C.assign_coords(j=j_transport_coords,
	                                                            i=i_transport_coords,
	                                                            tile=tile_coords)

	southern_ocean_mask_W = southern_ocean_mask_C.rename({"i":"i_g"}).drop("lon").drop("lat")
	southern_ocean_mask_S = southern_ocean_mask_C.rename({"j":"j_g"}).drop("lon").drop("lat")

	so_atl_basin_mask_W = full_atl_basin_mask_W + southern_ocean_mask_W.where(southern_ocean_mask_W==1, other=0) 
	so_atl_basin_mask_S = full_atl_basin_mask_S + southern_ocean_mask_S.where(southern_ocean_mask_S==1, other=0)
	so_atl_basin_mask_W = so_atl_basin_mask_W.where(so_atl_basin_mask_W > 0, other=np.nan)*0 + 1
	so_atl_basin_mask_S = so_atl_basin_mask_S.where(so_atl_basin_mask_S > 0, other=np.nan)*0 + 1


	so_indpac_basin_mask_W = full_indpac_basin_mask_W.where(full_indpac_basin_mask_W==1,other=0) + southern_ocean_mask_W.where(southern_ocean_mask_W==1,other=0) 
	so_indpac_basin_mask_S = full_indpac_basin_mask_S.where(full_indpac_basin_mask_S==1,other=0) + southern_ocean_mask_S.where(southern_ocean_mask_S==1,other=0)
	so_indpac_basin_mask_W = so_indpac_basin_mask_W.where(so_indpac_basin_mask_W != 0, other=np.nan)*0 + 1
	so_indpac_basin_mask_S = so_indpac_basin_mask_S.where(so_indpac_basin_mask_S != 0, other=np.nan)*0 + 1

	return southern_ocean_mask_W, southern_ocean_mask_S, so_atl_basin_mask_W, so_atl_basin_mask_S, so_indpac_basin_mask_W, so_indpac_basin_mask_S