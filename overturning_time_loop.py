import numpy as np
import xarray as xr
from importlib import reload

from analysis_package import plotting_functions
from analysis_package import open_datasets
from analysis_package import derive_potential_density_values_TEST
from analysis_package import ecco_masks
from analysis_package import potential_density_overturning

ecco_masks = reload(ecco_masks)
plotting_functions = reload(plotting_functions)
open_datasets = reload(open_datasets)
derive_potential_density_values_TEST = reload(derive_potential_density_values_TEST)
potential_density_overturning = reload(potential_density_overturning)

data_dir = "./nctiles_monthly/"

UVELMASS_var = "UVELMASS"
VVELMASS_var = "VVELMASS"
GM_PSIX_var = "GM_PsiX"
GM_PSIY_var = "GM_PsiY"

# RHOAnoma: insitu density anomaly
RHOAnoma_var_str = "RHOAnoma"
# PHIHYD: insitu pressure anomaly with respect to the depth integral of gravity and reference density (g*rho_reference)
PHIHYD_var_str = "PHIHYD"
# SALT: insitu salinity (psu)
SALT_var_str = "SALT"
# THETA: potential pressure (C)
THETA_var_str = "THETA"
# PDENS
PDENS_U_var_str = "PDENS_U"
# PDENS
PDENS_V_var_str = "PDENS_V"

grid_path = "./ecco_grid/ECCOv4r3_grid.nc"
grid = xr.open_dataset(grid_path)
print("loaded grid")

# make sure to use the "open_dataarray" command instead of open "open_dataset" 
# or xarray will read in the file incorrectly


time_slice = []
for i in range(0,24):
    time_slice.append(np.arange(i*12,(i+1)*12))

for t_slice in time_slice:
    
    # load data files from central directory
    UVELMASS_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, UVELMASS_var, t_slice)
    VVELMASS_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, VVELMASS_var, t_slice)
    GM_PSIX_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir,GM_PSIX_var, t_slice)
    GM_PSIY_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, GM_PSIY_var, t_slice)

    tiles = np.arange(0,13)

    GM_PSIX_ds_raw["tile"] = tiles
    GM_PSIY_ds_raw["tile"] = tiles
    GM_PSIX_ds_raw = GM_PSIX_ds_raw.drop("lon").drop("lat")
    GM_PSIY_ds_raw = GM_PSIY_ds_raw.drop("lon").drop("lat")
    GM_PSIX_ds_raw = GM_PSIX_ds_raw.set_coords(["tile"]).drop("land").drop("area").drop("thic")
    GM_PSIY_ds_raw = GM_PSIY_ds_raw.set_coords(["tile"]).drop("land").drop("area").drop("thic")
    # do some post-processing..
    GM_PSIX_ds_raw = GM_PSIX_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90),time=t_slice)
    GM_PSIY_ds_raw = GM_PSIY_ds_raw.assign_coords(k=np.arange(0,50),j=np.arange(0,90),i=np.arange(0,90),time=t_slice)
    GM_PSIX_ds_raw["GM_PsiX"] = GM_PSIX_ds_raw.GM_PsiX.chunk((13,len(t_slice),50,90,90)).rename({"i":"i_g"}) 
    GM_PSIY_ds_raw["GM_PsiY"] = GM_PSIY_ds_raw.GM_PsiY.chunk((13,len(t_slice),50,90,90)).rename({"j":"j_g"})

    # set data file indecies starting from zero.
    UVELMASS_ds_raw = UVELMASS_ds_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    VVELMASS_ds_raw = VVELMASS_ds_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    GM_PSIX_ds_raw = GM_PSIX_ds_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    GM_PSIY_ds_raw = GM_PSIY_ds_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    #PDENS_ds = PDENS_ds.assign_coords(i=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50))
    # calculate potential density and in situ pressure
    PDENS_U_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir,PDENS_U_var_str,t_slice,rename_indices=False)
    # set data file indecies starting from zero.
    PDENS_U_ds = PDENS_U_ds_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50))
    # calculate potential density and in situ pressure
    PDENS_V_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir,PDENS_V_var_str,t_slice,rename_indices=False)
    # set data file indecies starting from zero.
    PDENS_V_ds = PDENS_V_ds_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50))

    depth_integrated_trsp_x, depth_integrated_trsp_y = potential_density_overturning.perform_potential_density_overturning_calculation(t_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw, GM_PSIX_ds_raw, GM_PSIY_ds_raw)

    depth_integrated_trsp_x.to_netcdf("./overturning_output/depth_integrated_trsp_x"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    depth_integrated_trsp_y.to_netcdf("./overturning_output/depth_integrated_trsp_y"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    #plotting_functions.overturning_output_plots(atl_depth_integrated_pdens_transport_latx, atl_depth_integrated_pdens_transport_laty,t_slice,"atl_so")
    
    #indpac_depth_integrated_pdens_transport_latx.to_netcdf("./overturning_output/indpac_so_depth_integrated_pdens_transport_latx_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    #indpac_depth_integrated_pdens_transport_laty.to_netcdf("./overturning_output/indpac_so_depth_integrated_pdens_transport_laty_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    #plotting_functions.overturning_output_plots(indpac_depth_integrated_pdens_transport_latx, indpac_depth_integrated_pdens_transport_laty,t_slice,"indpac_so")
    
    #global_depth_integrated_pdens_transport_latx.to_netcdf("./overturning_output/global_depth_integrated_pdens_transport_latx_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    #global_depth_integrated_pdens_transport_laty.to_netcdf("./overturning_output/global_depth_integrated_pdens_transport_laty_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    #plotting_functions.overturning_output_plots(global_depth_integrated_pdens_transport_latx, global_depth_integrated_pdens_transport_laty,t_slice,"global")
   
    print()
