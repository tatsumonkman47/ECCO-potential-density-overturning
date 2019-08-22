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
BOLUS_UVEL_var = "BOLUS_UVEL"
BOLUS_VVEL_var = "BOLUS_VVEL"

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

# make sure to use the "open_dataarray" command instead of open "open_dataset" 
# or xarray will read in the file incorrectly

maskW = xr.open_dataarray("generic_masks/maskW.nc")
maskS = xr.open_dataarray("generic_masks/maskS.nc")
maskC = xr.open_dataarray("generic_masks/maskC.nc")
southern_ocean_mask_W, southern_ocean_mask_S, so_atl_basin_mask_W, so_atl_basin_mask_S, so_indpac_basin_mask_W, so_indpac_basin_mask_S = ecco_masks.get_basin_masks(maskW, maskS, maskC)

time_slice = []
for i in range(0,24):
    time_slice.append(np.arange(i*12,(i+1)*12))

for t_slice in time_slice:
    
    # load data files from central directory
    UVELMASS_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, UVELMASS_var, t_slice)
    VVELMASS_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, VVELMASS_var, t_slice)
    BOLUS_UVEL_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, BOLUS_UVEL_var, t_slice, rename_indices=False)
    BOLUS_VVEL_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir, BOLUS_VVEL_var, t_slice, rename_indices=False)

    # set data file indecies starting from zero.
    UVELMASS_ds_raw = UVELMASS_ds_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    VVELMASS_ds_raw = VVELMASS_ds_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    BOLUS_UVEL_raw = BOLUS_UVEL_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    BOLUS_VVEL_raw = BOLUS_VVEL_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50),time=t_slice)
    #PDENS_ds = PDENS_ds.assign_coords(i=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50))
    # calculate potential density and in situ pressure
    PDENS_U_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir,PDENS_U_var_str,t_slice,rename_indices=False)
    # set data file indecies starting from zero.
    PDENS_U_ds = PDENS_U_ds_raw.assign_coords(i_g=np.arange(0,90),j=np.arange(0,90),k=np.arange(0,50))
    # calculate potential density and in situ pressure
    PDENS_V_ds_raw = open_datasets.open_combine_raw_ECCO_tile_files(data_dir,PDENS_V_var_str,t_slice,rename_indices=False)
    # set data file indecies starting from zero.
    PDENS_V_ds = PDENS_V_ds_raw.assign_coords(i=np.arange(0,90),j_g=np.arange(0,90),k=np.arange(0,50))

    dint_latx, dint_laty, dint_interp_latx, dint_interp_laty = potential_density_overturning.perform_potential_density_overturning_calculation(t_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw,
                                                                                                                                               BOLUS_UVEL_raw, BOLUS_VVEL_raw, 
                                                                                                                                               so_atl_basin_mask_W,so_atl_basin_mask_S)
    dint_latx.to_netcdf("./overturning_output/atl_so_depth_integrated_pdens_transport_latx_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_laty.to_netcdf("./overturning_output/atl_so_depth_integrated_pdens_transport_laty_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_latx.to_netcdf("./overturning_output/atl_so_depth_integrated_x_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_laty.to_netcdf("./overturning_output/atl_so_depth_integrated_y_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    plotting_functions.overturning_output_plots(dint_latx, dint_laty, dint_interp_latx,dint_interp_laty,t_slice,"atl_so")
    
    dint_latx, dint_laty, dint_interp_latx, dint_interp_laty = potential_density_overturning.perform_potential_density_overturning_calculation(t_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw,
                                                                                                                                               BOLUS_UVEL_raw, BOLUS_VVEL_raw,
                                                                                                                                               so_indpac_basin_mask_W,so_indpac_basin_mask_S)
    dint_latx.to_netcdf("./overturning_output/indpac_so_depth_integrated_pdens_transport_latx_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_laty.to_netcdf("./overturning_output/indpac_so_depth_integrated_pdens_transport_laty_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_latx.to_netcdf("./overturning_output/indpac_so_depth_integrated_x_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_laty.to_netcdf("./overturning_output/indpac_so_depth_integrated_y_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    plotting_functions.overturning_output_plots(dint_latx, dint_laty,dint_interp_latx,dint_interp_laty,t_slice,"indpac_so")
    
    dint_latx, dint_laty, dint_interp_latx, dint_interp_laty = potential_density_overturning.perform_potential_density_overturning_calculation(t_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw,
                                                                                                                                               BOLUS_UVEL_raw, BOLUS_VVEL_raw,
                                                                                                                                               maskW,maskS)    
    dint_latx.to_netcdf("./overturning_output/global_depth_integrated_pdens_transport_latx_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_laty.to_netcdf("./overturning_output/global_depth_integrated_pdens_transport_laty_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_latx.to_netcdf("./overturning_output/global_depth_integrated_x_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    dint_interp_laty.to_netcdf("./overturning_output/global_depth_integrated_y_interp_results_"+str(t_slice[0])+"_to_"+str(t_slice[-1])+".nc")
    plotting_functions.overturning_output_plots(dint_latx, dint_laty,dint_interp_latx,dint_interp_laty,t_slice,"global")
   
    print()
