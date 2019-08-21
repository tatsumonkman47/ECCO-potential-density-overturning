import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr


from xmitgcm import open_mdsdataset
import xmitgcm
import ecco_v4_py as ecco


from netCDF4 import Dataset

import seawater

from analysis_package import plotting_functions
from analysis_package import open_datasets

from analysis_package import derive_potential_density_values_TEST
from analysis_package import ecco_masks
#from analysis_scripts import integrate_on_density_surfaces.py

from importlib import reload

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

grid_path = "./ecco_grid/ECCOv4r3_grid.nc"
grid = xr.open_dataset(grid_path)



def perform_potential_density_overturning_calculation(time_slice,PDENS_U_ds,PDENS_V_ds,UVELMASS_ds_raw,VVELMASS_ds_raw,
														BOLUS_UVEL_raw, BOLUS_VVEL_raw, basin_maskW,basin_maskS):

	cds = grid.coords.to_dataset()
	grid_xmitgcm = ecco.ecco_utils.get_llc_grid(cds)

	transport_x = (UVELMASS_ds_raw["UVELMASS"]*grid["drF"]*grid["dyG"] 
	               + BOLUS_UVEL_raw["bolus_uvel"]*grid["drF"]*grid["dyG"]*grid["hFacW"])*basin_maskW
	transport_y = (VVELMASS_ds_raw["VVELMASS"]*grid["drF"]*grid["dxG"] 
	               + BOLUS_VVEL_raw["bolus_vvel"]*grid["drF"]*grid["dxG"]*grid["hFacS"])*basin_maskS

	# create infrastructure for integrating in depth space

	lat_vals = np.arange(-88,88)

	# create an empty array with a stretched depth dimension
	# Set the coordinates of the stretched depth dimension to potential density values..
	# add padding to either end of the pot. density coordinates
	# just trying with slightly coarser resolution 
	#(what pot density resolution is valid in this case?)
	pot_dens_coord = np.arange(1032.0,1033.9,0.2)

	pot_dens_coord = np.concatenate((np.asarray([1000.]),pot_dens_coord, np.arange(1034.0,1038,0.1)))

	# set dimensions based on input dataset with modified vertical level spacing..
	pot_dens_dims = (len(time_slice),
	                 len(pot_dens_coord),
	                 len(lat_vals))

	empty_pot_coords_data = np.zeros(pot_dens_dims)
	# trying to make this as general as possible, but need to keep an eye on this..
	new_coords = [time_slice, pot_dens_coord, lat_vals]
	new_dims = ["time", "pot_rho", "lat"]

	pot_dens_array_x = PDENS_U_ds.PDENS.copy(deep=True)*basin_maskW
	pot_dens_array_y = PDENS_V_ds.PDENS.copy(deep=True)*basin_maskS



	depth_integrated_pdens_transport = xr.DataArray(data=empty_pot_coords_data,coords=new_coords,dims=new_dims)
	depth_integrated_pdens_transport.load()
	depth_integrated_pdens_transport_latx = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_pdens_transport_latx.load()
	depth_integrated_pdens_transport_laty = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_pdens_transport_laty.load()

	depth_integrated_x_interp_results = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_x_interp_results.load()
	depth_integrated_y_interp_results = depth_integrated_pdens_transport.copy(deep=True)
	depth_integrated_y_interp_results.load()

	for density in pot_dens_coord:
	    print("Started " + str(density) + " surface for time slice " + str(time_slice)) 
	    potdens_stencil_x_0 = pot_dens_array_x > density
	    potdens_stencil_y_0 = pot_dens_array_y > density
	    # this step is critical to remove low density anomalies in the deep ocean from stencil...
	    potdens_stencil_x = potdens_stencil_x_0.cumsum(dim="k") > 0
	    potdens_stencil_y = potdens_stencil_y_0.cumsum(dim="k") > 0
	    print("got to checkpoint 0")

	    ############################################################################################################
	    ###########################################     START INTERPOLATION    #####################################
	    ############################################################################################################
	    # set end-appended value equal to 1 for subtraction step..
	    potdens_stencil_x_shifted_up_one_cell = xr.concat((potdens_stencil_x.isel(k=slice(1,50)),
	                                                       potdens_stencil_x.isel(k=49)),
	                                                      dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_y_shifted_up_one_cell = xr.concat((potdens_stencil_y.isel(k=slice(1,50)),
	                                                       potdens_stencil_y.isel(k=49)),
	                                                      dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_x_shifted_down_one_cell = xr.concat((potdens_stencil_x.isel(k=0)*0,
	                                                         potdens_stencil_x.isel(k=slice(0,49))),
	                                                        dim="k").assign_coords(k=np.arange(0,50))
	    potdens_stencil_y_shifted_down_one_cell = xr.concat((potdens_stencil_y.isel(k=0)*0,
	                                                         potdens_stencil_y.isel(k=slice(0,49))),
	                                                        dim="k").assign_coords(k=np.arange(0,50))

	    potdens_stencil_x_one_above_top_level = potdens_stencil_x_shifted_up_one_cell*1 - potdens_stencil_x*1
	    potdens_stencil_y_one_above_top_level = potdens_stencil_y_shifted_up_one_cell*1 - potdens_stencil_y*1
	    # get rid of trailing negative values that occur at the ocean's bottom boundary..
	    potdens_stencil_x_one_above_top_level = potdens_stencil_x_one_above_top_level.where(potdens_stencil_x_one_above_top_level > 0,
	                                                                                        other=0)
	    potdens_stencil_y_one_above_top_level = potdens_stencil_y_one_above_top_level.where(potdens_stencil_y_one_above_top_level > 0,
	                                                                                        other=0)

	    potdens_stencil_x_top_level = potdens_stencil_x*1 - potdens_stencil_x_shifted_down_one_cell*1
	    potdens_stencil_y_top_level = potdens_stencil_y*1 - potdens_stencil_y_shifted_down_one_cell*1
	    # turn zeros into nans..
	    # NOTE SOMETIMES YOU GET PROTRUSIONS OF DENSITY ANOMALIES THAT SEEM TO CREATE TWO DENSITY SURFACES, LEADING TO A VALUE OF
	    # 2 IN THE STENCIL.. I eliminated this using "potdens_stencil_x = potdens_stencil_x_0.cumsum(dim="k") > 0" a couple lines above.
	    potdens_stencil_x_top_level = potdens_stencil_x_top_level.where(potdens_stencil_x_top_level > 0, other=np.nan)
	    potdens_stencil_y_top_level = potdens_stencil_y_top_level.where(potdens_stencil_y_top_level > 0, other=np.nan)
	    potdens_stencil_x_one_above_top_level = potdens_stencil_x_one_above_top_level.where(potdens_stencil_x_one_above_top_level > 0,
	                                                                                        other=np.nan)
	    potdens_stencil_y_one_above_top_level = potdens_stencil_y_one_above_top_level.where(potdens_stencil_y_one_above_top_level > 0,
	                                                                                        other=np.nan)
	    # multiply depth values by -1 to make them positive..
	    depth_above_x_top_level_raw = (-1*potdens_stencil_x_one_above_top_level.fillna(0)*transport_x.Z).sum(dim="k")
	    depth_x_top_level_raw = (-1*potdens_stencil_x_top_level.fillna(0)*transport_x.Z).sum(dim="k",skipna=True)
	    depth_above_y_top_level_raw = (-1*potdens_stencil_y_one_above_top_level.fillna(0)*transport_y.Z).sum(dim="k",skipna=True)
	    depth_y_top_level_raw = (-1*potdens_stencil_y_top_level.fillna(0)*transport_y.Z).sum(dim="k",skipna=True)
	    # turn zeros into nans..
	    depth_above_x_top_level = depth_above_x_top_level_raw.where(depth_above_x_top_level_raw > 0, other=np.nan)
	    depth_x_top_level = depth_x_top_level_raw.where(depth_x_top_level_raw > 0, other=np.nan)
	    depth_above_y_top_level = depth_above_y_top_level_raw.where(depth_above_y_top_level_raw > 0, other=np.nan)
	    depth_y_top_level = depth_y_top_level_raw.where(depth_y_top_level_raw > 0, other=np.nan)

	    thickness_above_x_top_level_raw = (potdens_stencil_x_one_above_top_level.fillna(0)*grid["drF"]).sum(dim="k")
	    thickness_x_top_level_raw = (potdens_stencil_x_top_level.fillna(0)*grid["drF"]).sum(dim="k")
	    thickness_above_y_top_level_raw = (potdens_stencil_y_one_above_top_level.fillna(0)*grid["drF"]).sum(dim="k")
	    thickness_y_top_level_raw = (potdens_stencil_y_top_level.fillna(0)*grid["drF"]).sum(dim="k")
	    # turn zeros into nans..
	    thickness_above_x_top_level = thickness_above_x_top_level_raw.where(thickness_above_x_top_level_raw > 0, other=np.nan)
	    thickness_x_top_level = thickness_x_top_level_raw.where(thickness_x_top_level_raw > 0, other=np.nan)
	    thickness_above_y_top_level = thickness_above_y_top_level_raw.where(thickness_above_y_top_level_raw > 0, other=np.nan)
	    thickness_y_top_level = thickness_y_top_level_raw.where(thickness_y_top_level_raw > 0, other=np.nan)    
	    
	    potdens_above_x_top_level = (potdens_stencil_x_one_above_top_level.fillna(0)*pot_dens_array_x.fillna(0)).sum(dim="k")
	    potdens_x_top_level = (potdens_stencil_x_top_level.fillna(0)*pot_dens_array_x.fillna(0)).sum(dim="k")
	    potdens_above_y_top_level = (potdens_stencil_y_one_above_top_level.fillna(0)*pot_dens_array_y.fillna(0)).sum(dim="k")
	    potdens_y_top_level = (potdens_stencil_y_top_level.fillna(0)*pot_dens_array_y.fillna(0)).sum(dim="k")
	    # turn zeros into nans..
	    potdens_above_x_top_level = potdens_above_x_top_level.where(potdens_above_x_top_level > 0, other=np.nan)
	    potdens_x_top_level = potdens_x_top_level.where(potdens_x_top_level > 0, other=np.nan)
	    potdens_above_y_top_level = potdens_above_y_top_level.where(potdens_above_y_top_level > 0, other=np.nan)
	    potdens_y_top_level = potdens_y_top_level.where(potdens_y_top_level > 0, other=np.nan)
	    
	    print("got to checkpoint 1")
	    # need to calculate transport per m so divide by cell thickness
	    transport_above_x_top_level = (potdens_stencil_x_one_above_top_level.fillna(0)*transport_x).sum(dim="k")
	    transport_x_top_level = (potdens_stencil_x_top_level.fillna(0)*transport_x).sum(dim="k")
	    transport_above_y_top_level = (potdens_stencil_y_one_above_top_level.fillna(0)*transport_y).sum(dim="k")
	    transport_y_top_level = (potdens_stencil_y_top_level.fillna(0)*transport_y).sum(dim="k")
	    
	    transport_above_x_top_level = transport_above_x_top_level.where(transport_above_x_top_level !=0, other=np.nan)    
	    transport_x_top_level = transport_x_top_level.where(transport_x_top_level != 0, other=np.nan)
	    transport_above_y_top_level = transport_above_y_top_level.where(transport_above_y_top_level !=0, other=np.nan)
	    transport_y_top_level = transport_y_top_level.where(transport_y_top_level !=0, other=np.nan)
	    
	    depth_potdens_slope_x = (depth_above_x_top_level - depth_x_top_level)/(potdens_above_x_top_level - potdens_x_top_level)                                                         
	    depth_potdens_slope_y = (depth_above_y_top_level - depth_y_top_level)/(potdens_above_y_top_level - potdens_y_top_level)
	        
	    # this is an issue... need to account for those low desnity protrusions..
	    h_array_x_0 = (density - potdens_x_top_level)*depth_potdens_slope_x
	    h_array_x = h_array_x_0.where(h_array_x_0 < 0, other=0 )*-1
	    h_array_y_0 = (density - potdens_y_top_level)*depth_potdens_slope_y 
	    h_array_y = h_array_y_0.where(h_array_y_0 < 0, other=0 )*-1
	    
	    print("got to checkpoint 2")
	    half_thickness_x_top_level = (potdens_stencil_x_top_level.fillna(0)*grid.drF/2.).sum(dim="k")
	    half_thickness_y_top_level = (potdens_stencil_y_top_level.fillna(0)*grid.drF/2.).sum(dim="k")
	    half_thickness_x_one_above_top_level = (potdens_stencil_x_one_above_top_level.fillna(0)*grid.drF/2.).sum(dim="k")
	    half_thickness_y_one_above_top_level = (potdens_stencil_y_one_above_top_level.fillna(0)*grid.drF/2.).sum(dim="k")
	    
	    overshoot_x_0 = h_array_x - half_thickness_x_top_level
	    overshoot_x = overshoot_x_0.where(overshoot_x_0 > 0).fillna(0)
	    undershoot_x = overshoot_x_0.where(overshoot_x_0 <= 0).fillna(0)
	    overshoot_y_0 = h_array_y - half_thickness_y_top_level
	    overshoot_y = overshoot_y_0.where(overshoot_y_0 > 0).fillna(0)
	    undershoot_y = overshoot_y_0.where(overshoot_y_0 <= 0).fillna(0)
	    
	    # remember we are interpolating from the center of the top cell
	    percent_top_filled_x = (half_thickness_x_top_level + undershoot_x)/(half_thickness_x_top_level*2)
	    percent_top_filled_y = (half_thickness_y_top_level + undershoot_y)/(half_thickness_y_top_level*2)
	    
	    percent_one_above_top_filled_x = overshoot_x/(half_thickness_x_one_above_top_level*2)
	    percent_one_above_top_filled_y = overshoot_y/(half_thickness_y_one_above_top_level*2)  
	    
	    trsp_interpolated_x = (percent_top_filled_x*transport_x_top_level).fillna(0) + (percent_one_above_top_filled_x*transport_above_x_top_level).fillna(0)
	    trsp_interpolated_y = (percent_top_filled_y*transport_y_top_level).fillna(0) + (percent_one_above_top_filled_y*transport_above_y_top_level).fillna(0)
	    
	    # "transport_integral_x/y" is the vertical sum of the interpolated grid cell tranposrt
	    trsp_interpolated_x.load()
	    trsp_interpolated_y.load()g
	    ############################################################################################################
	    ###########################################     END INTERPOLATION    #######################################
	    ############################################################################################################    
	    
	    # split the top cell in half since we are putting it into the interpolation,
	    # but only in cases where there actually is a cell above it.
	    depth_integrated_trsp_x = transport_x*(potdens_stencil_x.where(potdens_stencil_x>0,other=np.nan))
	    depth_integrated_trsp_x.load()
	    depth_integrated_trsp_x = depth_integrated_trsp_x.sum(dim='k') + trsp_interpolated_x.fillna(0)
	    
	    depth_integrated_trsp_y = transport_y*(potdens_stencil_y.where(potdens_stencil_y>0,other=np.nan))
	    depth_integrated_trsp_y.load()
	    depth_integrated_trsp_y = depth_integrated_trsp_y.sum(dim='k') + trsp_interpolated_y.fillna(0)
	    
	    depth_integrated_trsp_x_no_interp = (transport_x*potdens_stencil_x).sum(dim='k')
	    depth_integrated_trsp_x_no_interp.load()
	    depth_integrated_trsp_y_no_interp = (transport_y*potdens_stencil_y).sum(dim='k')
	    depth_integrated_trsp_y_no_interp.load()
	                                          
	                                           
	    print('starting lat-band filtering')
	    for lat in lat_vals:
	        # Compute mask for particular latitude band
	        print(str(lat)+' ',end='')
	        lat_maskW, lat_maskS = ecco.vector_calc.get_latitude_masks(lat, cds['YC'], grid_xmitgcm)
	       
	        # Sum horizontally
	        lat_trsp_x = (depth_integrated_trsp_x * lat_maskW).sum(dim=['i_g','j','tile'],skipna=True)
	        lat_trsp_y = (depth_integrated_trsp_y * lat_maskS).sum(dim=['i','j_g','tile'],skipna=True)
	        lat_trsp_x_interp = (trsp_interpolated_x * lat_maskW).sum(dim=['i_g','j','tile'],skipna=True)
	        lat_trsp_y_interp = (trsp_interpolated_y * lat_maskS).sum(dim=['i','j_g','tile'],skipna=True)
	        
	        depth_integrated_pdens_transport_latx.loc[{'lat':lat,'pot_rho':density}] = lat_trsp_x
	        depth_integrated_pdens_transport_laty.loc[{'lat':lat,'pot_rho':density}] = lat_trsp_y
	        depth_integrated_x_interp_results.loc[{'lat':lat,'pot_rho':density}] = lat_trsp_x_interp
	        depth_integrated_y_interp_results.loc[{'lat':lat,'pot_rho':density}] = lat_trsp_y_interp

	    print("\n")
	    
	    
	return depth_integrated_pdens_transport_latx, depth_integrated_pdens_transport_laty, depth_integrated_x_interp_results, depth_integrated_y_interp_results
	# depth_integrated_pdens.to_netcdf("./depth_integrated_pdens_TEST11.nc")
