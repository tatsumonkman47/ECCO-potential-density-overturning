import matplotlib.gridspec as gridspec
import xarray as xr 
import matplotlib as mtp
import matplotlib.pyplot as plt
import numpy as np

def world_plot(dataset,label_tiles=True,ticks_on=False):
	"""
	# A function to all plot ECCO grid tiles simulateously
	# data should be entered as an xarray dataarray, not a dataset
	# data should have dimensions (tile, i, j) = (13,i,j)
	# return a matplotlib plot object..

	# need to enter variable with timestep and k-level already selected like
	# etc: RHOAnoma_ds.RHOAnoma.isel(time=1,k=1)
	"""


	tmp_plt = dataset

	# normalize colorscale across all grid 
	var_min = tmp_plt.min()
	var_max = tmp_plt.max()

	if var_min == var_max:
		var_min = None
		var_max = None

	# initialize figure and gridspec
	fig = plt.figure(figsize=(15,15))
	gs = gridspec.GridSpec(nrows=4,ncols=4,hspace=0.02,wspace=0.05)
	n_levels = 20

	# Eastern Atlantic Ocean tiles
	ax0 = fig.add_subplot(gs[3,1])
	ax1 = fig.add_subplot(gs[2,1])
	ax2 = fig.add_subplot(gs[1,1])

	# Indian Ocean tiles
	ax3 = fig.add_subplot(gs[3,2])
	ax4 = fig.add_subplot(gs[2,2])
	ax5 = fig.add_subplot(gs[1,2])

	# Arctic Ocean tile
	ax6 = fig.add_subplot(gs[0,2])

	# Pacific Ocean Tiles
	ax7 = fig.add_subplot(gs[1,3])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,3])

	# Western Atlantic Ocean tiles
	ax10 = fig.add_subplot(gs[1,0])
	ax11 = fig.add_subplot(gs[2,0])
	ax12 = fig.add_subplot(gs[3,0])



	im0 = ax0.contourf(tmp_plt.isel(tile=0),n_levels,vmin=var_min,vmax=var_max)
	im1 = ax1.contourf(tmp_plt.isel(tile=1),n_levels,vmin=var_min,vmax=var_max)
	im2 = ax2.contourf(tmp_plt.isel(tile=2),n_levels,vmin=var_min,vmax=var_max)	
	im3 = ax3.contourf(tmp_plt.isel(tile=3),n_levels,vmin=var_min,vmax=var_max)	
	im4 = ax4.contourf(tmp_plt.isel(tile=4),n_levels,vmin=var_min,vmax=var_max)	
	im5 = ax5.contourf(tmp_plt.isel(tile=5),n_levels,vmin=var_min,vmax=var_max)
	im6 = ax6.contourf(tmp_plt.isel(tile=6),n_levels,vmin=var_min,vmax=var_max)	

	# reorient flipped grid tiles..	
	im7 = ax7.contourf(np.flip(tmp_plt.isel(tile=7).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im8 = ax8.contourf(np.flip(tmp_plt.isel(tile=8).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im9 = ax9.contourf(np.flip(tmp_plt.isel(tile=9).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im10 = ax10.contourf(np.flip(tmp_plt.isel(tile=10).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im11 = ax11.contourf(np.flip(tmp_plt.isel(tile=11).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im12 = ax12.contourf(np.flip(tmp_plt.isel(tile=12).T,axis=0),n_levels,vmin=var_min,vmax=var_max)

	if label_tiles == True:
		ax0.set_title("tile 0")
		ax1.set_title("tile 1")
		ax2.set_title("tile 2")
		ax3.set_title("tile 3")
		ax4.set_title("tile 4")
		ax5.set_title("tile 5")
		ax6.set_title("tile 6")
		ax7.set_title("tile 7")
		ax8.set_title("tile 8")
		ax9.set_title("tile 9")
		ax10.set_title("tile 10")
		ax11.set_title("tile 11")
		ax12.set_title("tile 12")

	if ticks_on == False:
		ax0.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax1.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax2.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax3.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax4.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax5.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax6.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax7.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax8.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax9.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax10.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)	
		ax11.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax12.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)

	fig.subplots_adjust(right=0.8)
	cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])

	# I chose tile 8 since I don't know how to manually adjust the colorbar...
	cbar = fig.colorbar(im12,cax=cb_ax)

	# return the plot object..
	return plt 

#


def world_east_vel_plot(uveldataset, vveldataset, label_tiles=True,ticks_on=False):
	"""
	# A function to all plot ECCO grid tiles simulateously
	# data should be entered as an xarray dataarray, not a dataset
	# data should have dimensions (tile, i, j) = (13,i,j)
	# return a matplotlib plot object..

	# need to enter variable with timestep and k-level already selected like
	# etc: RHOAnoma_ds.RHOAnoma.isel(time=1,k=1)
	"""


	#tmp_plt = dataset

	# normalize colorscale across all grid 
	var_min = min(uveldataset.isel(tile=slice(0,7)).min(), vveldataset.isel(tile=slice(7,13)).min())
	var_max = max(uveldataset.isel(tile=slice(0,7)).max(), vveldataset.isel(tile=slice(7,13)).max())

	if var_min == var_max:
		var_min = None
		var_max = None

	# initialize figure and gridspec
	fig = plt.figure(figsize=(15,15))
	gs = gridspec.GridSpec(nrows=4,ncols=4,hspace=0.02,wspace=0.05)
	n_levels = 20

	# Eastern Atlantic Ocean tiles
	ax0 = fig.add_subplot(gs[3,1])
	ax1 = fig.add_subplot(gs[2,1])
	ax2 = fig.add_subplot(gs[1,1])

	# Indian Ocean tiles
	ax3 = fig.add_subplot(gs[3,2])
	ax4 = fig.add_subplot(gs[2,2])
	ax5 = fig.add_subplot(gs[1,2])

	# Arctic Ocean tile
	ax6 = fig.add_subplot(gs[0,2])

	# Pacific Ocean Tiles
	ax7 = fig.add_subplot(gs[1,3])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,3])

	# Western Atlantic Ocean tiles
	ax10 = fig.add_subplot(gs[1,0])
	ax11 = fig.add_subplot(gs[2,0])
	ax12 = fig.add_subplot(gs[3,0])



	im0 = ax0.contourf(uveldataset.isel(tile=0),n_levels,vmin=var_min,vmax=var_max)
	im1 = ax1.contourf(uveldataset.isel(tile=1),n_levels,vmin=var_min,vmax=var_max)
	im2 = ax2.contourf(uveldataset.isel(tile=2),n_levels,vmin=var_min,vmax=var_max)	
	im3 = ax3.contourf(uveldataset.isel(tile=3),n_levels,vmin=var_min,vmax=var_max)	
	im4 = ax4.contourf(uveldataset.isel(tile=4),n_levels,vmin=var_min,vmax=var_max)	
	im5 = ax5.contourf(uveldataset.isel(tile=5),n_levels,vmin=var_min,vmax=var_max)
	im6 = ax6.contourf(uveldataset.isel(tile=6),n_levels,vmin=var_min,vmax=var_max)	

	# reorient flipped grid tiles..	
	im7 = ax7.contourf(np.flip(vveldataset.isel(tile=7).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im8 = ax8.contourf(np.flip(vveldataset.isel(tile=8).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im9 = ax9.contourf(np.flip(vveldataset.isel(tile=9).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im10 = ax10.contourf(np.flip(vveldataset.isel(tile=10).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im11 = ax11.contourf(np.flip(vveldataset.isel(tile=11).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im12 = ax12.contourf(np.flip(vveldataset.isel(tile=12).T,axis=0),n_levels,vmin=var_min,vmax=var_max)

	if label_tiles == True:
		ax0.set_title("tile 0")
		ax1.set_title("tile 1")
		ax2.set_title("tile 2")
		ax3.set_title("tile 3")
		ax4.set_title("tile 4")
		ax5.set_title("tile 5")
		ax6.set_title("tile 6")
		ax7.set_title("tile 7")
		ax8.set_title("tile 8")
		ax9.set_title("tile 9")
		ax10.set_title("tile 10")
		ax11.set_title("tile 11")
		ax12.set_title("tile 12")

	if ticks_on == False:
		ax0.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax1.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax2.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax3.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax4.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax5.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax6.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax7.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax8.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax9.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax10.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)	
		ax11.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax12.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)

	fig.subplots_adjust(right=0.8)
	cb_ax = fig.add_axes([0.83, 0.12, 0.02, 0.75])
	plt.subplots_adjust(top=0.93)
	# I chose tile 8 since I don't know how to manually adjust the colorbar...
	cbar = fig.colorbar(im1,cax=cb_ax)
	cbar.ax.get_yaxis().labelpad = 18
	cbar.ax.tick_params(labelsize=15) 
	if title != None:
		plt.suptitle(title,fontsize=35)
	if cbar_title != None:
		cbar.set_label(cbar_title,fontsize=25,rotation=270)
	return plt 


def world_north_vel_plot(uveldataset, vveldataset, label_tiles=True,ticks_on=False):
	"""
	# A function to all plot ECCO grid tiles simulateously
	# data should be entered as an xarray dataarray, not a dataset
	# data should have dimensions (tile, i, j) = (13,i,j)
	# return a matplotlib plot object..

	# need to enter variable with timestep and k-level already selected like
	# etc: RHOAnoma_ds.RHOAnoma.isel(time=1,k=1)
	"""


	#tmp_plt = dataset

	# normalize colorscale across all grid 
	var_min = min(uveldataset.isel(tile=slice(7,13)).min(), vveldataset.isel(tile=slice(0,7)).min())
	var_max = max(uveldataset.isel(tile=slice(7,13)).max(), vveldataset.isel(tile=slice(0,7)).max())

	if var_min == var_max:
		var_min = None
		var_max = None

	# initialize figure and gridspec
	fig = plt.figure(figsize=(15,15))
	gs = gridspec.GridSpec(nrows=4,ncols=4,hspace=0.02,wspace=0.05)
	n_levels = 20

	# Eastern Atlantic Ocean tiles
	ax0 = fig.add_subplot(gs[3,1])
	ax1 = fig.add_subplot(gs[2,1])
	ax2 = fig.add_subplot(gs[1,1])

	# Indian Ocean tiles
	ax3 = fig.add_subplot(gs[3,2])
	ax4 = fig.add_subplot(gs[2,2])
	ax5 = fig.add_subplot(gs[1,2])

	# Arctic Ocean tile
	ax6 = fig.add_subplot(gs[0,2])

	# Pacific Ocean Tiles
	ax7 = fig.add_subplot(gs[1,3])
	ax8 = fig.add_subplot(gs[2,3])
	ax9 = fig.add_subplot(gs[3,3])

	# Western Atlantic Ocean tiles
	ax10 = fig.add_subplot(gs[1,0])
	ax11 = fig.add_subplot(gs[2,0])
	ax12 = fig.add_subplot(gs[3,0])



	im0 = ax0.contourf(vveldataset.isel(tile=0),n_levels,vmin=var_min,vmax=var_max)
	im1 = ax1.contourf(vveldataset.isel(tile=1),n_levels,vmin=var_min,vmax=var_max)
	im2 = ax2.contourf(vveldataset.isel(tile=2),n_levels,vmin=var_min,vmax=var_max)	
	im3 = ax3.contourf(vveldataset.isel(tile=3),n_levels,vmin=var_min,vmax=var_max)	
	im4 = ax4.contourf(vveldataset.isel(tile=4),n_levels,vmin=var_min,vmax=var_max)	
	im5 = ax5.contourf(vveldataset.isel(tile=5),n_levels,vmin=var_min,vmax=var_max)
	im6 = ax6.contourf(vveldataset.isel(tile=6),n_levels,vmin=var_min,vmax=var_max)	

	# reorient flipped grid tiles..	
	im7 = ax7.contourf(np.flip(-1*uveldataset.isel(tile=7).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im8 = ax8.contourf(np.flip(-1*uveldataset.isel(tile=8).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im9 = ax9.contourf(np.flip(-1*uveldataset.isel(tile=9).T,axis=0),n_levels,vmin=var_min,vmax=var_max)	
	im10 = ax10.contourf(np.flip(-1*uveldataset.isel(tile=10).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im11 = ax11.contourf(np.flip(-1*uveldataset.isel(tile=11).T,axis=0),n_levels,vmin=var_min,vmax=var_max)
	im12 = ax12.contourf(np.flip(-1*uveldataset.isel(tile=12).T,axis=0),n_levels,vmin=var_min,vmax=var_max)

	if label_tiles == True:
		ax0.set_title("tile 0")
		ax1.set_title("tile 1")
		ax2.set_title("tile 2")
		ax3.set_title("tile 3")
		ax4.set_title("tile 4")
		ax5.set_title("tile 5")
		ax6.set_title("tile 6")
		ax7.set_title("tile 7")
		ax8.set_title("tile 8")
		ax9.set_title("tile 9")
		ax10.set_title("tile 10")
		ax11.set_title("tile 11")
		ax12.set_title("tile 12")

	if ticks_on == False:
		ax0.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax1.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax2.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax3.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax4.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax5.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax6.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax7.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax8.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax9.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax10.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)	
		ax11.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)
		ax12.tick_params(tick1On=False,tick2On=False,labelleft=False,labelbottom=False)

	fig.subplots_adjust(right=0.8)
	cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])

	# I chose tile 8 since I don't know how to manually adjust the colorbar...
	cbar = fig.colorbar(im12,cax=cb_ax)
	fig.suptitle("meridional velocity plot", fontsize=16)

	# return the plot object..
	return plt 




def overturning_output_plots(depth_integrated_pdens_transport_latx, depth_integrated_pdens_transport_laty, 
							 time_slice,
							 basin_name):

	min_lat = 0
	max_lat = 178
	d_levels = np.arange(0,50)

	pvmin = -2e7
	pvmax = 2e7

	plt.figure(figsize=(20,10))
	#vmin = depth_integrated_pdens_transport.min()
	#vmax = depth_integrated_pdens_transport.max()
	plt.contourf(depth_integrated_pdens_transport_laty.lat[min_lat:max_lat],
	             depth_integrated_pdens_transport_laty.pot_rho[1:],
	             -1*depth_integrated_pdens_transport_laty.mean(dim='time')[1:,min_lat:max_lat]-1*depth_integrated_pdens_transport_latx.mean(dim='time')[1:,min_lat:max_lat],
	             30,
	             cmap='bwr',
	            vmin=pvmin,
	            vmax=pvmax)
	cbar = plt.colorbar()
	cbar.set_label("Transport (Sv)",rotation=270)
	plt.title("Meridional overturning results with interpolation",fontname='times new roman',fontsize=24)
	plt.xlabel("Latitude (deg)",fontname='Times New Roman',fontsize=16)
	plt.ylabel("$\sigma_{2}$ (kg/m^3)",fontname='Times New Roman',fontsize=16)
	cbar.set_label("Transport (Sv)",fontname='Times New Roman',fontsize=16,rotation=270)
	plt.grid()
	plt.gca().invert_yaxis()
	plt.savefig("./figures/"+basin_name+"_overturning_with_interp"+str(time_slice[0])+"to"+str(time_slice[-1])+".png")
	plt.close()




	