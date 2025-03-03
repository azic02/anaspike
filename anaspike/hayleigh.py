#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:15:55 2020

@author: haleigh
"""

# import matplotlib
# matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
#from mustela.datasets import Dataset, descriptions

from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.special import jv
from scipy.ndimage import map_coordinates
''' Code from mustela_bh.autocorr_wavelength, just moved the functions to their own script'''

## Fit functions
def gauss(x,a,sd):
	return a*np.exp(-x**2/(2*sd**2))

def gabor(x,sd,alpha):
	return np.exp(-x**2/(2*sd**2))*np.cos(alpha*x)

def expcos(x,var,alpha):
	return np.exp(-abs(x)/(2*var))*np.cos(alpha*x)

def cauchy(x,gamma,alpha):
	return gamma**2/(x**2+gamma**2)*np.cos(alpha*x)

def damped_bessel(x,var,alpha):
	return np.exp(-abs(x)/2./var)*jv(0,alpha*x)

def linfct(x,a,b):
	return a*x+b


def average_over_interp2d(array,conversion_factor_to_mm,nangles):
	"""calculates radial average of interpolated copy of array
	nangles : number of angles on which to interpolate"""
	if array.ndim>2:
		ys,xs = [],[]
		for frame in array:
			y,x = average_over_interp2d(frame,conversion_factor_to_mm,nangles)
			ys.append(y)
			xs.append(x)
		return np.array(ys),np.array(xs)
	else:
		h,w = array.shape
		xnew = np.linspace(-(w//2),w//2,w,endpoint=True)
		ynew = 0*xnew
		coords_new = np.vstack([xnew,ynew])
		angles = np.linspace(0,2*np.pi,nangles)
		spectrum = []
		for phi in angles:
			rotm = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
			this_coord = np.dot(rotm,coords_new)
			## coordinate origin is at upper left corner for this function!
			spectrum.append( map_coordinates(array,np.vstack([this_coord[1,:],\
				this_coord[0,:]])+h//2,order=1) )
		return np.nanmean(np.array(spectrum),axis=0), xnew*conversion_factor_to_mm


def get_autocorr(pattern,rough_patch_size,norm):
	"""calculates autocorrelation of array pattern"""
	hs,ws = pattern.shape
	max_lag = rough_patch_size
	#autocorr = np.zeros((max_lag+1,max_lag+1))
	#for xlag in range(max_lag):
		#for ylag in range(max_lag):
			#autocorr[ylag,xlag] =\
			 # np.nansum(pattern[:hs-ylag,:ws-xlag]*np.conjugate(pattern[ylag:,xlag:]))/pattern[:hs-ylag,:ws-xlag].size
	
	## wiener-khinchin
	
	#frm bettina code snippet
	norm=hs*ws
	max_lag=hs//2 # min([hs//2,ws//2]) should use this probably when pattern is not square
	#first normalize pattern to 0 mean and SD 1
	pattern_norm=pattern-np.nanmean(pattern)
	pattern_norm /= np.nanstd(pattern_norm)
	pattern_norm[np.logical_not(np.isfinite(pattern_norm))]=0.0
	
	
	fft_spectrum = abs(np.fft.fft2(pattern_norm, s=(2*hs,2*ws)))**2
	autocorr = np.fft.ifft2( fft_spectrum )/norm#[:hs,:ws]
	autocorr = np.fft.fftshift(autocorr)[hs-max_lag:hs+max_lag+1,ws-max_lag:ws+max_lag+1]
	return np.real(autocorr)

def eventAutocorrFitWavelength(events,roi,ump=None,dataset=None,makeFig=True,saveFig=None,clipROI=True):
	''' computes autocorrelation for events and then attempts to fit wavelength (using bessel, gabor, etc)
	
		Inputs:
			events- event frames, filtered, roi masked
			roi- roi for event frames (including blood vessel mask if desired)
            ump= microns per pixel 
			dataset- used to get umperpix (if ump is not provided)
			makeFig- Make figure
			saveFig- filepath to save figure to
		
		Returns: 
			Autocorr-autocorrelations for each event (2d)
			spectrum- radial average of autocorrelations (1d)
			distance- distance in microns from center
			w_bessel-Bessel fit wavelength
			w_gabor- Gabor fit wavelength
			w_cauchy- Cauchy fit wavelength
			area_peak_locs- Location of first trough ([:,0]), and first peak ([:,1])
			area_peak_vals-Correlation values at trough and peak
	
	
	'''
	
	if ump is None:
		resolution=descriptions.get_umperpix(dataset)
	else:
		resolution=ump    
	#local_neighbourhood_size = int(1000/(resolution*2))		## radius of local neighbourhood around seed point in um
	rough_patch_size = int(1500/(resolution))
	#disk_neighbourhood = disk(rough_patch_size)>0
	idy,idx = np.where(roi)
	norm = np.sum(roi[min(idy):max(idy),min(idx):max(idx)])
	events[np.isnan(events)]=0
	nframes=events.shape[0]

	# take window that contains entire ROI but cuts away extra space of frame to minimize influence of ROI size
	if clipROI:
		yroi,xroi = np.where(roi)
		top_roi,bot_roi = np.nanmin(yroi),np.nanmax(yroi)
		left_roi,right_roi = np.nanmin(xroi),np.nanmax(xroi)
		center_roi_x = (right_roi + left_roi)//2
		center_roi_y = (top_roi + bot_roi)//2
		delta_x = right_roi - left_roi
		delta_y = bot_roi - top_roi
		delta = np.min([delta_x,delta_y])
		delta -= delta%2 # make sure that window length is even, easier to handle averaging I think
		events= events[:,center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]
		roi_mask_clipped = roi[center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]


	#calculate autocorrelation for each event
	print('Calculating autocorrelation')
	autocorr = []
	for frame in events:
		autocorr.append(get_autocorr(frame,rough_patch_size,norm=norm) )
	autocorr = np.array(autocorr)
	
	spectrum,distance = average_over_interp2d(autocorr,resolution/1000,360)
	distance = distance[0,:]
	
	m = np.ones(distance.shape[0])
	m[distance>0.6]=0.3
	m[0] = 2
	print('Fitting wavelength')
	area_peak_locs,area_peak_vals = [],[]
	w_gabor,w_bessel,w_cauchy = [],[],[]
	for i in range(nframes):
		## Spline interpolation
		# print("CHECK",distance.shape, spectrum[i,:].shape,\
		# np.sum(np.isfinite(distance)),np.sum(np.isfinite(spectrum[i,:])))
		tck = interpolate.splrep(distance, spectrum[i,:], k=3)#s=50  w=1./np.abs(spectrum)*m,
		distance2 = np.copy(distance)
		distance2[1:-1] += np.random.randn(len(distance)-2)*0.001
		distance2 = np.sort(distance2)
		spectrum2 = interpolate.splev(distance2, tck)

		## Location of first peak and trough of corr fct
		derivative = interpolate.splev(distance2,tck,der=1)
		tck_der = interpolate.splrep(distance, derivative, w=1./np.abs(spectrum[i,:]),\
				 s=50, k=3)
		peak_locs = interpolate.sproot(tck_der)
		# print("peak_locs",peak_locs)
		if len(peak_locs)>0:
			peak_vals = interpolate.splev(peak_locs,tck)
			peak_locs = np.abs(peak_locs)
			peak_origin = np.nanargmin(peak_locs)

			if (peak_origin+1)<len(peak_locs):
				low_right = peak_locs[peak_origin+1]
				val_right = peak_vals[low_right==peak_locs][0]
			else:
				low_right = np.nan
				val_right = np.nan
			if (peak_origin-1)>0:
				low_left = peak_locs[peak_origin-1]
				val_left = peak_vals[low_left==peak_locs][0]
			else:
				low_left = np.nan
				val_left = np.nan
			first_low = np.nanmean([low_right,low_left])
			low_val = np.nanmean([val_right,val_left])

			if (peak_origin+2)<len(peak_locs):
				peak_right = peak_locs[peak_origin+2]
				val_right = peak_vals[peak_right==peak_locs][0]
			else:
				peak_right = np.nan
				val_right = np.nan
			if (peak_origin-2)>0:
				peak_left = peak_locs[peak_origin-2]
				val_left = peak_vals[peak_left==peak_locs][0]
			else:
				peak_left = np.nan
				val_left = np.nan
			first_peak = np.nanmean([peak_right,peak_left])
			peak_val = np.nanmean([val_right,val_left])

			area_peak_locs.append(np.array([first_low,first_peak]))
			area_peak_vals.append(np.array([low_val,peak_val]))

		else:
			peak_vals = [np.nan]
			peak_locs = [np.nan]

			area_peak_locs.append(np.array([np.nan,np.nan]))
			area_peak_vals.append(np.array([np.nan,np.nan]))

			
		## FIT
		fit_vals = np.isfinite(spectrum[i,:])#*(spectrum>0)*(distance<0.4)
		init_var = np.array([0.006,0.06,0.1,0.25,0.3])#np.array([0.06])#
		init_k = np.array([5,7,10])#np.array([10])#
		pcov = np.array([[1,1],[1,1]])
		popt = np.array([np.nan,np.nan])
		try:
			for ivar in init_var:
				for ik in init_k:
					ipopt,ipcov = curve_fit( gabor,distance[fit_vals],spectrum[i,fit_vals],
								p0=[ivar,ik])
					if np.mean(np.sqrt(np.diag(ipcov)))<np.mean(np.sqrt(np.diag(pcov))):
						pcov = ipcov
						popt = ipopt
								
								
								
		except:
			pass
		perr = np.sqrt(np.diag(pcov))
		w_gabor.append(np.array([2*np.pi/abs(popt[1]),\
					2*np.pi/abs(popt[1])**2*perr[1]]))
				
		initvar = 0.15
		initk = 10.
		pcov3 = np.array([[1,1],[1,1]])
		popt3 = np.array([np.nan,np.nan])
		try:
			popt3,pcov3 = curve_fit( damped_bessel,distance[fit_vals],spectrum[i,fit_vals],\
						p0=[initvar,initk])
		except:
			pass
		perr3 = np.sqrt(np.diag(pcov3))
		w_bessel.append(np.array([abs(popt3[1]),perr3[1]]))

		initgamma = 0.2
		initk = 7.
		pcov4 = np.array([[1,1],[1,1]])
		popt4 = np.array([np.nan,np.nan])
		try:
			popt4,pcov4 = curve_fit( cauchy,distance[fit_vals],\
						spectrum[i,fit_vals],p0=[initgamma,initk])
		except:
			pass
		perr4 = np.sqrt(np.diag(pcov4))
		w_cauchy.append(np.array([2*np.pi/abs(popt4[1]),\
					2*np.pi/abs(popt4[1])**2*perr4[1]]))
			
	area_peak_locs = np.array(area_peak_locs)
	area_peak_vals = np.array(area_peak_vals)
	w_gabor = np.array(w_gabor)
	w_bessel = np.array(w_bessel)
	w_cauchy = np.array(w_cauchy)
		
	print(peak_locs)	
	if makeFig:
			## Plot mean corr fcts
			fig = plt.figure(figsize=(5*6,5))
			ax = fig.add_subplot(151)
			ax.set_title("Avg +/- SD of all events")
			ax.plot([-1.5,1.5],[0,0],'--',c='gray')
			ax.plot(distance,np.nanmean(spectrum,axis=0),'-k',label='Mean,all',mfc="None")
			ax.plot(distance,np.nanmean(spectrum,axis=0)+np.nanstd(spectrum,axis=0),\
					'--k',label='Mean+/-SD',mfc="None")
			ax.plot(distance,np.nanmean(spectrum,axis=0)-np.nanstd(spectrum,axis=0),\
					'--k',mfc="None")
			ax.legend(loc="best")
			ax.set_xlabel('Distance (mm)')
			ax.set_ylabel('Correlation coeff.')
	
			ax = fig.add_subplot(152)
			ax.set_title("10 examples")
			ax.plot([-1.5,1.5],[0,0],'--',c='gray')
			
			if spectrum.shape[0]<10:
				sizeExamples=spectrum.shape[0]
			else:
			   sizeExamples=10
			idx = np.random.choice(np.arange(spectrum.shape[0]),size=sizeExamples,replace=False)
			ax.plot(distance,spectrum[idx,:].T,'-',c="k",alpha=0.5)
			ax.set_xlabel('Distance (mm)')
			ax.set_ylabel('Correlation coeff.')
	
			## plot exemplary fits for last activity pattern
			ax = fig.add_subplot(153)
			ax.set_title("Fits to exemplary corr fct")
			ax.plot(distance2,spectrum2,'--',c="k",label='spline',zorder=10)
			ax.plot(distance[fit_vals],gabor(distance[fit_vals],*popt),'-b',label='Gabor')
			ax.plot(distance[fit_vals],damped_bessel(distance[fit_vals],*popt3),'-r',label='Bessel')
			ax.plot(distance[fit_vals],cauchy(distance[fit_vals],*popt4),'-g',label='Cauchy')
			ax.plot(peak_locs,peak_vals,'sm',label="Extrema",zorder=20)
			ax.legend(loc='best')
			ax.set_xlabel('Distance (mm)')
			ax.set_ylabel('Correlation coeff.')
			ax.set_xlim(-0.1,1.5)
#    		ax.set_ylim(-0.2,1.05)
			ax.set_ylim(np.min(spectrum2),np.max(spectrum2))
			
					## plot corr fct of individual activity patterns
			ax = fig.add_subplot(154)
			ax.set_title("1d indiv corr")
			ax.plot(distance,spectrum.T,'-',c="k",alpha=0.3)
			ax.set_xlabel('Distance (mm)')
			ax.set_ylabel('Correlation coeff.')
			ax.set_xlim(0,1.5)
#    		ax.set_ylim(-0.2,1.05)
	
			ax = fig.add_subplot(155)
			ax.set_title("2d avg corr")
			#ax.imshow(autocorr[-1,:,:],interpolation='nearest',cmap='RdBu_r',vmin=-np.percentile(autocorr,99),vmax=np.percentile(autocorr,99))
			ax.imshow(autocorr[-1,:,:],interpolation='nearest',cmap='RdBu_r',vmin=-0.3,vmax=0.3)
			ax.set_xticks(np.linspace(0,rough_patch_size*2,7))
			ax.set_xticklabels(np.arange(0,3.1,0.5))
			ax.set_xlabel("Distance (mm)")
			ax.set_yticks([])
	if saveFig is not None: 
		ferret=dataset.ferretNumber
		fig.savefig(saveFig + 'F' + (f'{ferret:04}') + '_event_autocorr_wavelength.png')
		
	return autocorr,spectrum,distance,w_bessel,w_gabor,w_cauchy,area_peak_locs,area_peak_vals

def compute_ft_modularity(event_frames,roi_mask,wavelength,resolution):
	"""
	compute modularity based on Fourier spectrum as F1/F0
	input:
	event_frames: not bandpass filtered 3d stack of 2d array of activity patterns
	roi_mask: boolean array indicating which pixels do not belong to background (BV and ROI)
	wavelength: wavelength at whcih to compute f1 (is currently being overwritten) -- list for each event
	resolution: transformation factor for pixel to um
	output: 
	"""


	# take window that contains entire ROI but cuts away extra space of frame
	# to minimize influence of ROI size
	yroi,xroi = np.where(roi_mask)
	top_roi,bot_roi = np.nanmin(yroi),np.nanmax(yroi)
	left_roi,right_roi = np.nanmin(xroi),np.nanmax(xroi)
	center_roi_x = (right_roi + left_roi)//2
	center_roi_y = (top_roi + bot_roi)//2
	delta_x = right_roi - left_roi
	delta_y = bot_roi - top_roi
	delta = np.min([delta_x,delta_y])
	delta -= delta%2 # make sure that window length is even, easier to handle averaging I think
	event_clipped = event_frames[:,center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]
	roi_mask_clipped = roi_mask[center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]

	# set NaN values to zero before computing Fourier spectrum
	# TODO: set them to mean value instead to minimize boundaries at BVs
	event_clipped_avg = np.nanmean(event_clipped,axis=(1,2))
	event_clipped[np.logical_not(np.isfinite(event_clipped))] = 0.0
	event_clipped[:,np.logical_not(roi_mask_clipped)] = event_clipped_avg[:,None]
	event_clip=[]
	for ievt, event in enumerate(event_clipped): 
		event_clip.append(event-event_clipped_avg[ievt])
	event_clipped=np.array(event_clip)
	event_spectrum = np.fft.fftshift(np.abs(np.fft.fft2(event_clipped,axes=(1,2))),axes=(1,2))
	spectrum,frequency = average_over_interp2d(event_spectrum,1./(delta * resolution/1000),360)
	frequency_inv_mm = frequency[0,:]

	## compute modularity for control which is taken as mean over all or 200 (min) responses
	num_events = event_frames.shape[0]
	min_num_events = np.min([200,num_events])
	idx = np.random.choice(np.arange(num_events),size=min_num_events,replace=False)
	event_mean = np.nanmean(event_frames[idx,:,:],axis=0)
	event_mean = event_mean[center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]
	event_mean[np.logical_not(np.isfinite(event_mean))] = 0.0
	event_mean_avg = np.nanmean(event_mean)
	event_mean[np.logical_not(roi_mask_clipped)] = event_mean_avg
	event_spectrum_mean = np.fft.fftshift(np.abs(np.fft.fft2(event_mean,axes=(0,1))),axes=(0,1))
	spectrum_mean,_ = average_over_interp2d(event_spectrum_mean,1./(delta * resolution/1000),360)

	# average frequency (=1/wavelength) per age bin over all animals/areas
	pd_bin=0  #For animals before eye opening -- see BH code mustela_bh.pfc.autocorr_wavelength
	frequency_range = {0: [1/0.88,1./0.75], 1: [1/0.77,1/0.69], 2: [1/0.72,1/0.64], 3: [1/0.72,1/0.64]}
	wavelength_range = {0: [0.75,0.88], 1: [0.69,0.77], 2: [0.64,0.72], 3: [0.64,0.72]}
	wavelength = np.clip(wavelength[:,1],*wavelength_range[pd_bin])
	wavelength[np.logical_not(np.isfinite(wavelength))] = np.nanmean(wavelength_range[pd_bin])

	# compute average of spontaneous events and average frame
	f0 = spectrum[:,delta//2]
	f0_m = spectrum_mean[delta//2]

	# initialize list of f1 value per event
	f1,f1_interp = [],[]
	f1_m,f1_interp_m = [],[]

	# spline interpolation of 1d spectrum of average frame
	tck = interpolate.splrep(frequency_inv_mm, spectrum_mean, k=3)#s=50  w=1./np.abs(spectrum)*m,
	#f1_range = interpolate.splev(frequency_range[pd_bin], tck)
	f1_range = interpolate.splev(frequency_range[pd_bin], tck)
	#f1_interp_m = np.nanmean(f1_range)


	for i in range(num_events):
		## Spline interpolation of 1d spectrum of indiv events
		tck = interpolate.splrep(frequency_inv_mm, spectrum[i,:], k=3)#s=50  w=1./np.abs(spectrum)*m,
		f1_range = interpolate.splev(frequency_range[pd_bin], tck)
		f1_interp.append(np.nanmean(f1_range))

		wavelength_band = np.abs(frequency_inv_mm - 1./wavelength[i])<0.25
		f1.append(np.nanmean(spectrum[i,wavelength_band]))
		f1_m.append(np.nanmean(spectrum_mean[wavelength_band]))

	
		
	modularity = f1/f0
	modularity_interp = f1_interp/f0
	modularity_m = np.nanmean(f1_m)/f0_m
	modularity_interp_m = f1_interp_m/f0_m



	# plot intermediate results
	# plot activity, 2d spectrum, 1d spectrum, wavelength location
	if False:
# 		import matplotlib
# 		matplotlib.use("agg")
# 		import matplotlib.pyplot as plt

		nrow,ncol = 5,3
		fig = plt.figure(figsize=(6*ncol,5*nrow))

		ax = fig.add_subplot(nrow,ncol,1)
		#ax.set_title("Mean,Mod={:.2f},ModI={:.2f}".format(modularity_m,modularity_interp_m))
		im=ax.imshow(event_mean,interpolation="nearest",cmap="binary_r")
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,2)
		h,w = event_spectrum_mean.shape
		im = ax.imshow(event_spectrum_mean[h//2-20:h//2+20,w//2-20:w//2+20],interpolation="nearest",cmap="RdBu_r",vmax=4000)
		plt.colorbar(im,ax=ax)
		ax = fig.add_subplot(nrow,ncol,3)
		ax.plot(frequency_inv_mm,spectrum_mean,'-k')
		ax.plot([frequency_range[pd_bin][0],]*2,[0,1000],"-r")
		ax.plot([frequency_range[pd_bin][1],]*2,[0,1000],"-m")
		ax.set_xlim(-5,5)

		tmp = event_frames[:,center_roi_y-delta//2:center_roi_y+delta//2,center_roi_x-delta//2:center_roi_x+delta//2]
		MU = np.nanmean(event_clipped)
		SD = np.nanstd(event_clipped)
		for i in range(1,nrow):
			ax = fig.add_subplot(nrow,ncol,1+i*ncol)
			#ax.set_title("Mod={:0.2f},ModI={:0.2f},f0={:0.2f},f1={:0.2f}".format(modularity[i],modularity_interp[i],f0[i],f1[i]))
			im=ax.imshow(tmp[i,:,:],interpolation="nearest",cmap="binary_r")
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(nrow,ncol,2+i*ncol)
			h,w = event_spectrum[i,:,:].shape
			im = ax.imshow(event_spectrum[i,h//2-20:h//2+20,w//2-20:w//2+20],interpolation="nearest",cmap="RdBu_r",vmax=4000)
			plt.colorbar(im,ax=ax)
			ax = fig.add_subplot(nrow,ncol,3+i*ncol)
			ax.plot(frequency_inv_mm,spectrum[i,:],'-k')
			ax.plot([1./wavelength[i]]*2,[0,2000],'-c')
			ax.plot([frequency_range[pd_bin][0],]*2,[0,1000],"-r")
			ax.plot([frequency_range[pd_bin][1],]*2,[0,1000],"-m")
			ax.set_xlim(-5,5)
			# ax.set_ylim(-0.2,1.02)

		#plt.close(fig)

	labels = ["modularity_ft","modularity_interp","modularity_m","modularity_interp_m",\
				"ft_spectrum","frequency","f1","f1_interp","f0",'f0_m']
	values = [modularity,modularity_interp,modularity_m,modularity_interp_m,spectrum,frequency_inv_mm,f1,f1_m,f0,f0_m]
	
	return labels,values
