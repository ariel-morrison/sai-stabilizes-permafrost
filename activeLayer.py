#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:52:30 2024

@author: arielmor
"""

def activeLayerDepth():
    import numpy as np
    from tippingPoints import findPermafrost
    from plottingFunctions import make_maps, get_colormap
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(21)
    
    bwr = cm.get_cmap('RdBu_r', (21))
    newcolors = bwr(np.linspace(0, 1, 256))
    newcolors[122:134, :] = np.array([1, 1, 1, 1])
    newcolors[255:, :] = np.array([0, 0, 0, 1])
    newcolors[:2, :] = np.array([0.1,0.8,0.1,1.])
    altCmap = ListedColormap(newcolors)
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    
    ## -- read active layer depth -- ##
    altAnnMeanCONTROL,altmaxMonthlyCONTROL,altmaxAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912')
    altAnnMeanFEEDBACK,altmaxMonthlyFeedback,altmaxAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    
    ###### -- control -- ######
    altCONTROL_mean_20352044 = {}
    altCONTROL_mean_20602069 = {}
    for i in range(len(ens)):
        altCONTROL_mean_20352044[ens[i]] = altmaxAnnCONTROL[ens[i]][20:30,:,:].mean(dim='year', skipna=True)
        altCONTROL_mean_20602069[ens[i]] = altmaxAnnCONTROL[ens[i]][-10:,:,:].mean(dim='year', skipna=True)

    altCONTROL_ensmean_20352044 = np.nanmean(np.stack((altCONTROL_mean_20352044.values())),axis=0) #make_ensemble_mean_timeseries(altCONTROL_mean_20352044,10) 
    altCONTROL_ensmean_20602069 = np.nanmean(np.stack((altCONTROL_mean_20602069.values())),axis=0) #make_ensemble_mean_timeseries(altCONTROL_mean_20602069,10) 
    altCONTROL_diff = (altCONTROL_ensmean_20602069 - altCONTROL_ensmean_20352044)#.to_numpy()
    print(np.nanmin(altCONTROL_diff), np.nanmax(altCONTROL_diff))
    
    diffCONTROLEns = {}
    for i in range(len(ens)):
        diffCONTROLEns[ens[i]] = (altCONTROL_mean_20602069[ens[i]] - altCONTROL_mean_20352044[ens[i]]).to_numpy()
    
    
    ''' mask cells that thawed completely or gained permafrost'''
    altCONTROL_diff[
        (np.isnan(altCONTROL_ensmean_20602069)) & 
        (~np.isnan(altCONTROL_ensmean_20352044))] = 100
    
    altCONTROL_diff[
        (~np.isnan(altCONTROL_ensmean_20602069)) & 
        (np.isnan(altCONTROL_ensmean_20352044))] = -100
    

    ###### -- feedback -- ######
    altFEEDBACK_mean_20352044 = {}
    altFEEDBACK_mean_20602069= {}
    for i in range(len(ens)):
        altFEEDBACK_mean_20352044[ens[i]] = altmaxAnnFEEDBACK[ens[i]][:10,:,:].mean(dim='year', skipna=True)
        altFEEDBACK_mean_20602069[ens[i]] = altmaxAnnFEEDBACK[ens[i]][-10:,:,:].mean(dim='year', skipna=True)
        
    altFEEDBACK_ensmean_20352044 = np.nanmean(np.stack((altFEEDBACK_mean_20352044.values())),axis=0) #make_ensemble_mean_timeseries(altFEEDBACK_mean_20352044,10)
    altFEEDBACK_ensmean_20602069 = np.nanmean(np.stack((altFEEDBACK_mean_20602069.values())),axis=0)# make_ensemble_mean_timeseries(altFEEDBACK_mean_20602069,10)
    altFEEDBACK_diff = (altFEEDBACK_ensmean_20602069 - altFEEDBACK_ensmean_20352044)#.to_numpy()
    print(np.nanmin(altFEEDBACK_diff), np.nanmax(altFEEDBACK_diff))
    
    ''' mask cells that thawed completely or gained pfrost'''
    altFEEDBACK_diff[
        (np.isnan(altFEEDBACK_ensmean_20602069)) & 
        (~np.isnan(altFEEDBACK_ensmean_20352044))] = 100
    
    altFEEDBACK_diff[
        (~np.isnan(altFEEDBACK_ensmean_20602069)) & 
        (np.isnan(altFEEDBACK_ensmean_20352044))] = -100


    ###### -- Make figures - Fig. 2 -- ######
    ## A) control 2035-2044
    longitude = lon
    fig, ax = make_maps(altCONTROL_ensmean_20352044,lat,longitude,0,3.5,21,reds,'ALTMAX (m)',
                        'a) SSP2-4.5, 2035-2044','Fig2a_active_layer_depth_CONTROL_20352044','max',
                        False, False)
    del fig, ax, longitude
    
    ## B) feedback 2035-2044
    longitude = lon
    fig, ax = make_maps(altFEEDBACK_ensmean_20352044,lat,longitude,0,3.5,21,reds,'ALTMAX (m)',
                        'b) ARISE-SAI-1.5, 2035-2044','Fig2b_active_layer_depth_FEEDBACK_20352044','max',
                        False, False)
    del fig, ax, longitude
    
    ## C) control 2060-2069
    longitude = lon
    fig, ax = make_maps(altCONTROL_ensmean_20602069,lat,longitude,0,3.5,21,reds,'ALTMAX (m)',
                        'c) SSP2-4.5, 2060-2069','Fig2c_active_layer_depth_CONTROL_20602069','max',
                        False, False)
    del fig, ax, longitude
    
    ## D) feedback 2060-2069
    longitude = lon
    fig, ax = make_maps(altFEEDBACK_ensmean_20602069,lat,longitude,0,3.5,21,reds,'ALTMAX (m)',
                        'd) ARISE-SAI-1.5, 2060-2069','Fig2d_active_layer_depth_FEEDBACK_20602069','max',
                        False, False)
    del fig, ax, longitude
    
    ## E) control difference 2060-2069 minus 2035-2044
    longitude = lon
    fig, ax = make_maps(altCONTROL_diff,lat,longitude,-1,1,21,altCmap,'\u0394 ALTMAX (m)',
                        'e) SSP2-4.5 difference','Fig2e_active_layer_depth_CONTROL_difference_20602069_minus_20352044','both',
                        False, False)
    del fig, ax, longitude
    
    ## F) feedback difference 2060-2069 minus 2035-2044    
    longitude = lon
    fig, ax = make_maps(altFEEDBACK_diff,lat,longitude,-1,1,21,altCmap,'\u0394 ALTMAX (m)',
                        'f) ARISE-SAI-1.5 difference','Fig2f_active_layer_depth_FEEDBACK_difference_20602069_minus_20352044','both',
                        False, False)
    del fig, ax, longitude
    
    
    return