#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 14:03:08 2022

@author: Ariel L. Morrison

"Tipping points" in the climate system are thresholds that, once
crossed, lead to sustained and irreversible warming. A tipping point
can also be a threshold beyond which the initial conditions of a system
cannot be recovered. Permafrost has a 'tipping point' threshold of ~0degC
(the thawing temperature) because it's hard to refreeze permafrost once
it has been thawed. Talik formation is a tipping point because it promotes
increased soil respiration and soil carbon loss (i.e., makes soil carbon
more accessible to decomposition and respiration.)
"""
     
def talikFormation(lat_lon_series, N):
    ## N = number of years that soil is thawed (warmer than -0.5C (Parazoo et al., 2018))
    ## for talik formation, needs to be perenially thawed
    ## so N = number of years left in the simulation
     import numpy as np
     mask = np.convolve(np.greater(lat_lon_series,272.65),np.ones(N,dtype=int))>=N
     if mask.any():
         return mask.argmax() - N + 1
     else:
         return None
     

def readDataForTippingPoints(datadir, var, controlSim, simulation, timePeriod, lnd, warmPfrost):
    import xarray as xr
    import pandas as pd
    
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'

    myvar = {}
    myvarAnn = {}
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
        
    numEns = len(ens)
    print("reading " + str(var) + " for " + str(simulation))
    
    for i in range(numEns):
        ds = xr.open_dataset(datadir + '/' + str(simulation) + str(ens[i]) +
                             '.clm2.h0.' + str(var) + '.' + 
                             str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        lat = ds.lat[41:]
        lon = ds.lon
        
        ## only want top 20 layers because last 5 layers are always bedrock
        myvar[ens[i]] = ds[str(var)][:,:20,41:,:]
        ## get COLDEST monthly mean soil temp for each year ##
        ## this is for talik formation ##
        ## the coldest temp of the year needs to be > -0.5deg to be talik ##
        myvarAnn[ens[i]] = myvar[ens[i]].groupby('time.year').min(dim='time', skipna=True)
        
    return lat, lon, myvar, myvarAnn, ens

        
def findTippingPoint(datadir, figuredir):
    import numpy as np
    import xarray as xr
    from tippingPoints import readDataForTippingPoints, talikFormation, findPermafrost
    from peatlandPermafrostCalculations import getLandType
    from make_timeseries import make_ensemble_mean_timeseries
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import matplotlib.ticker as mticker
    from cartopy.util import add_cyclic_point
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.colors as mcolors
    from plottingFunctions import get_colormap, circleBoundary, mapsSubplotsDiff, mapsSubplots
    from matplotlib import colors as c
    import matplotlib as mpl
   
    circle = circleBoundary
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(27)
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    numEns = len(ens)
    
    ## landmask ##
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask
    ds.close()
    
    landMask = landmask.where(np.isnan(landmask))
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)    
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    
    ########################################################
    ####     Pfrost extent (active layer)     ####
    ########################################################
    altAnnMeanCONTROL,pfrostCONTROL,pfrostAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912')
    altAnnMeanFEEDBACK,pfrostFEEDBACK,pfrostAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    
    pfrostExtentCONTROL = {}
    pfrostExtentFEEDBACK = {}
    for numEns in range(len(ens)):
        '''ssp'''
        pfrostExtentCONTROL[ens[numEns]] = pfrostAnnCONTROL[ens[numEns]].copy()
        pfrostExtentCONTROL[ens[numEns]] = xr.where(pfrostAnnCONTROL[ens[numEns]][0,:,:].notnull(), 1, 0)
        '''default arise'''
        pfrostExtentFEEDBACK[ens[numEns]] = pfrostAnnFEEDBACK[ens[numEns]].copy()
        pfrostExtentFEEDBACK[ens[numEns]] = xr.where(pfrostAnnFEEDBACK[ens[numEns]][0,:,:].notnull(), 1, 0)
    
    
    #########################################################
    #### Total column soil temp - for talik formation
    #########################################################
    lat, lon, tsCONTROL, tsAnnCONTROL, ens = readDataForTippingPoints(
        'TSOI', False, False, 'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.', '201501-206912', 
                            True, warmPfrost)
    lat, lon, tsFEEDBACK, tsAnnFEEDBACK, ensFEEDBACK = readDataForTippingPoints(
        'TSOI', False, False, 'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912', 
                            True, warmPfrost)
    
    ## can only have talik where permafrost exists
    for ensNum in range(len(ens)):
        mask = np.repeat(pfrostCONTROL[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsCONTROL[ens[ensNum]] = xr.where(~np.isnan(mask), tsCONTROL[ens[ensNum]], np.nan)
        del mask
        mask = np.repeat(pfrostAnnCONTROL[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsAnnCONTROL[ens[ensNum]] = xr.where(~np.isnan(mask), tsAnnCONTROL[ens[ensNum]], np.nan)
        del mask
        
    
    for ensNum in range(len(ens)):
        mask = np.repeat(pfrostFEEDBACK[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsFEEDBACK[ens[ensNum]] = xr.where(~np.isnan(mask), tsFEEDBACK[ens[ensNum]], np.nan)
        del mask
        mask = np.repeat(pfrostAnnFEEDBACK[ens[ensNum]].values[:,np.newaxis,:,:],20,axis=1)
        tsAnnFEEDBACK[ens[ensNum]] = xr.where(~np.isnan(mask), tsAnnFEEDBACK[ens[ensNum]], np.nan)
        del mask
    
    
    tsAnnCONTROLmean  = np.nanmean(np.stack((tsAnnCONTROL.values())),axis=0) 
    tsAnnFEEDBACKmean = np.nanmean(np.stack((tsAnnFEEDBACK.values())),axis=0) 
    
    #### Talik formation timing
    ''' ------------------------------------------------------ '''
    ''' Find talik formation timing based on column soil temp  '''
    ''' ------------------------------------------------------ '''
    #########################################################
    ## CONTROL TALIK FORMATION
    #########################################################
    ds = xr.open_dataset(datadir + 
                               '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(20)
    ds.close()
    
    numEns = len(ens)
    talikAnnCONTROL = {}
    for numEns in range(len(ens)):
        print(numEns)
        talikAnnCONTROL[ens[numEns]] = np.empty((tsAnnCONTROLmean.shape[1]-1,len(lat),len(lon)))*np.nan
        for ilat in range(len(lat)):
            for ilon in range(len(lon)):
                for ilev in range(int(nbedrock[ilat,ilon])-1):
                # for ilev in range(tsAnnCONTROLmean.shape[1]-1):
                    talikAnnCONTROL[ens[numEns]][ilev,ilat,ilon] = talikFormation((tsAnnCONTROL[ens[numEns]][:,ilev+1,ilat,ilon]),N=2)
        talikAnnCONTROL[ens[numEns]] = np.nanmin(talikAnnCONTROL[ens[numEns]], axis=0)
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'wb') as fp:
        pickle.dump(talikAnnCONTROL, fp)
        print('Your dictionary has been saved successfully to file')
    
                
    talikAnnCONTROLmean = np.empty((tsAnnCONTROLmean.shape[1]-1,len(lat),len(lon)))*np.nan
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            for ilev in range(int(nbedrock[ilat,ilon])-1):
            # for ilev in range(tsAnnCONTROLmean.shape[1]-1):
                talikAnnCONTROLmean[ilev,ilat,ilon] = talikFormation((tsAnnCONTROLmean[:,ilev+1,ilat,ilon]),N=2)
    talikAnnCONTROLmean = np.nanmin(talikAnnCONTROLmean,axis=0)
                
    
    #########################################################
    ## FEEDBACK TALIK FORMATION
    #########################################################
    numEns = len(ens)
    talikAnnFEEDBACK = {}
    for numEns in range(len(ens)):
        print(numEns)
        tsAnnFEEDBACK_combined = np.concatenate((tsAnnCONTROL[ens[numEns]][:20,:,:,:],tsAnnFEEDBACK[ens[numEns]]),axis=0)
        talikAnnFEEDBACK[ens[numEns]] = np.empty((tsAnnFEEDBACKmean.shape[1]-1,len(lat),len(lon)))*np.nan
        for ilat in range(len(lat)):
            for ilon in range(len(lon)):
                for ilev in range(int(nbedrock[ilat,ilon])-1):
                # for ilev in range(tsAnnFEEDBACK[ensFEEDBACK[0]].shape[1]-1):
                    talikAnnFEEDBACK[ens[numEns]][ilev,ilat,ilon] = talikFormation((tsAnnFEEDBACK_combined[:,ilev+1,ilat,ilon]),N=2)
        del tsAnnFEEDBACK_combined
        talikAnnFEEDBACK[ensFEEDBACK[numEns]] = np.nanmin(talikAnnFEEDBACK[ens[numEns]], axis=0)
        
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'wb') as fp:
        pickle.dump(talikAnnFEEDBACK, fp)
        print('Your dictionary has been saved successfully to file')
                
    
    ## start from 2015 to see where talik already formed by 2035
    tsAnnFEEDBACK_combined_mean = np.concatenate((tsAnnCONTROLmean[:20,:,:,:],tsAnnFEEDBACKmean),axis=0)
    talikAnnFEEDBACKmean = (np.empty((tsAnnFEEDBACKmean.shape[1]-1,len(lat),len(lon))))*np.nan
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            for ilev in range(int(nbedrock[ilat,ilon])-1):
            # for ilev in range(tsAnnFEEDBACKmean.shape[1]-1):
                talikAnnFEEDBACKmean[ilev,ilat,ilon] = talikFormation((tsAnnFEEDBACK_combined_mean[:,ilev+1,ilat,ilon]),N=2)
    talikAnnFEEDBACKmean = np.nanmin(talikAnnFEEDBACKmean,axis=0)
    
    ## save as pickles - big files, takes a long time to rerun!
    np.save(datadir + '/talikAnnCONTROL_ens.npy', talikAnnCONTROL)
    np.save(datadir + '/talikAnnCONTROLmean.npy', talikAnnCONTROLmean)
    np.save(datadir + '/talikAnnFEEDBACK_ens.npy', talikAnnFEEDBACK)
    np.save(datadir + '/talikAnnFEEDBACKmean.npy', talikAnnFEEDBACKmean)
    
    
    ''' ------------------------- '''      
    #### Peatland extent       
    ''' ------------------------- ''' 
    #########################################################
    # Peatland fraction (at least 10%) for contour on map
    #########################################################
    peatland = getLandType('ER',False,False)
    peatlandContour = peatland.copy(); peatlandContour = xr.where(peatland >= 10, 1, 0)
    
    
    ''' ------------------------------ '''
    #### FIGURES: talik
    ''' ------------------------------ '''
    #########################################################
    #### Talik: ens mean control
    #########################################################
    import pickle
    with open(datadir + '/talikAnnFEEDBACK.pkl', 'rb') as fp:
        talikAnnFEEDBACK = pickle.load(fp)
    with open(datadir + '/talikAnnCONTROL.pkl', 'rb') as fp:
        talikAnnCONTROL = pickle.load(fp)
    
    talikAnnCONTROLmean = np.load(datadir + '/talikAnnCONTROLmean_NEW.npy')
    talikAnnFEEDBACKmean = np.load(datadir + '/talikAnnFEEDBACKmean_NEW.npy')
    
    longitude = lon
    var,lon2 = add_cyclic_point(talikAnnCONTROLmean[11:,:],coord=longitude)
    
    ## Create figure
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.Normalize(vmin=0, vmax=54)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 49, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    
    ## Filled contour map
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    cf1 = ax.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=magma)
    
    ax.contour(lon,lat,peatland[41:,:],[10],colors='g',
                    linewidth=0.4,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'rotation':0}
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    # cbar = plt.colorbar(cf1, ax=ax, ticks=[0,10,20,30,40,50], fraction=0.045,location='bottom',orientation='horizontal')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=magma),
             ax=ax, orientation='vertical', ticks=[0,10,20,30,40,50],fraction=0.05) # change orientation if needed
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_yticklabels(['2015','2025','2035','2045','2055','2065'])
    cbar.set_label('Talik formation year', fontsize=12)
    plt.title('a) SSP2-4.5 talik formation year', fontsize=13, fontweight='bold')
    ## Save figure
    plt.savefig(figuredir + '/Fig3a_control_talik_formation_year_soil_only_MEAN_peat.pdf', 
                dpi=1200, bbox_inches='tight')
    del fig,ax,var,lon2,longitude,cf1,cbar
    
    #########################################################
    #### Talik: ens mean feedback
    #########################################################
    longitude = lon
    var,lon2 = add_cyclic_point(talikAnnFEEDBACKmean[11:,:],coord=longitude)
    
    ## Create figure
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.Normalize(vmin=0, vmax=54)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 49, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_facecolor('0.8')
    ## Filled contour map
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    cf1 = ax.pcolormesh(lon2,lat,var,transform=ccrs.PlateCarree(), 
                  norm=norm, cmap=magma)
    ax.contour(lon,lat,peatland[41:,:],[10],colors='g',
                linewidth=0.4,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'rotation':0}
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=magma),
             ax=ax, orientation='vertical', ticks=[0,10,20,30,40,50],fraction=0.05) # change orientation if needed
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_yticklabels(['2015','2025','2035','2045','2055','2065'])
    cbar.set_label('Talik formation year', fontsize=12)
    plt.title('b) ARISE-SAI-1.5 talik formation year', fontsize=13, fontweight='bold')
    ## Save figure
    plt.savefig(figuredir + '/Fig3b_feedback_talik_formation_year_soil_only_MEAN_peat.pdf', 
                dpi=1200, bbox_inches='tight')
    
    
    #########################################################
    #### Talik: diff
    #########################################################
    talikAnnFEEDBACKmean = np.load(datadir + '/talikAnnFEEDBACKmean.npy')
    talikAnnCONTROLmean = np.load(datadir + '/talikAnnCONTROLmean.npy')
    cMapthawedControlNotFeedback = c.ListedColormap(['xkcd:aqua blue'])
    cMapthawedFeedbackNotControl = c.ListedColormap(['xkcd:bright yellow'])
    cMapALWAYSTHAW = c.ListedColormap(['k'])
    
    
    ## thawed in control but not in feedback
    thawedControlNotFeedback = talikAnnFEEDBACKmean.copy()
    thawedControlNotFeedback[
        (np.isnan(talikAnnFEEDBACKmean)) & 
        (~np.isnan(talikAnnCONTROLmean))] = 100 # nan in feedback mean = didn't thaw in FB
    thawedControlNotFeedback[thawedControlNotFeedback < 100.] = np.nan
    
    ## talik in feedback but not in control:
    thawedFeedbackNotControl = talikAnnCONTROLmean.copy()
    thawedFeedbackNotControl[
        (np.isnan(talikAnnCONTROLmean)) & 
        (~np.isnan(talikAnnFEEDBACKmean))] = -100
    thawedFeedbackNotControl[thawedFeedbackNotControl > -100.] = np.nan
    
    ## thawed by 2035:
    talikALWAYSmean = talikAnnCONTROLmean.copy()
    talikALWAYSmean[
        (talikAnnCONTROLmean >= 20) |
        (talikAnnFEEDBACKmean >= 20)] = np.nan
    # talikALWAYSmean = talikAnnCONTROLmean + talikAnnFEEDBACKmean
    # talikALWAYSmean[talikALWAYSmean != 0] = np.nan
            
    ## difference in thaw timing between control and feedback
    diffMEAN = talikAnnCONTROLmean - talikAnnFEEDBACKmean
    
    ## mask out cells that didn't thaw in control or feedback (diff = nan) and always thawed
    diffMEAN[(thawedControlNotFeedback == 100) | (thawedFeedbackNotControl == -100) | (talikALWAYSmean < 20)] = np.nan # == 0

    
    ## Create figure ##
    fig = plt.figure(figsize=(10,6))
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
    longitude = lon; plottingVarMean,lon2 = add_cyclic_point(diffMEAN,coord=longitude)
    
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([180, -180, 49, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes); ax.set_facecolor('0.8')
    
    ## land mask
    ax.pcolormesh(lon,lat,landMask[41:,:],transform=ccrs.PlateCarree(),cmap=cmapLand)
    ## thawed in control but not feedback
    ax.pcolormesh(lon,lat,thawedControlNotFeedback[11:,:],transform=ccrs.PlateCarree(),
                        cmap=cMapthawedControlNotFeedback)
    ## thawed in feedback but not control
    ax.pcolormesh(lon,lat,thawedFeedbackNotControl[11:,:],transform=ccrs.PlateCarree(),
                        cmap=cMapthawedFeedbackNotControl)
    ## always thawed = black
    ax.pcolormesh(lon,lat,talikALWAYSmean[11:,:],transform=ccrs.PlateCarree(),
                        cmap=cMapALWAYSTHAW)
    ## difference in thaw timing
    cf1 = ax.pcolormesh(lon2,lat,plottingVarMean[11:,:],transform=ccrs.PlateCarree(), 
                  norm=norm, cmap='bwr') # seismic
    ## peatland contour
    ax.contour(lon,lat,peatland[41:,:],[10],colors='g',
                    linewidth=0.5,transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.9)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True,
                      linewidth=1, color='C7', 
                  alpha=0.8, linestyle=':',
                  x_inline=False,
                  y_inline=True,
                  rotate_labels=False)
    gl.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #gl.xlabel_style = {'rotation':0}
    gl.ylabel_style = {'size': 9, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`
    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
            
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'),
             ax=ax, orientation='vertical', ticks=[-8.5, 0, 8.5], fraction=0.045) # change orientation if needed
    # cbar = plt.colorbar(cf1, ax=ax, ticks=[-8.5, 0, 8.5], fraction=0.045,location='bottom',orientation='horizontal')
    cbar.ax.set_yticklabels(['Talik forms\nearlier in\nSSP2-4.5', 
                             'Talik forms\nsame year', 
                             'Talik forms\nearlier in\nARISE-1.5'])
    cbar.ax.tick_params(size=0)
    
    plt.title('c) Talik formation year, SSP2-4.5 minus ARISE-SAI-1.5', fontsize=13, fontweight='bold')
    plt.savefig(figuredir + '/Fig3c_control_minus_feedback_talik_formation_soil_only_MEAN_peat.pdf', 
                dpi=1200, bbox_inches='tight')
    del fig, ax, cf1
    
    
    #### area of talik prevented
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area
    ds.close()
    
    ## grid areas
    print("Talik area in SSP2-4.5: ", (np.array(np.nansum(gridArea[41:,:].where(~np.isnan(talikAnnCONTROLmean))/(1000**2),axis=(0,1))))/1e6, "million km2")
    gridAreaThaw = np.array(np.nansum((gridArea[41:,:].where(
        ~np.isnan(thawedControlNotFeedback))/(1000**2)),axis=(0,1)))
    print("Talik prevented by SAI: ", np.round(gridAreaThaw/1e6, decimals=2), "million km2")
    gridAreaSAI = np.array(np.nansum((gridArea[41:,:].where(~np.isnan(thawedFeedbackNotControl))/(1000**2)),axis=(0,1)))
    print("Talik caused by SAI: ", np.round(gridAreaSAI/1e6, decimals=2), "million km2")
    gridAreaSSP = np.array(np.nansum((gridArea[41:,:].where((diffMEAN < 0))/(1000**2)),axis=(0,1)))
    print("Talik delayed by SAI: ", np.round(gridAreaSSP/1e6, decimals=2), "million km2")
    # gridAreaPeat = np.array(np.nansum((gridArea[41:,:].where(
    #     (~np.isnan(thawedControlNotFeedback)) | (diffMEAN < 0) & (
    #         peatland[41:,:].values > 10))/(1000**2)),axis=(0,1)))
    gridAreaPeat = np.array(np.nansum((gridArea[41:,:].where(
        (~np.isnan(thawedControlNotFeedback)) & (
            peatland[41:,:].values > 0))/(1000**2)),axis=(0,1)))
    print("Peat talik prevented by SAI: ", np.round(gridAreaPeat/1e6, decimals=2), "million km2")
    print("Percent talik prevented by SAI in peat: ", (gridAreaPeat/(gridAreaThaw+gridAreaSSP))*100)
    
    
    #### talik area time series
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area[41:,:]
    ds.close()
    
    import pickle
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnFEEDBACK.pkl', 'rb') as fp:
        talikAnnFEEDBACK = pickle.load(fp)
    with open('/Users/arielmor/Desktop/SAI/data/ARISE/data/talikAnnCONTROL.pkl', 'rb') as fp:
        talikAnnCONTROL = pickle.load(fp)
    
    # control
    talikAreaCONTROL  = np.zeros((55))
    talikAreaCONTROL[0] = np.nansum(gridArea.where(talikAnnCONTROLmean[11:,:] == 0),axis=(0,1))
    # feedback
    talikAreaFEEDBACK = np.zeros((55))
    talikAreaFEEDBACK[0] = talikAreaFEEDBACK[0] + np.nansum(
        gridArea.where(talikAnnFEEDBACKmean[11:,:] == 0),axis=(0,1))
    # calculation
    for iyear in range(1,55):
        talikAreaCONTROL[iyear] = talikAreaCONTROL[iyear-1] + np.nansum(
            gridArea.where(talikAnnCONTROLmean[11:,:] == iyear),axis=(0,1))
        talikAreaFEEDBACK[iyear] = talikAreaFEEDBACK[iyear-1] + np.nansum(
            gridArea.where(talikAnnFEEDBACKmean[11:,:] == iyear),axis=(0,1))
        
    ## ensemble members
    talikAreaEnsCONTROL  = {}
    talikAreaEnsFEEDBACK = {}
    for i in range(len(ens)):
        talikAreaEnsCONTROL[ens[i]]  = np.zeros((55))
        talikAreaEnsCONTROL[ens[i]][0] = np.nansum(gridArea.where(talikAnnCONTROL[ens[i]] == 0),axis=(0,1))
        talikAreaEnsFEEDBACK[ens[i]] = np.zeros((55))
        talikAreaEnsFEEDBACK[ens[i]][0] = np.nansum(gridArea.where(
                                    talikAnnFEEDBACK[ens[i]] == 0),axis=(0,1))
        for iyear in range(1,55):
            talikAreaEnsCONTROL[ens[i]][iyear] = talikAreaEnsCONTROL[ens[i]][iyear-1] + np.nansum(
                gridArea.where(talikAnnCONTROL[ens[i]] == iyear),axis=(0,1))
            talikAreaEnsFEEDBACK[ens[i]][iyear] = talikAreaEnsFEEDBACK[ens[i]][iyear-1] + np.nansum(
                gridArea.where(talikAnnFEEDBACK[ens[i]] == iyear),axis=(0,1))
            
    talikAreaCONTROL = make_ensemble_mean_timeseries(talikAreaEnsCONTROL, 10)
    talikAreaFEEDBACK = make_ensemble_mean_timeseries(talikAreaEnsFEEDBACK, 10)
        
    #### Fig. 4 talik time series
    fig, ax = plt.subplots(1,1, figsize=(9,5))
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),talikAreaEnsCONTROL[ens[i]]/(1000**2)/1e6,color='xkcd:pale red',
                label='SSP2-4.5',linestyle='--',linewidth=0.9)
        ax.plot(np.linspace(2035,2069,35),talikAreaEnsFEEDBACK[ens[i]][20:]/(1000**2)/1e6,color='xkcd:sky blue',
                label='ARISE-SAI-1.5',linewidth=0.9)
    ax.plot(np.linspace(2015,2069,55),talikAreaCONTROL/(1000**2)/1e6,linewidth=3,color='xkcd:scarlet',linestyle='--')
    ax.plot(np.linspace(2035,2069,35),talikAreaFEEDBACK[20:]/(1000**2)/1e6,linewidth=3,color='xkcd:blue')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:scarlet', lw=3, linestyle='--'),
                    Line2D([0], [0], color='xkcd:blue', lw=3)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.ylabel('Area (km$^2$ x 10$^6$)', fontsize=12)
    plt.title('Talik area', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(figuredir + '/Fig4_talik_area_timeseries.pdf',
                dpi=1200, bbox_inches='tight')
    
    
    return