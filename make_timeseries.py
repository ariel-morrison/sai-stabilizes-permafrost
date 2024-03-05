#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:19:32 2022

@author: Ariel L. Morrison
"""

def make_timeseries(numEns,var,lat,lon,latmax,latmin,lonmax,lonmin,dataDict):
    import numpy as np
    import warnings
    # import xarray as xr
        
    if len(lat) == 85: lat = lat[41:]
    
    lengthDictionary = len(dataDict)
    
    if lengthDictionary > 1: ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    else: ens = ['001']
    numEns = len(ens)
    
    # datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    # ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    # landmask = ds.landmask
    # ds.close()
    
    latmin_ind = int(np.abs(latmin-lat).argmin())
    latmax_ind = int(np.abs(latmax-lat).argmin())+1
    lonmin_ind = int(np.abs(lonmin-lon).argmin())
    lonmax_ind = int(np.abs(lonmax-lon).argmin())+1
    print(latmin_ind, latmax_ind)
    print(lonmin_ind, lonmax_ind)
    
    # Latitude weighting
    lonmesh,latmesh = np.meshgrid(lon,lat)
    
    # Mask out ocean and non-permafrost land:
    weights2D = {}
    for i in range(numEns):
        weights2D[ens[i]] = np.full((dataDict[ens[i]].shape[0],len(lat),len(lon)),np.nan) 
        for iyear in range(dataDict[ens[0]].shape[0]):
            weights2D[ens[i]][iyear,:,:] = np.cos(np.deg2rad(latmesh))
            weights2D[ens[i]][iyear,:,:][np.isnan(dataDict[ens[i]][iyear,:,:])] = np.nan
    
    
    # Annual time series for each ensemble member
    ensMemberTS = {}
    for ensNum in range(numEns):
        warnings.simplefilter("ignore")
                                            
        ensMasked         = dataDict[ens[ensNum]]
        ensMasked_grouped = ensMasked[:,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind]
        ensMasked_grouped = np.ma.MaskedArray(ensMasked_grouped, mask=np.isnan(ensMasked_grouped))
        weights           = np.ma.asanyarray(weights2D[ens[ensNum]][
                                :,latmin_ind:latmax_ind,lonmin_ind:lonmax_ind])
        weights.mask      = ensMasked_grouped.mask
        ensMemberTS[ens[ensNum]] = np.array([np.ma.average(
                                            ensMasked_grouped[i],
                                            weights=weights[i]
                                            ) for i in range((ensMasked_grouped.shape)[0])])
    return ensMemberTS


def make_permafrost_extent_timeseries(datadir,numEns,lat,lon,dataDict):
    import numpy as np
    import xarray as xr
    import warnings
    
    if len(lat) == 85: lat = lat[41:]
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area[41:,:]
    ds.close()
    
    # years = len(dataDict[ens[0]].time) / 12.
    years = len(dataDict[ens[0]].year)
    gridArea = np.repeat(gridArea.values[None,...],years,axis=0)

    
    #### Annual time series for each ensemble member
    permafrostExtentMembersTS = {}
    for ensNum in range(numEns):
        warnings.simplefilter("ignore")
        dataDict_annMean = dataDict[ens[ensNum]]#.groupby('time.year').max(dim='time',skipna=True)
        dataDict_masked = (np.ma.masked_where(np.isnan(dataDict_annMean), gridArea)) / (1000**2) # convert to sq km
        permafrostExtentMembersTS[ens[ensNum]] = np.array(np.nansum(dataDict_masked, axis=(1,2)))
        
    # Ensemble mean
    permafrostExtentTS = 0
    for val in permafrostExtentMembersTS.values():
        permafrostExtentTS += val
    permafrostExtentTS = permafrostExtentTS/numEns 
    
    
    #### Annual mean permafrost volume
    from tippingPoints import findPermafrost
    
    ds = xr.open_dataset('/Users/arielmor/Desktop/SAI/data/CESM2/gridareaNH.nc')
    gridArea = ds.cell_area[41:,:]
    ds.close()
    
    # years = len(dataDict[ens[0]].time) / 12.
    gridAreaCONTROL = np.repeat(gridArea.values[None,...],55,axis=0)
    gridAreaFEEDBACK = np.repeat(gridArea.values[None,...],35,axis=0)
    
    altAnnMeanCONTROL,altmaxMonthlyCONTROL,altmaxAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.','201501-206912')
    altAnnMeanFEEDBACK,altmaxMonthlyFEEDBACK,altmaxAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    pfrostVolumeAnnCONTROL  = {}
    pfrostVolumeAnnFEEDBACK = {}
    for ensNum in range(numEns):
        pfrostVolumeAnnCONTROL[ens[ensNum]]  = (8.6 - altAnnMeanCONTROL[ens[ensNum]]) * gridAreaCONTROL
        pfrostVolumeAnnFEEDBACK[ens[ensNum]] = (8.6 - altAnnMeanFEEDBACK[ens[ensNum]]) * gridAreaFEEDBACK
    
    return permafrostExtentTS, permafrostExtentMembersTS, pfrostVolumeAnnCONTROL, pfrostVolumeAnnFEEDBACK


def make_ensemble_mean_timeseries(ensMemberTS,numEns):
    ensMeanTS = 0
    if type(ensMemberTS).__module__ == 'numpy':
        for val in ensMemberTS:
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    else:
        for val in ensMemberTS.values():
            ensMeanTS += val
        ensMeanTS = ensMeanTS/numEns 
    return ensMeanTS
