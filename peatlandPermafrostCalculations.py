#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:45:21 2022

@author: Ariel L. Morrison

The 'irreversibility' of carbon loss from permafrost thaw depends on the ground
type: forests can resequester carbon during the growing season via photosynthesis
and carbon fixing, but peatland, as an anaerobic environment, cannot resequester
lost carbon on decadal time scales. As a result, carbon lost from thawing peatlands 
can be considered 'permanently' lost from the permafrost reservoir.
"""
def findPermafrost(simulation, timePeriod, datadir):
    if simulation == 'arise':
        simName = "b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT."
    elif simulation == 'ssp':
        simName = "b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM."
    
    import numpy as np
    import xarray as xr
    import pandas as pd
        
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    numEns       = len(ens)
    alt          = {}
    altMasked    = {}
    altAnnMean   = {}
    altmax       = {}
    altmaxMasked = {}
    altmaxAnn    = {}
    
    ds = xr.open_dataset(datadir + 
                         '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    nbedrock = ds.nbedrock[41:,:]
    nbedrock = nbedrock.fillna(19)
    landmask = ds.landmask[41:,:]
    levgrnd  = [0.02,0.06,0.12,0.2,0.32,0.48,0.68,0.92,1.2,1.52,1.88,2.28,2.72,3.26,3.9,
                4.64,5.48,6.42,7.46,8.6,10.99,15.666,23.301,34.441,49.556]
    if len(ds.lat) > 50: lat = ds.lat[41:]
    else: lat = ds.lat
    lon = ds.lon
    ds.close()
    
    
    #### index of shallowest bedrock layer
    bedrock = np.zeros((len(lat),len(lon)))
    for ilat in range(len(lat)):
        for ilon in range(len(lon)):
            bedrockIndex = int(nbedrock[ilat,ilon].values)
            bedrock[ilat,ilon] = levgrnd[bedrockIndex]
    
    
    if timePeriod == '201501-206912':
        bedrock = np.repeat(bedrock[None,...],660,axis=0)  
    elif timePeriod == '203501-206912':
        bedrock = np.repeat(bedrock[None,...],420,axis=0)
            
    #### constrain permafrost to only soil
    for i in range(numEns):
        ## alt
        ds = xr.open_dataset(datadir + '/' + str(simName) + str(ens[i]) +
                             '.clm2.h0.ALT.' + 
                             str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        alt[ens[i]] = ds.ALT[:,41:,:]
        ds.close()
        ## restrict to soil layers in CLM5
        altMasked[ens[i]] = alt[ens[i]].where(alt[ens[i]] <= bedrock)
        ## top 20 layers are soil - deepest bedrock starts at 8.6m, cut off at 8.60
        altMasked[ens[i]] = altMasked[ens[i]].where(altMasked[ens[i]] <= 8.60)
        ## get annual mean active layer depth
        altAnnMean[ens[i]] = altMasked[ens[i]].groupby('time.year').mean(dim='time', skipna = True)
        
    for i in range(numEns):
        ## altmax
        ds = xr.open_dataset(datadir + '/' + str(simName) + str(ens[i]) +
                             '.clm2.h0.ALTMAX.' + 
                             str(timePeriod) + '_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        ## restrict to poleward of 50N ##
        altmax[ens[i]] = ds.ALTMAX[:,41:,:]
        ## restrict to soil layers in CLM5
        altmaxMasked[ens[i]] = altmax[ens[i]].where(altmax[ens[i]] <= bedrock)
        ## top 20 layers are soil - deepest bedrock starts at 8.6m, cut off at 8.60
        altmaxMasked[ens[i]] = altmaxMasked[ens[i]].where(altmaxMasked[ens[i]] <= 8.60)
        ## get annual maximum active layer depth
        altmaxAnn[ens[i]] = altmaxMasked[ens[i]].groupby('time.year').max(dim='time', skipna = True)
        
    return altAnnMean,altmaxMasked,altmaxAnn,lat,lon


def getLandType(dataDir,figureDir):
    import xarray as xr
    import numpy as np
    import pandas as pd
    from plottingFunctions import make_maps
    from make_timeseries import make_timeseries, make_ensemble_mean_timeseries
    import matplotlib.pyplot as plt
    
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    numEns = len(ens)
    
    ds = xr.open_dataset(dataDir + '/peatareaGlobal.nc')
    peatland = ds.peatf[-64:,:] * 100. #restrict to +30N
    peatland = peatland.where(peatland > 0.)
    ds.close()
    
    ''' Grid cell area '''
    ds = xr.open_dataset(dataDir + '/gridareaNH.nc')
    gridArea = ds.cell_area
    ds.close()
    
    ''' Get total peatland area in sq km (>5% peatland in grid cell) '''
    peatlandArea = np.array(np.nansum((gridArea.where(peatland.values >= 0.1)/(1000**2)),axis=(0,1)))
    print("Area of grid cells poleward of 10N that are >= 10% peatland: ", 
          np.round(peatlandArea/1e6, decimals=2), "million km2")
    
    ''' Max active layer depth '''
    ########################################################
    #### Pfrost extent (active layer)
    ########################################################
    altAnnMeanCONTROL,pfrostCONTROL,altmaxAnnCONTROL,lat,lon = findPermafrost('ssp', '201501-206912', dataDir)
    altAnnMeanFEEDBACK,pfrostFEEDBACK,altmaxAnnFEEDBACK,lat,lon = findPermafrost('arise', '203501-206912', dataDir)
    
    numEns = len(ens)
    pfrostControl = {}
    pfrostControl2034 = {} # for calculating carbon stocks
    pfrostFeedback = {}
    for i in range(numEns):
        pfrostControl[ens[i]] = altmaxAnnCONTROL[ens[i]][20:,:,:]
        pfrostControl2034[ens[i]] = altmaxAnnCONTROL[ens[i]][19,:,:]
        pfrostFeedback[ens[i]] = altmaxAnnFEEDBACK[ens[i]]
        
    
    ''' Each year, how much permafrost is in peatland? '''
    years                       = np.linspace(2035,2069,35)
    pfrostInPeatlandFeedback    = {}
    pfrostInPeatlandControl     = {}
    pfrostInPeatlandControl2034 = {}
    peatlandPfrostFeedbackArea  = np.zeros((numEns,35)) * np.nan
    totalPfrostFeedbackArea     = np.zeros((numEns,35)) * np.nan
    peatlandPfrostControlArea   = np.zeros((numEns,35)) * np.nan
    totalPfrostControlArea      = np.zeros((numEns,35)) * np.nan
    
    for i in range(numEns):
        pfrostInPeatlandFeedback[ens[i]]     = np.zeros((35,pfrostCONTROL[ens[0]].shape[1],pfrostCONTROL[ens[0]].shape[2])) * np.nan
        pfrostInPeatlandControl[ens[i]]      = np.zeros((35,pfrostCONTROL[ens[0]].shape[1],pfrostCONTROL[ens[0]].shape[2])) * np.nan
        pfrostInPeatlandControl2034[ens[i]]  = np.zeros((pfrostCONTROL[ens[0]].shape[1],pfrostCONTROL[ens[0]].shape[2])) * np.nan
        pfrostInPeatlandControl2034[ens[i]]  = pfrostControl[ens[i]][19,:,:].where(
                                                    peatland[20:,:].values >= 0.1)
        
        for iyear in range(len(years)):
            ## pfrostFeedback = annual average (30x85x288)
            ## altmaxFeedback = every month (420x85x288)
            ## -- restricted only to where >= 10% of cell is considered peat -- ##
            pfrostInPeatlandFeedback[ens[i]][iyear,:,:] = pfrostFeedback[ens[i]][iyear,:,:].where(
                                                                                    peatland[20:,:].values >= 0.1)
            pfrostInPeatlandControl[ens[i]][iyear,:,:]  = pfrostControl[ens[i]][iyear,:,:].where(
                                                                                    peatland[20:,:].values >= 0.1)
            
            ## add grid area for permafrost soils
            totalPfrostFeedbackArea[i,iyear] = np.array(np.nansum((gridArea[20:,:].where(
                                                    ~np.isnan(pfrostFeedback[ens[i]][iyear,:,:]))/(
                                                        1000**2)),axis=(0,1)))
            totalPfrostControlArea[i,iyear]  = np.array(np.nansum((gridArea[20:,:].where(
                                                    ~np.isnan(pfrostControl[ens[i]][iyear,:,:]))/(
                                                    1000**2)),axis=(0,1)))
            
            ## add grid area for permafrost soils in peatland
            peatlandPfrostFeedbackArea[i,iyear] = np.array(np.nansum((gridArea[20:,:].where(
                                                      ~np.isnan(pfrostInPeatlandFeedback[ens[i]][iyear,:,:]))/(
                                                      1000**2)),axis=(0,1)))
            peatlandPfrostControlArea[i,iyear]  = np.array(np.nansum((gridArea[20:,:].where(
                                                      ~np.isnan(pfrostInPeatlandControl[ens[i]][iyear,:,:]))/(
                                                      1000**2)),axis=(0,1)))
                                                          
                                                          
    #### Fig. 5a peatland map
    import matplotlib as mpl
    pinks = mpl.colormaps['pink_r'].resampled(20)
    fig, ax = make_maps(peatland[20:,:],lat,lon,0,100,20,pinks,'% peatland cover',
                        'a) Fixed peatland area in CESM2','Fig5a_peat_fraction','neither',False,False)
    
    #### Fig. 5b permafrost peatland figure
    ensMeanFeedback = make_ensemble_mean_timeseries(peatlandPfrostFeedbackArea, numEns)
    ensMeanControl = make_ensemble_mean_timeseries(peatlandPfrostControlArea, numEns)
    
    fig, ax = plt.subplots(figsize=(12,5),dpi=1200)    
    for i in range(numEns):
        plt.plot(
            (100 - (peatlandPfrostFeedbackArea[i,0] - peatlandPfrostFeedbackArea[
                i,:])/peatlandPfrostFeedbackArea[i,0]*100),
                color='xkcd:sky blue',linewidth=0.9)
        plt.plot(
            (100 - (peatlandPfrostControlArea[i,0] - peatlandPfrostControlArea
                    [i,:])/peatlandPfrostControlArea[i,0]*100),
                color='xkcd:pale red',linewidth=0.9,linestyle='--')
    plt.plot(100 - ((ensMeanFeedback[0] - ensMeanFeedback[:])/ensMeanFeedback[0]*100),
             color='xkcd:blue',linewidth=3,label='ARISE-SAI-1.5')
    plt.plot(100 - ((ensMeanControl[0] - ensMeanControl[:])/ensMeanControl[0]*100),
             color='xkcd:scarlet',linewidth=3,label='SSP2-4.5',linestyle='--')
    print("control percent: ", (100 - ((ensMeanControl[0] - ensMeanControl[:])/ensMeanControl[0]*100)))
    print(" ")
    print("feedback percent: ", (100 - ((ensMeanFeedback[0] - ensMeanFeedback[:])/ensMeanFeedback[0]*100)))
    plt.ylabel('% of 2035 permafrost peatland remaining', fontsize=11)
    plt.title('b) Permafrost peatland remaining', fontsize=14, fontweight='bold')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim(top=105)
    plt.axhline(100,linestyle='dotted',color='gray',linewidth=1)
    plt.legend(fancybox=True, fontsize=13)
    plt.savefig(figureDir + 'Fig5b_percent_peatland_permafrost_soil_only_timeseries.jpg', 
                dpi=1200, bbox_inches='tight')
    
    from scipy.stats import ttest_ind
    v1 = (100 - ((ensMeanFeedback[0] - ensMeanFeedback[:])/ensMeanFeedback[0]*100))
    v2 = (100 - ((ensMeanControl[0] - ensMeanControl[:])/ensMeanControl[0]*100))
    res = ttest_ind(v1[:5], v2[:5])
    print("permafrost peatland remaining stats: ", res) # sig different at 95% confidence level by 4th year
                                                        
    
    ####################################
    ## CARBON OUT OF PERMAFROST SOILS ##
    ####################################
    '''
    PF Domain Soil C (soil and litter C changes)
    PF Domain TOTECOSYSC (total ecosystem carbon)
    PF Domain Vegetation C
    '''
        
    #### TOTECOSYSC
    '''total ecosystem carbon - control '''
    TOTECOSYSCControl           = {}
    TOTECOSYSCPeatControl       = {}
    TOTECOSYSCPermafrostControl = {}
    TOTECOSYSCControl_gC           = {}
    TOTECOSYSCPermafrostControl_gC = {}
    TOTECOSYSCPeatControl_gC       = {}
    
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) +
                             '.clm2.h0.TOTECOSYSC.201501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat; lon = ds.lon; 
        
        TOTECOSYSCControl_all           = ds['TOTECOSYSC'][240:,41:,:]
        
        ## annual mean in pfrost, gC/m2
        TOTECOSYSCControl[ens[i]]           = TOTECOSYSCControl_all.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTECOSYSCPermafrostControl[ens[i]] = TOTECOSYSCControl[ens[i]].where(
                                                        ~np.isnan(pfrostControl[ens[i]]))
        # only in peatland permafrost
        TOTECOSYSCPeatControl[ens[i]]       = TOTECOSYSCPermafrostControl[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl[ens[i]]))
        
        ## annual mean in pfrost, gC in 2034 pf domain
        TOTECOSYSCControl_gC[ens[i]]           = ((TOTECOSYSCControl_all.groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea[20:,:])
        TOTECOSYSCPermafrostControl_gC[ens[i]] = TOTECOSYSCControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostControl2034[ens[i]]))
        # only in peatland permafrost
        TOTECOSYSCPeatControl_gC[ens[i]]       = TOTECOSYSCPermafrostControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
        ds.close()
        
        
    '''total ecosystem carbon from permafrost soils - FEEDBACK'''
    TOTECOSYSCFeedback           = {}
    TOTECOSYSCPeatFeedback       = {}
    TOTECOSYSCPermafrostFeedback = {}
    TOTECOSYSCFeedback_gC           = {}
    TOTECOSYSCPermafrostFeedback_gC = {}
    TOTECOSYSCPeatFeedback_gC       = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) +
                             '.clm2.h0.TOTECOSYSC.203501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat
        lon = ds.lon
        TOTECOSYSCFeedback_all          = ds['TOTECOSYSC'][:,41:,:]
        ds.close()
        
        ## annual mean in pfrost, gC/m2
        TOTECOSYSCFeedback[ens[i]]           = TOTECOSYSCFeedback_all.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTECOSYSCPermafrostFeedback[ens[i]] = TOTECOSYSCFeedback[ens[i]].where(
                                                ~np.isnan(pfrostFeedback[ens[i]]))
        # only in permafrost peatland
        TOTECOSYSCPeatFeedback[ens[i]]       = TOTECOSYSCFeedback[ens[i]].where(
                                                ~np.isnan(pfrostInPeatlandFeedback[ens[i]]))
        
        ## annual mean in pfrost, gC in inital (2035) pf domain
        TOTECOSYSCFeedback_gC[ens[i]]           = (TOTECOSYSCFeedback_all.groupby('time.year').sum(
                                                                    dim='time', skipna=True)*gridArea[20:,:])
        TOTECOSYSCPermafrostFeedback_gC[ens[i]] = TOTECOSYSCFeedback_gC[ens[i]].where(
                                                        ~np.isnan(pfrostControl2034[ens[i]]))
        # only in peatland permafrost
        TOTECOSYSCPeatFeedback_gC[ens[i]]       = TOTECOSYSCPermafrostFeedback_gC[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
        
    #### TOTSOMC
    '''total soil and litter carbon - control '''
    TOTSoilCPeatControl       = {}
    TOTSoilCPermafrostControl = {}
    TOTSoilCPeatControl_gC       = {}
    TOTSoilCPermafrostControl_gC = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) +
                             '.clm2.h0.TOTSOMC.201501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        TOTSOMCControl_ungrouped         = ds['TOTSOMC'][240:,:,:]
        ds.close()
        
        ds = xr.open_dataset(dataDir + '/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) +
                             '.clm2.h0.TOTLITC.201501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        TOTLitCControl_ungrouped         = ds['TOTLITC'][240:,:,:]
        ds.close()
        
        TOTSoilCControl_ungrouped = TOTSOMCControl_ungrouped + TOTLitCControl_ungrouped
        
        ## annual mean in pfrost, gC/m2
        TOTSoilCPermafrostControl[ens[i]] = TOTSoilCControl_ungrouped.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTSoilCPermafrostControl[ens[i]] = TOTSoilCPermafrostControl[ens[i]].where(
                                                        ~np.isnan(pfrostControl[ens[i]]))
        # only in peatland permafrost
        TOTSoilCPeatControl[ens[i]]       = TOTSoilCPermafrostControl[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl[ens[i]]))
        
        ## annual mean in pfrost, gC
        TOTSoilCPermafrostControl_gC[ens[i]] = (TOTSoilCControl_ungrouped.groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea[20:,:]
        TOTSoilCPermafrostControl_gC[ens[i]] = TOTSoilCPermafrostControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostControl2034[ens[i]]))
        # only in peatland permafrost
        TOTSoilCPeatControl_gC[ens[i]]       = TOTSoilCPermafrostControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
        
        
    '''total soil organic matter carbon - FEEDBACK'''
    TOTSoilCPeatFeedback       = {}
    TOTSoilCPermafrostFeedback = {}
    TOTSoilCPeatFeedback_gC       = {}
    TOTSoilCPermafrostFeedback_gC = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) +
                             '.clm2.h0.TOTSOMC.203501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        TOTSOMCFeedback_ungrouped          = ds['TOTSOMC']
        ds.close()
        
        ds = xr.open_dataset(dataDir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) +
                             '.clm2.h0.TOTLITC.203501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        TOTLITCFeedback_ungrouped          = ds['TOTLITC']
        ds.close()
        
        TOTSoilCFeedback = TOTSOMCFeedback_ungrouped + TOTLITCFeedback_ungrouped
        
        ## annual mean in pfrost, gC/m2
        TOTSoilCPermafrostFeedback[ens[i]] = TOTSoilCFeedback.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTSoilCPermafrostFeedback[ens[i]]  = TOTSoilCPermafrostFeedback[ens[i]].where(
                                                ~np.isnan(pfrostFeedback[ens[i]]))
        # only in permafrost peatland
        TOTSoilCPeatFeedback[ens[i]]        = TOTSoilCPermafrostFeedback[ens[i]].where(
                                                ~np.isnan(pfrostInPeatlandFeedback[ens[i]]))
        
        ## annual mean in pfrost, gC
        TOTSoilCPermafrostFeedback_gC[ens[i]] = (TOTSoilCFeedback.groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea[20:,:]
        TOTSoilCPermafrostFeedback_gC[ens[i]]  = TOTSoilCPermafrostFeedback_gC[ens[i]].where(
                                                ~np.isnan(pfrostControl2034[ens[i]]))
        # only in permafrost peatland
        TOTSoilCPeatFeedback_gC[ens[i]]        = TOTSoilCPermafrostFeedback_gC[ens[i]].where(
                                                ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
        
    #### TOTVEGC
    '''total veg carbon - control '''
    TOTVEGCControl           = {}
    TOTVEGCPeatControl       = {}
    TOTVEGCPermafrostControl = {}
    TOTVEGCControl_gC        = {}
    TOTVEGCPeatControl_gC       = {}
    TOTVEGCPermafrostControl_gC = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) +
                             '.clm2.h0.TOTVEGC.201501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat; lon = ds.lon; 
        
        TOTVEGCControl_ungrouped          = ds['TOTVEGC'][240:,:,:]
        
        ## annual mean in pfrost, gC/m2
        TOTVEGCControl[ens[i]]           = TOTVEGCControl_ungrouped.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTVEGCPermafrostControl[ens[i]] = TOTVEGCControl[ens[i]].where(
                                                        ~np.isnan(pfrostControl[ens[i]]))
        # only in peatland permafrost
        TOTVEGCPeatControl[ens[i]]       = TOTVEGCPermafrostControl[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl[ens[i]]))
        
        ## annual mean in pfrost, gC
        TOTVEGCControl_gC[ens[i]]           = (TOTVEGCControl_ungrouped.groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea[20:,:]
        TOTVEGCPermafrostControl_gC[ens[i]] = TOTVEGCControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostControl2034[ens[i]]))
        # only in peatland permafrost
        TOTVEGCPeatControl_gC[ens[i]]       = TOTVEGCPermafrostControl_gC[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
        ds.close()
        
        
    '''total veg carbon - FEEDBACK'''
    TOTVEGCFeedback           = {}
    TOTVEGCPeatFeedback       = {}
    TOTVEGCPermafrostFeedback = {}
    TOTVEGCFeedback_gC         = {}
    TOTVEGCPeatFeedback_gC     = {}
    TOTVEGCPermafrostFeedback_gC = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) +
                             '.clm2.h0.TOTVEGC.203501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat
        lon = ds.lon
        TOTVEGCFeedback_ungrouped           = ds['TOTVEGC']
        ds.close()
        
        ## annual mean in pfrost, gC/m2
        TOTVEGCFeedback[ens[i]]           = TOTVEGCFeedback_ungrouped.groupby('time.year').mean(
                                                                    dim='time', skipna=True)
        TOTVEGCPermafrostFeedback[ens[i]] = TOTVEGCFeedback[ens[i]].where(
                                                ~np.isnan(pfrostFeedback[ens[i]]))
        # only in permafrost peatland
        TOTVEGCPeatFeedback[ens[i]]       = TOTVEGCFeedback[ens[i]].where(
                                                ~np.isnan(pfrostInPeatlandFeedback[ens[i]]))
        
        ## annual mean in pfrost, gC
        TOTVEGCFeedback_gC[ens[i]]           = (TOTVEGCFeedback_ungrouped.groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea[20:,:]
        TOTVEGCPermafrostFeedback_gC[ens[i]] = TOTVEGCFeedback_gC[ens[i]].where(
                                                ~np.isnan(pfrostControl2034[ens[i]]))
        # only in permafrost peatland
        TOTVEGCPeatFeedback_gC[ens[i]]       = TOTVEGCFeedback_gC[ens[i]].where(
                                                ~np.isnan(pfrostInPeatlandControl2034[ens[i]]))
        
            
    ''' FIGURES'''
    #### time series in permafrost soils
    ensMembersTOTECOSYSCCONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPermafrostControl)
    ensMeanTOTECOSYSCCONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCONTROL_ts, numEns)
    
    ensMembersTOTECOSYSCFEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPermafrostFeedback)
    ensMeanTOTECOSYSCFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCFEEDBACK_ts, numEns)
    
    ensMembersTOTECOSYSCCONTROL_gC_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPermafrostControl_gC)
    ensMeanTOTECOSYSCCONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCONTROL_gC_ts, numEns)
    
    ensMembersTOTECOSYSCFEEDBACK_gC_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPermafrostFeedback_gC)
    ensMeanTOTECOSYSCFEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCFEEDBACK_gC_ts, numEns)
    
    ensMembersTOTVEGCCONTROL_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPermafrostControl)
    ensMeanTOTVEGCCONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGCCONTROL_ts, numEns)
    
    ensMembersTOTVEGCFEEDBACK_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPermafrostFeedback)
    ensMeanTOTVEGCFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGCFEEDBACK_ts, numEns)
    
    ensMembersTOTVEGCCONTROL_gC_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPermafrostControl_gC)
    ensMeanTOTVEGCCONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGCCONTROL_gC_ts, numEns)
    
    ensMembersTOTVEGCFEEDBACK_gC_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPermafrostFeedback_gC)
    ensMeanTOTVEGCFEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGCFEEDBACK_gC_ts, numEns)
    
    ensMembersTOTSoilCCONTROL_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPermafrostControl)
    ensMeanTOTSoilCCONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilCCONTROL_ts, numEns)
    
    ensMembersTOTSoilCFEEDBACK_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPermafrostFeedback)
    ensMeanTOTSoilCFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilCFEEDBACK_ts, numEns)
    
    ensMembersTOTSoilCCONTROL_gC_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPermafrostControl_gC)
    ensMeanTOTSoilCCONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilCCONTROL_gC_ts, numEns)
    
    ensMembersTOTSoilCFEEDBACK_gC_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPermafrostFeedback_gC)
    ensMeanTOTSoilCFEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilCFEEDBACK_gC_ts, numEns)
    
    
    ## peatland pfrost
    ensMembersTOTECOSYSC_PEAT_CONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatControl)
    ensMeanTOTECOSYSC_PEAT_CONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_CONTROL_ts, numEns)
    
    ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatFeedback)
    ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts, numEns)
    
    ensMembersTOTECOSYSC_PEAT_CONTROL_gC_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatControl_gC)
    ensMeanTOTECOSYSC_PEAT_CONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_CONTROL_gC_ts, numEns)
    
    ensMembersTOTECOSYSC_PEAT_FEEDBACK_gC_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatFeedback_gC)
    ensMeanTOTECOSYSC_PEAT_FEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_FEEDBACK_gC_ts, numEns)
    
    ensMembersTOTSoilC_PEAT_CONTROL_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPeatControl)
    ensMeanTOTSoilC_PEAT_CONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilC_PEAT_CONTROL_ts, numEns)
    
    ensMembersTOTSoilC_PEAT_FEEDBACK_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPeatFeedback)
    ensMeanTOTSoilC_PEAT_FEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilC_PEAT_FEEDBACK_ts, numEns)
    
    ensMembersTOTSoilC_PEAT_CONTROL_gC_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPeatControl_gC)
    ensMeanTOTSoilC_PEAT_CONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilC_PEAT_CONTROL_gC_ts, numEns)
    
    ensMembersTOTSoilC_PEAT_FEEDBACK_gC_ts = make_timeseries(numEns,'TOTSoilC',lat,lon,90,49.5,360,0,TOTSoilCPeatFeedback_gC)
    ensMeanTOTSoilC_PEAT_FEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTSoilC_PEAT_FEEDBACK_gC_ts, numEns)
    
    ensMembersTOTVEGC_PEAT_CONTROL_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPeatControl)
    ensMeanTOTVEGC_PEAT_CONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGC_PEAT_CONTROL_ts, numEns)
    
    ensMembersTOTVEGC_PEAT_FEEDBACK_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPeatFeedback)
    ensMeanTOTVEGC_PEAT_FEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGC_PEAT_FEEDBACK_ts, numEns)
    
    ensMembersTOTVEGC_PEAT_CONTROL_gC_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPeatControl_gC)
    ensMeanTOTVEGC_PEAT_CONTROL_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGC_PEAT_CONTROL_gC_ts, numEns)
    
    ensMembersTOTVEGC_PEAT_FEEDBACK_gC_ts = make_timeseries(numEns,'TOTVEGC',lat,lon,90,49.5,360,0,TOTVEGCPeatFeedback_gC)
    ensMeanTOTVEGC_PEAT_FEEDBACK_gC_ts = make_ensemble_mean_timeseries(ensMembersTOTVEGC_PEAT_FEEDBACK_gC_ts, numEns)
    
    
    #### Fig. S1a all permafrost per square meter - change from 2035
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]] - ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]] - ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTECOSYSCCONTROL_ts - ensMeanTOTECOSYSCCONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTECOSYSCFEEDBACK_ts - ensMeanTOTECOSYSCFEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([-12,27])
    plt.ylabel('$\Delta$ ecosystem C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('a) $\Delta$ Total ecosystem C from permafrost (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS1a_nh_annual_mean_change_in_TOTECOSYSC_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 6a total over entire 2034 pf domain
    fig = plt.figure(figsize=(9,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTECOSYSCCONTROL_gC_ts[ens[ensNum]] - ensMembersTOTECOSYSCCONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTECOSYSCFEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTECOSYSCFEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTECOSYSCCONTROL_gC_ts - ensMeanTOTECOSYSCCONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTECOSYSCFEEDBACK_gC_ts - ensMeanTOTECOSYSCFEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-1.7*1e13),(2.5*1e13)])
    plt.ylabel('$\Delta$ ecosystem gC', fontsize=12)
    plt.title('a) $\Delta$ Total ecosystem C from permafrost', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig6a_nh_annual_mean_change_in_TOTECOSYSC_in_2034_pf_domain_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
   
    
    #### Fig. S1b total soil C
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTSoilCCONTROL_ts[ens[ensNum]] - ensMembersTOTSoilCCONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTSoilCFEEDBACK_ts[ens[ensNum]] - ensMembersTOTSoilCFEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTSoilCCONTROL_ts - ensMeanTOTSoilCCONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTSoilCFEEDBACK_ts - ensMeanTOTSoilCFEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([-12,27])
    plt.ylabel('$\Delta$ permafrost soil C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('b) $\Delta$ Total soil C from permafrost (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS1b_nh_annual_mean_change_in_TOTSoilC_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 6b total soil C entire pf domain
    fig = plt.figure(figsize=(9,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTSoilCCONTROL_gC_ts[ens[ensNum]] - ensMembersTOTSoilCCONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTSoilCFEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTSoilCFEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTSoilCCONTROL_gC_ts - ensMeanTOTSoilCCONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTSoilCFEEDBACK_gC_ts - ensMeanTOTSoilCFEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-1.7*1e13),(2.5*1e13)])
    plt.ylabel('$\Delta$ permafrost soil gC', fontsize=12)
    plt.title('b) $\Delta$ Total soil C from permafrost', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig6b_nh_annual_mean_change_in_TOTSoilC_in_2034_pf_domain_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig

    
    
    #### Fig. S1c total veg C per area
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTVEGCCONTROL_ts[ens[ensNum]] - ensMembersTOTVEGCCONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTVEGCFEEDBACK_ts[ens[ensNum]] - ensMembersTOTVEGCFEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTVEGCCONTROL_ts - ensMeanTOTVEGCCONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTVEGCFEEDBACK_ts - ensMeanTOTVEGCFEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); #plt.ylim([-12,27])
    plt.ylabel('$\Delta$ vegetation C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('c) $\Delta$ Total vegetation C from permafrost (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS1c_nh_annual_mean_change_in_TOTVEGC_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 6c total veg C entire pf domain
    fig = plt.figure(figsize=(9,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTVEGCCONTROL_gC_ts[ens[ensNum]] - ensMembersTOTVEGCCONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTVEGCFEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTVEGCFEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTVEGCCONTROL_gC_ts - ensMeanTOTVEGCCONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTVEGCFEEDBACK_gC_ts - ensMeanTOTVEGCFEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-1.7*1e13),(2.5*1e13)])
    plt.ylabel('$\Delta$ vegetation gC', fontsize=12)
    plt.title('c) $\Delta$ Total vegetation C from permafrost', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig6c_nh_annual_mean_change_in_TOTVEGC_in_2034_pf_domain_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
        

    
    #### Fig. S2a from peat - change from 2035
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]] - ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]] - ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTECOSYSC_PEAT_CONTROL_ts - ensMeanTOTECOSYSC_PEAT_CONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5 permafrost peat')
    plt.plot((ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts - ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5 permafrost peat')
    # plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([-12,27])
    plt.ylabel('$\Delta$ ecosystem C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('a) $\Delta$ Total ecosystem C from permafrost peatland (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS2a_nh_annual_mean_change_in_TOTECOSYSC_PEATLAND_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 7a total over entire 2035 pf domain
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTECOSYSC_PEAT_CONTROL_gC_ts[ens[ensNum]] - ensMembersTOTECOSYSC_PEAT_CONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTECOSYSC_PEAT_FEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTECOSYSC_PEAT_FEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTECOSYSC_PEAT_CONTROL_gC_ts - ensMeanTOTECOSYSC_PEAT_CONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTECOSYSC_PEAT_FEEDBACK_gC_ts - ensMeanTOTECOSYSC_PEAT_FEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-2.3*1e13),(2.9*1e13)])
    plt.ylabel('$\Delta$ ecosystem gC', fontsize=11)
    plt.title('a) $\Delta$ Total ecosystem C from permafrost peatland', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig7a_nh_annual_mean_change_in_TOTECOSYSC_PEAT_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    
    
    
    #### Fig. S2b total soil C
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTSoilC_PEAT_CONTROL_ts[ens[ensNum]] - ensMembersTOTSoilC_PEAT_CONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTSoilCFEEDBACK_ts[ens[ensNum]] - ensMembersTOTSoilCFEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTSoilC_PEAT_CONTROL_ts - ensMeanTOTSoilC_PEAT_CONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTSoilCFEEDBACK_ts - ensMeanTOTSoilCFEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([-12,27])
    plt.ylabel('$\Delta$ permafrost soil C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('b) $\Delta$ Total soil C from permafrost peatland (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS2b_nh_annual_mean_change_in_TOTSoilC_PEAT_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 7b total soil C entire pf domain
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTSoilC_PEAT_CONTROL_gC_ts[ens[ensNum]] - ensMembersTOTSoilC_PEAT_CONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTSoilC_PEAT_FEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTSoilC_PEAT_FEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTSoilC_PEAT_CONTROL_gC_ts - ensMeanTOTSoilC_PEAT_CONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTSoilC_PEAT_FEEDBACK_gC_ts - ensMeanTOTSoilC_PEAT_FEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-2.3*1e13),(2.9*1e13)])
    plt.ylabel('$\Delta$ permafrost soil gC', fontsize=11)
    plt.title('b) $\Delta$ Total soil C from permafrost peatland', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig7b_nh_annual_mean_change_in_TOTSoilC_2PEAT_035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    
    
    #### Fig. S2c total veg C per area
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTVEGC_PEAT_CONTROL_ts[ens[ensNum]] - ensMembersTOTVEGC_PEAT_CONTROL_ts[ens[ensNum]][0])/1000.,
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTVEGCFEEDBACK_ts[ens[ensNum]] - ensMembersTOTVEGCFEEDBACK_ts[ens[ensNum]][0])/1000.,
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTVEGC_PEAT_CONTROL_ts - ensMeanTOTVEGC_PEAT_CONTROL_ts[0])/1000.,
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTVEGCFEEDBACK_ts - ensMeanTOTVEGCFEEDBACK_ts[0])/1000.,
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); #plt.ylim([-12,27])
    plt.ylabel('$\Delta$ vegetation C (kg $\mathregular{m^{-2}}$)', fontsize=11)
    plt.title('c) $\Delta$ Total vegetation C from permafrost peatland (per $\mathregular{m^{-2}})$', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/FigS2c_nh_annual_mean_change_in_TOTVEGC_PEAT_per_area_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    #### Fig. 7c total veg C entire pf domain
    fig = plt.figure(figsize=(12,5))
    for ensNum in range(len(ens)):
        plt.plot((ensMembersTOTVEGC_PEAT_CONTROL_gC_ts[ens[ensNum]] - ensMembersTOTVEGC_PEAT_CONTROL_gC_ts[ens[ensNum]][0]),
                 linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        plt.plot((ensMembersTOTVEGC_PEAT_FEEDBACK_gC_ts[ens[ensNum]] - ensMembersTOTVEGC_PEAT_FEEDBACK_gC_ts[ens[ensNum]][0]),
                 color='xkcd:sky blue',linewidth=0.9)
    plt.plot((ensMeanTOTVEGC_PEAT_CONTROL_gC_ts - ensMeanTOTVEGC_PEAT_CONTROL_gC_ts[0]),
                 color='xkcd:scarlet',linestyle='dashed',linewidth=3,
                 label='SSP2-4.5')
    plt.plot((ensMeanTOTVEGC_PEAT_FEEDBACK_gC_ts - ensMeanTOTVEGC_PEAT_FEEDBACK_gC_ts[0]),
                  color='xkcd:blue',linewidth=3,
                  label='ARISE-SAI-1.5')
    plt.legend(fancybox=True, fontsize=13)
    plt.axhline(y = 0, color = 'k', linewidth=0.7, linestyle = 'dotted')
    plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
    plt.xlim([0,34]); plt.ylim([(-2.3*1e13),(2.9*1e13)])
    plt.ylabel('$\Delta$ vegetation gC', fontsize=11)
    plt.title('c) $\Delta$ Total vegetation C from permafrost peatland', 
              fontsize=14, fontweight='bold')
    plt.savefig(figureDir + '/Fig7c_nh_annual_mean_change_in_TOTVEGC_PEAT_2035-2069.pdf',
                bbox_inches='tight',dpi=1200)
    del fig
    
    
    return 



def permafrostExtentTimeseries(dataDir, figureDir):
    from tippingPoints import findPermafrost
    from make_timeseries import make_timeseries, make_ensemble_mean_timeseries, make_permafrost_extent_timeseries
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    ens = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010']
    
    
    ## get soil permafrost (exclude bedrock permafrost)
    altAnnMeanCONTROL,altmaxMonthlyCONTROL,altmaxAnnCONTROL,lat,lon = findPermafrost(
        'b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.','201501-206912')
    altAnnMeanFEEDBACK,altmaxMonthlyFEEDBACK,altmaxAnnFEEDBACK,lat,lon = findPermafrost(
        'b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.', '203501-206912')
    
    #### extent time series - read in km^2
    ensMeanSoilPfrostARISE_ts, ensMembersSoilPfrostARISE_ts, ensMembersSoilPfrostVolumeCONTROL, ensMembersSoilPfrostVolumeFEEDBACK = make_permafrost_extent_timeseries(
                                                                                dataDir,
                                                                                10,lat,lon,
                                                                                altAnnMeanFEEDBACK)
    ensMeanSoilPfrostSSP_ts, ensMembersSoilPfrostSSP_ts, ensMembersSoilPfrostVolumeCONTROL, ensMembersSoilPfrostVolumeFEEDBACK  = make_permafrost_extent_timeseries(
                                                                                dataDir,
                                                                                10,lat,lon,
                                                                                altAnnMeanCONTROL)
    
    
    #### Fig. 1a permafrost extent
    fig, ax = plt.subplots(1,1, figsize=(9,5), dpi=1200)
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),ensMembersSoilPfrostSSP_ts[ens[i]]/1e6,color='xkcd:pale red',
                label='SSP2-4.5', linestyle='--', linewidth=0.9)
        ax.plot(np.linspace(2035,2069,35),ensMembersSoilPfrostARISE_ts[ens[i]]/1e6,color='xkcd:sky blue',
                label='ARISE-SAI-1.5', linewidth=0.9)
    ax.plot(np.linspace(2015,2069,55),ensMeanSoilPfrostSSP_ts/1e6,color='xkcd:scarlet',label='SSP2-4.5',linewidth=3,linestyle='--')
    ax.plot(np.linspace(2035,2069,35),ensMeanSoilPfrostARISE_ts/1e6,color='xkcd:blue',label='ARISE-SAI-1.5',linewidth=3)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:pale red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:blue', lw=2)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    # ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    plt.ylabel('Total extent (km$^2$ x 10$^6$)', fontsize=12)
    plt.title('a) Annual maximum permafrost extent', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(figureDir + '/Fig1a_annual_maximum_soil_permafrost_extent.jpg',
                dpi=1200,bbox_inches='tight')
    
    
    #### Fig. 1b permafrost volume
    from tippingPoints import permafrostVolume
    ensMembersPfrostVolumeARISE_ts,ensMembersPfrostVolumeSSP_ts,ensMeanPfrostVolumeARISE_ts,ensMeanPfrostVolumeSSP_ts = permafrostVolume()
    
    fig, ax = plt.subplots(1,1, figsize=(9,5), dpi=1200)
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),ensMembersPfrostVolumeSSP_ts[ens[i]],color='xkcd:pale red',
                label='SSP2-4.5',linestyle='--', linewidth=0.9)
        ax.plot(np.linspace(2035,2069,35),ensMembersPfrostVolumeARISE_ts[ens[i]],color='xkcd:sky blue',
                label='ARISE-SAI-1.5', linewidth=0.9)
    ax.plot(np.linspace(2015,2069,55),ensMeanPfrostVolumeSSP_ts,color='xkcd:scarlet',label='SSP2-4.5',linewidth=3,linestyle='--')
    ax.plot(np.linspace(2035,2069,35),ensMeanPfrostVolumeARISE_ts,color='xkcd:blue',label='ARISE-SAI-1.5',linewidth=3)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:pale red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:blue', lw=2)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    ax.set_yticks([33,34,35,36,37])
    ax.set_yticklabels(['33','34','35','36','37'])
    plt.ylabel('Upper bound on volume (km$^3$)', fontsize=12)
    plt.title('b) Annual mean permafrost volume', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(figureDir + '/Fig1b_annual_mean_permafrost_volume_timeseries.jpg',
                dpi=1200,bbox_inches='tight')
    
    
    #### Fig. 1c permafrost temperature (weighted mean temperature between bottom of ALT and top of bedrock)
    import pickle
    with open(dataDir + '/permafrost_temperature_FEEDBACK.pkl', 'rb') as fp:
        pfrostTempFEEDBACK = pickle.load(fp)
    with open(dataDir + '/permafrost_temperature_CONTROL.pkl', 'rb') as fp:
        pfrostTempCONTROL = pickle.load(fp)
    
    ensMembersPfrostTempControl_ts = make_timeseries(10,'ALT',lat,lon,91,50,360,-1,pfrostTempCONTROL)
    ensMeanPfrostTempCONTROL_ts = make_ensemble_mean_timeseries(ensMembersPfrostTempControl_ts, 10)
    ensMembersPfrostTempFeedback_ts = make_timeseries(10,'ALT',lat,lon,91,50,360,-1,pfrostTempFEEDBACK)
    ensMeanPfrostTempFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersPfrostTempFeedback_ts, 10)
    
    fig, ax = plt.subplots(1,1, figsize=(9,5), dpi=1200)
    for i in range(len(ens)):
        ax.plot(np.linspace(2015,2069,55),ensMembersPfrostTempControl_ts[ens[i]]-273.15,color='xkcd:pale red',
                label='SSP2-4.5',linestyle='--', linewidth=0.9)
        ax.plot(np.linspace(2035,2069,35),ensMembersPfrostTempFeedback_ts[ens[i]]-273.15,color='xkcd:sky blue',
                label='ARISE-SAI-1.5', linewidth=0.9)
    ax.plot(np.linspace(2015,2069,55),ensMeanPfrostTempCONTROL_ts-273.15,color='xkcd:scarlet',label='SSP2-4.5',linewidth=3,linestyle='--')
    ax.plot(np.linspace(2035,2069,35),ensMeanPfrostTempFEEDBACK_ts-273.15,color='xkcd:blue',label='ARISE-SAI-1.5',linewidth=3)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='xkcd:pale red', lw=2, linestyle='--'),
                    Line2D([0], [0], color='xkcd:blue', lw=2)]
    plt.legend(custom_lines, ['SSP2-4.5','ARISE-SAI-1.5'], fancybox=True, fontsize=12)
    plt.xlim([2015,2069])
    plt.xlim([2015,2069])
    ax.set_xticks([2015,2025,2035,2045,2055,2065])
    ax.set_xticklabels(['2015','2025','2035','2045','2055','2065'])
    plt.ylabel('Temperature ($^\circ$C)', fontsize=12)
    plt.title('c) Annual mean permafrost temperature', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.savefig(figureDir + '/Fig1c_annual_mean_permafrost_temperature_timeseries.jpg',
                dpi=1200, bbox_inches='tight')
    
    
    return
