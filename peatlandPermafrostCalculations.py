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


def getLandType(makeFigures,dataDir,figureDir):
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
    pfrostFeedback = {}
    for i in range(numEns):
        pfrostControl[ens[i]] = altmaxAnnCONTROL[ens[i]][20:,:,:]
        pfrostFeedback[ens[i]] = altmaxAnnFEEDBACK[ens[i]]
        
    

    #### Fig. 5a peatland map
    import matplotlib as mpl
    pinks = mpl.colormaps['pink_r'].resampled(20)
    fig, ax = make_maps(peatland[20:,:],lat,lon,0,100,20,pinks,'% peatland cover',
                        'a) Fixed peatland area in CESM2','Fig5a_peat_fraction','neither',False,False)
    
    
    ''' Each year, how much permafrost is in peatland? '''
    years                      = np.linspace(2035,2069,35)
    pfrostInPeatlandFeedback   = {}
    pfrostInPeatlandControl    = {}
    peatlandPfrostFeedbackArea = np.zeros((numEns,35)) * np.nan
    totalPfrostFeedbackArea    = np.zeros((numEns,35)) * np.nan
    peatlandPfrostControlArea  = np.zeros((numEns,35)) * np.nan
    totalPfrostControlArea     = np.zeros((numEns,35)) * np.nan
    
    for i in range(numEns):
        pfrostInPeatlandFeedback[ens[i]] = np.zeros((35,pfrostCONTROL[ens[0]].shape[1],pfrostCONTROL[ens[0]].shape[2])) * np.nan
        pfrostInPeatlandControl[ens[i]]  = np.zeros((35,pfrostCONTROL[ens[0]].shape[1],pfrostCONTROL[ens[0]].shape[2])) * np.nan
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
    
    ####################################
    ## CARBON OUT OF PERMAFROST SOILS ##
    ####################################
    '''
    Cumulative annual total respiration from 2035 to 2069 to see if model can 
    detect a difference in 'irreversible' soil carbon loss
    1. Multiply monthly mean rate by days in month
    2. Multiply by seconds per day to get total land emissions per month
    3. Sum over year to get annual land emissions
    4. Cumulative sum over time
    '''
        
    #### TOTECOSYSC
    '''total ecosystem carbon - control '''
    TOTECOSYSCControl         = {}
    TOTECOSYSCPeatControl     = {}
    TOTECOSYSCControl_all     = {}
    TOTECOSYSCCumControl      = {}
    TOTECOSYSCCumPeatControl  = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BWSSP245cmip6.f09_g17.CMIP6-SSP2-4.5-WACCM.' + str(ens[i]) +
                             '.clm2.h0.TOTECOSYSC.201501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat; lon = ds.lon; 
        
        TOTECOSYSCControl[ens[i]]        = ds['TOTECOSYSC'][240:,41:,:]
        
        # annual gC/m2 - summed
        TOTECOSYSCControl_all[ens[i]]    = ((TOTECOSYSCControl[ens[i]].groupby('time.year').mean(
                                                                    dim='time', skipna=True))*gridArea).cumsum(axis=0,skipna=True)
        TOTECOSYSCControl_all[ens[i]]    = TOTECOSYSCControl_all[ens[i]].cumsum(axis=0,skipna=True)#.where(
                                                        #~np.isnan(pfrostControl[ens[i]))
        
        # # ## cumulative since 2035
        # TOTECOSYSCCumControl[ens[i]]     = TOTECOSYSCControl[ens[i]].cumsum(
        #                                                 axis=0)
        # TOTECOSYSCCumControl[ens[i]]     = TOTECOSYSCCumControl[ens[i]].where(
        #                                                       ~np.isnan(pfrostControl[ens[i]]))
        # TOTECOSYSCCumPeatControl[ens[i]] = TOTECOSYSCCumControl[ens[i]].where(
        #                                                 ~np.isnan(pfrostInPeatlandControl[ens[i]]))
        ## annual mean in pfrost
        TOTECOSYSCControl[ens[i]]        = (TOTECOSYSCControl[ens[i]].groupby('time.year').mean(
                                                                    dim='time', skipna=True))
        TOTECOSYSCControl[ens[i]]        = TOTECOSYSCControl[ens[i]].where(
                                                        ~np.isnan(pfrostControl[ens[i]]))
        TOTECOSYSCPeatControl[ens[i]]    = TOTECOSYSCControl[ens[i]].where(
                                                        ~np.isnan(pfrostInPeatlandControl[ens[i]]))
        ds.close()
        
        
    '''total ecosystem carbon from permafrost soils - FEEDBACK'''
    TOTECOSYSCFeedback        = {}
    TOTECOSYSCFeedback_all    = {}
    TOTECOSYSCPeatFeedback    = {}
    TOTECOSYSCCumFeedback     = {}
    TOTECOSYSCCumPeatFeedback = {}
    for i in range(len(ens)):
        ds = xr.open_dataset(dataDir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.' + str(ens[i]) +
                             '.clm2.h0.TOTECOSYSC.203501-206912_NH.nc', decode_times=False)
        units, reference_date = ds.time.attrs['units'].split('since')
        ds['time'] = pd.date_range(start=reference_date, periods=ds.sizes['time'], freq='MS')
        lat = ds.lat
        lon = ds.lon
        TOTECOSYSCFeedback[ens[i]]        = ds['TOTECOSYSC'][:,41:,:]
        ds.close()
        # annual gC/m2
        TOTECOSYSCFeedback_all[ens[i]]    = (TOTECOSYSCFeedback[ens[i]].groupby('time.year').sum(
                                                                    dim='time', skipna=True))*gridArea
        ## cumulative since 2035
        TOTECOSYSCCumFeedback[ens[i]]     = TOTECOSYSCFeedback[ens[i]].cumsum(
                                                                axis=0)
        TOTECOSYSCCumFeedback[ens[i]]     = TOTECOSYSCCumFeedback[ens[i]].where(
                                                              ~np.isnan(pfrostFeedback[ens[i]]))
        TOTECOSYSCCumPeatFeedback[ens[i]] = TOTECOSYSCCumFeedback[ens[i]].where(
                                                              ~np.isnan(pfrostInPeatlandFeedback[ens[i]]))
        ## annual mean in pfrost 
        TOTECOSYSCFeedback[ens[i]]        = (TOTECOSYSCFeedback[ens[i]].groupby('time.year').mean(
                                                                    dim='time', skipna=True))
        TOTECOSYSCFeedback[ens[i]]        = TOTECOSYSCFeedback[ens[i]].where(
                                                              ~np.isnan(pfrostFeedback[ens[i]]))
        TOTECOSYSCPeatFeedback[ens[i]]    = TOTECOSYSCFeedback[ens[i]].where(
                                                              ~np.isnan(pfrostInPeatlandFeedback[ens[i]]))
        
        
    
    #### figures 
    ''' FIGURES'''
    if makeFigures:
        #### TOTECOSYSC
        ensMembersTOTECOSYSCCumCONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCCumControl)
        ensMeanTOTECOSYSCCumCONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCumCONTROL_ts, numEns)
        ensMembersTOTECOSYSCCumFEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCCumFeedback)
        ensMeanTOTECOSYSCCumFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCumFEEDBACK_ts, numEns)
        ensMembersTOTECOSYSCCum_PEAT_CONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCCumPeatControl)
        ensMeanTOTECOSYSCCum_PEAT_CONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCum_PEAT_CONTROL_ts, numEns)
        ensMembersTOTECOSYSCCum_PEAT_FEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCCumPeatFeedback)
        ensMeanTOTECOSYSCCum_PEAT_FEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCum_PEAT_FEEDBACK_ts, numEns)
        
        # ## ---- cumulative carbon per square meter from permafrost soils ---- ##
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot(ensMembersTOTECOSYSCCumCONTROL_ts[ens[ensNum]]/1000.,
        #              color='xkcd:pale red',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCCumFEEDBACK_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCCum_PEAT_CONTROL_ts[ens[ensNum]]/1000.,
        #              linestyle='dotted',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCCum_PEAT_FEEDBACK_ts[ens[ensNum]]/1000.,
        #              linestyle='dotted',color='xkcd:sky blue',linewidth=0.9)
        # plt.plot(ensMeanTOTECOSYSCCumCONTROL_ts/1000.,
        #              color='xkcd:scarlet',linewidth=2.5,
        #              label='SSP2-4.5 total permafrost')
        # plt.plot(ensMeanTOTECOSYSCCumFEEDBACK_ts/1000.,
        #              color='xkcd:blue',linewidth=2.5,
        #              label='ARISE-SAI-1.5 total permafrost')
        # plt.plot(ensMeanTOTECOSYSCCum_PEAT_CONTROL_ts/1000.,
        #              color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #              label='SSP2-4.5 permafrost peat')
        # plt.plot(ensMeanTOTECOSYSCCum_PEAT_FEEDBACK_ts/1000.,
        #               color='xkcd:blue',linestyle='dashed',linewidth=2.5,
        #               label='ARISE-SAI-1.5 permafrost peat')
        # plt.legend(fancybox=True, fontsize=13, loc=(0.12,-0.275), ncol=2)
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); plt.ylim(bottom=0)
        # plt.ylabel('Permafrost ecosystem C (kgC $\mathregular{m^{-2}}$)', fontsize=11)
        # plt.title('a) Cumulative total ecosystem C from permafrost region (per $\mathregular{m^{-2}})$', fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + '/nh_cumulative_TOTECOSYSC_per_area_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        
        ensMembersTOTECOSYSCCONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCControl)
        ensMeanTOTECOSYSCCONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCONTROL_ts, numEns)
        ensMembersTOTECOSYSCFEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCFeedback)
        ensMeanTOTECOSYSCFEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCFEEDBACK_ts, numEns)
        ensMembersTOTECOSYSC_PEAT_CONTROL_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatControl)
        ensMeanTOTECOSYSC_PEAT_CONTROL_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_CONTROL_ts, numEns)
        ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts = make_timeseries(numEns,'TOTECOSYSC',lat,lon,90,49.5,360,0,TOTECOSYSCPeatFeedback)
        ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts, numEns)
        
        ## ---- annual mean carbon per square meter from permafrost soils ---- ##
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot(ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]]/1000.,
        #              color='xkcd:pale red',linestyle='dashed',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        # plt.plot(ensMeanTOTECOSYSCCONTROL_ts/1000.,
        #              color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #              label='SSP2-4.5')
        # plt.plot(ensMeanTOTECOSYSCFEEDBACK_ts/1000.,
        #              color='xkcd:blue',linewidth=2.5,
        #              label='ARISE-SAI-1.5')
        # plt.legend(fancybox=True, fontsize=14, loc='lower right')
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); 
        # plt.ylabel('Permafrost ecosystem C (kgC $\mathregular{m^{-2}}$)', fontsize=11)
        # plt.title('a) Total ecosystem C from permafrost (per $\mathregular{m^{-2}})$', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + '/nh_annual_mean_TOTECOSYSC_per_area_summed_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        #### Fig. 6a all permafrost per square meter - change from 2035
        fig = plt.figure(figsize=(12,5),dpi=1200)
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
        plt.ylabel('$\Delta$ permafrost soil C (kg $\mathregular{m^{-2}}$)', fontsize=11)
        plt.title('a) $\Delta$ Total ecosystem C from permafrost (per $\mathregular{m^{-2}})$', 
                  fontsize=14, fontweight='bold')
        plt.savefig(figureDir + '/Fig6a_nh_annual_mean_change_in_TOTECOSYSC_per_area_2035-2069.jpg',
                    bbox_inches='tight',dpi=1200)
        del fig
        
        # ## total carbon all permafrost - change from 2035
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot((ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]]*totalPfrostControlArea[ensNum,:] - ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]][0]*totalPfrostControlArea[ensNum,0])/1e12,
        #              linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot((ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]]*totalPfrostFeedbackArea[ensNum,:] - ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]][0]*totalPfrostFeedbackArea[ensNum,0])/1e12,
        #               color='xkcd:sky blue',linewidth=0.9)
        # plt.plot((ensMeanTOTECOSYSCCONTROL_ts*pfrostEnsMeanControlArea - ensMeanTOTECOSYSCCONTROL_ts[0]*pfrostEnsMeanControlArea[0])/1e12,
        #               color='xkcd:scarlet',linestyle='dashed',linewidth=3,
        #               label='SSP2-4.5')
        # plt.plot((ensMeanTOTECOSYSCFEEDBACK_ts*pfrostEnsMeanFeedbackArea - ensMeanTOTECOSYSCFEEDBACK_ts[0]*pfrostEnsMeanFeedbackArea[0])/1e12,
        #               color='xkcd:blue',linewidth=3,
        #               label='ARISE-SAI-1.5')
        # plt.legend(fancybox=True, fontsize=13, loc='lower left')
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); plt.ylim([-1.9,0.15])
        # plt.ylabel('$\Delta$ permafrost soil C (Tg)', fontsize=11)
        # plt.title('a) $\Delta$ Total ecosystem C from permafrost', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + '/nh_annual_mean_change_in_TOTECOSYSC_summed_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        # ## from peat
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot(ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]]/1000.,
        #              linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        # plt.plot(ensMeanTOTECOSYSC_PEAT_CONTROL_ts/1000.,
        #              color='xkcd:scarlet',linestyle='dashed',linewidth=3,
        #              label='SSP2-4.5 permafrost peat')
        # plt.plot(ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts/1000.,
        #               color='xkcd:blue',linewidth=3,
        #               label='ARISE-SAI-1.5 permafrost peat')
        # # plt.legend(fancybox=True, fontsize=13)
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); plt.ylim([45,130])
        # plt.ylabel('Permafrost ecosystem C (kgC $\mathregular{m^{-2}}$)', fontsize=11)
        # plt.title('b) Total ecosystem C from permafrost peatland (per $\mathregular{m^{-2}})$', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + '/nh_annual_mean_TOTECOSYSC_PEATLAND_per_area_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        #### Fig. 6b from peat - change from 2035
        fig = plt.figure(figsize=(12,5),dpi=1200)
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
        plt.ylabel('$\Delta$ permafrost peat C (kg $\mathregular{m^{-2}}$)', fontsize=11)
        plt.title('b) $\Delta$ Total ecosystem C from permafrost peatland (per $\mathregular{m^{-2}})$', 
                  fontsize=14, fontweight='bold')
        plt.savefig(figureDir + '/Fig6b_nh_annual_mean_change_in_TOTECOSYSC_PEATLAND_per_area_2035-2069.jpg',
                    bbox_inches='tight',dpi=1200)
        del fig
        
        
        # ## total carbon peatland - change from 2035
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot((ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]]*peatlandPfrostControlArea[ensNum,:] - ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]][0]*peatlandPfrostControlArea[ensNum,0])/1e12,
        #              linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot((ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]]*peatlandPfrostFeedbackArea[ensNum,:] - ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]][0]*peatlandPfrostFeedbackArea[ensNum,0])/1e12,
        #               color='xkcd:sky blue',linewidth=0.9)
        # plt.plot((ensMeanTOTECOSYSC_PEAT_CONTROL_ts*peatlandEnsMeanControlArea - ensMeanTOTECOSYSC_PEAT_CONTROL_ts[0]*peatlandEnsMeanControlArea[0])/1e12,
        #               color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #               label='SSP2-4.5')
        # plt.plot((ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts*peatlandEnsMeanFeedbackArea - ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts[0]*peatlandEnsMeanFeedbackArea[0])/1e12,
        #               color='xkcd:blue',linewidth=2.5,
        #               label='ARISE-SAI-1.5')
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); #plt.ylim([-1.9,0.15])
        # plt.ylabel('$\Delta$ permafrost soil C (Tg)', fontsize=11)
        # plt.title('b) $\Delta$ Peatland ecosystem C from permafrost', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + '/nh_annual_mean_change_in_TOTECOSYSC_PEATLAND_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        
        # ## in talik
        # import pickle
        # with open(dataDir + '/talikAnnFEEDBACK.pkl', 'rb') as fp:
        #     talikAnnFEEDBACK = pickle.load(fp)
        # with open(dataDir + '/talikAnnCONTROL.pkl', 'rb') as fp:
        #     talikAnnCONTROL = pickle.load(fp)
           
        # '''control'''
        # TOTECOSYSC_mask = {}
        # talikAnnCONTROL_mask = talikAnnCONTROL.copy()
        # for i in range(len(ens)):
        #     print(i)
        #     TOTECOSYSC_mask[ens[i]] = np.empty((35,44,288)) * np.nan
        #     talikAnnCONTROL_mask[ens[i]][talikAnnCONTROL_mask[ens[i]] <=20] = 20
        #     # first year = only where there's talik from 2015-2035, otherwise replace with nan
        #     for iyear in range(35): 
        #         TOTECOSYSC_mask[ens[i]][iyear,:,:] = np.where(
        #                                     (talikAnnCONTROL_mask[ens[i]] == iyear+20), 
        #                                     TOTECOSYSCControl[ens[i]][iyear,:,:], np.nan)
                
        # # ensMembersTOTECOSYSCCONTROL_talik_ts = {}
        # cumulativeEcoC = {}
        # for i in range(len(ens)):
        #     cumulativeEcoC[ens[i]] = np.nancumsum(TOTECOSYSC_mask[ens[i]],axis=0)
        #     cumulativeEcoC[ens[i]][cumulativeEcoC[ens[i]] == 0] = np.nan
        
        # ensMembersTOTECOSYSCCONTROL_talik_ts = make_timeseries(10,'TOTECOSYSC',lat,lon,
        #                                                                90,49.5,360,0,
        #                                                                cumulativeEcoC)
        # del TOTECOSYSC_mask, cumulativeEcoC
        
        # '''feedback'''
        # TOTECOSYSC_mask = {}
        # talikAnnFEEDBACK_mask = talikAnnFEEDBACK.copy()
        # for i in range(len(ens)):
        #     print(i)
        #     TOTECOSYSC_mask[ens[i]] = np.empty((35,44,288)) * np.nan
        #     talikAnnFEEDBACK_mask[ens[i]][talikAnnFEEDBACK_mask[ens[i]] <=20] = 20
        #     # first year = only where there's talik from 2015-2035, otherwise replace with nan
        #     for iyear in range(35): 
        #         TOTECOSYSC_mask[ens[i]][iyear,:,:] = np.where(
        #                                     (talikAnnFEEDBACK_mask[ens[i]] == iyear+20), 
        #                                     TOTECOSYSCFeedback[ens[i]][iyear,:,:], np.nan)
                
        # cumulativeEcoC = {}
        # for i in range(len(ens)):
        #     cumulativeEcoC[ens[i]] = np.nancumsum(TOTECOSYSC_mask[ens[i]],axis=0)
        #     cumulativeEcoC[ens[i]][cumulativeEcoC[ens[i]] == 0] = np.nan
        
        # ensMembersTOTECOSYSCFEEDBACK_talik_ts = make_timeseries(10,'TOTECOSYSC',lat,lon,
        #                                                                90,49.5,360,0,
        #                                                                cumulativeEcoC)
        
        
        # ## ensemble mean
        # TOTECOSYSC_mean_control_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCCONTROL_talik_ts, 10)
        # TOTECOSYSC_mean_feedback_ts = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCFEEDBACK_talik_ts, 10)
        
        # TOTECOSYSCmean_control = np.nanmean(np.stack((TOTECOSYSCControl.values())),axis=0)
        # talikAnnCONTROLmean_mask = talikAnnCONTROLmean.copy()
        # talikAnnCONTROLmean_mask[talikAnnCONTROLmean_mask <= 20] = 20
        # TOTECOSYSCmean_mask = np.empty((35,44,288))
        # for iyear in range(35):
        #     TOTECOSYSCmean_mask[iyear,:,:] = np.where(talikAnnCONTROLmean_mask == iyear+20,
        #                                               TOTECOSYSCmean_control[iyear,:,:], np.nan)
            
        # TOTECOSYSC_mean_control_ts = make_timeseries(10,'TOTECOSYSC',
        #                                              lat,lon,90,49.5,360,0,TOTECOSYSCmean_mask.cumsum(axis=0))
        
        
        # ## figure for TOTECOSYSC in talik
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot(ensMembersTOTECOSYSCCONTROL_talik_ts[ens[ensNum]]/1000.,
        #              color='xkcd:pale red',linestyle='dashed',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCFEEDBACK_talik_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        # plt.plot(TOTECOSYSC_mean_control_ts/1000.,
        #               color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #               label='SSP2-4.5')
        # plt.plot(TOTECOSYSC_mean_feedback_ts/1000.,
        #               color='xkcd:blue',linewidth=2.5,
        #               label='ARISE-SAI-1.5')
        # #plt.legend(fancybox=True, fontsize=14, loc='lower right')
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); #plt.ylim([45,130])
        # plt.ylabel('Total ecosystem C (kgC $\mathregular{m^{-2}}$)', fontsize=11)
        # plt.title('c) Total ecosystem C from talik (per $\mathregular{m^{-2}})$', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + 'nh_annual_mean_TOTECOSYSC_TALIK_per_area_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        # ## change from 2035
        # fig = plt.figure(figsize=(12,5),dpi=1200)
        # for ensNum in range(len(ens)):
        #     plt.plot((ensMembersTOTECOSYSCCONTROL_talik_ts[ens[ensNum]] - ensMembersTOTECOSYSCCONTROL_talik_ts[ens[ensNum]][0])/1000.,
        #              linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot((ensMembersTOTECOSYSCFEEDBACK_talik_ts[ens[ensNum]] - ensMembersTOTECOSYSCFEEDBACK_talik_ts[ens[ensNum]][0])/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        # plt.plot((TOTECOSYSC_mean_control_ts - TOTECOSYSC_mean_control_ts[0])/1000.,
        #              color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #              label='SSP2-4.5 permafrost peat')
        # plt.plot((TOTECOSYSC_mean_feedback_ts - TOTECOSYSC_mean_feedback_ts[0])/1000.,
        #               color='xkcd:blue',linewidth=2.5,
        #               label='ARISE-SAI-1.5 permafrost peat')
        # # plt.legend(fancybox=True, fontsize=13)
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'])
        # plt.xlim([0,34]); plt.ylim([-5,27])
        # plt.ylabel('$\Delta$ talik C (kg $\mathregular{m^{-2}}$)', fontsize=11)
        # plt.title('c) $\Delta$ Total ecosystem C from talik (per $\mathregular{m^{-2}})$', 
        #           fontsize=14, fontweight='bold')
        # plt.savefig(figureDir + 'nh_annual_mean_change_in_TOTECOSYSC_TALIK_per_area_2035-2069.jpg',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        # ## all land types on same figure
        # fig = plt.figure(figsize=(12,6),dpi=1200)
        # for ensNum in range(len(ens)):
        #     '''total'''
        #     plt.plot(ensMembersTOTECOSYSCCONTROL_ts[ens[ensNum]]/1000.,
        #              color='xkcd:pale red',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCFEEDBACK_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linewidth=0.9)
        #     '''peat'''
        #     plt.plot(ensMembersTOTECOSYSC_PEAT_CONTROL_ts[ens[ensNum]]/1000.,
        #              linestyle='dashed',color='xkcd:pale red',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSC_PEAT_FEEDBACK_ts[ens[ensNum]]/1000.,
        #              linestyle='dashed',color='xkcd:sky blue',linewidth=0.9)
        #     '''talik'''
        #     plt.plot(ensMembersTOTECOSYSCCONTROL_talik_ts[ens[ensNum]]/1000.,
        #              color='xkcd:pale red',linestyle='dotted',linewidth=0.9)
        #     plt.plot(ensMembersTOTECOSYSCFEEDBACK_talik_ts[ens[ensNum]]/1000.,
        #              color='xkcd:sky blue',linestyle='dotted',linewidth=0.9)
        # '''total'''
        # plt.plot(ensMeanTOTECOSYSCCONTROL_ts/1000.,
        #              color='xkcd:scarlet',linewidth=2.5,
        #              label='SSP2-4.5 total permafrost')
        # plt.plot(ensMeanTOTECOSYSCFEEDBACK_ts/1000.,
        #              color='xkcd:blue',linewidth=2.5,
        #              label='ARISE-SAI-1.5 total permafrost')
        # '''peat'''
        # plt.plot(ensMeanTOTECOSYSC_PEAT_CONTROL_ts/1000.,
        #              color='xkcd:scarlet',linestyle='dashed',linewidth=2.5,
        #              label='SSP2-4.5 permafrost peat')
        # plt.plot(ensMeanTOTECOSYSC_PEAT_FEEDBACK_ts/1000.,
        #               color='xkcd:blue',linestyle='dashed',linewidth=2.5,
        #               label='ARISE-SAI-1.5 permafrost peat')
        # '''talik'''
        # plt.plot(TOTECOSYSC_mean_control_ts/1000.,
        #               color='xkcd:scarlet',linestyle='dotted',linewidth=2.5,
        #               label='SSP2-4.5 talik')
        # plt.plot(TOTECOSYSC_mean_feedback_ts/1000.,
        #               color='xkcd:blue',linestyle='dotted',linewidth=2.5,
        #               label='ARISE-SAI-1.5 talik')
        
        # plt.legend(fancybox=True, fontsize=13, loc=[-0.015,-0.22], ncol=3)
        # plt.xticks([0,5,10,15,20,25,30,34],['2035','2040','2045','2050','2055','2060','2065','2069'],
        #            fontsize=12)
        # plt.xlim([0,34]); #plt.ylim([45,130])
        # plt.ylabel('Permafrost ecosystem C (kgC $\mathregular{m^{-2}}$)', fontsize=13)
        # plt.title('Total ecosystem C from permafrost (per $\mathregular{m^{-2}})$', 
        #           fontsize=15, fontweight='bold')
        # plt.savefig(figureDir + 'nh_annual_mean_TOTECOSYSC_ALL_LAND_TYPES_per_area_2035-2069.pdf',
        #             bbox_inches='tight',dpi=1200)
        # del fig
        
        
        
        # #### TOTECOSYSC, 50-90N
        # diff_totecocontrol = {}
        # diff_totecofeedback = {}
        # for i in range(len(ens)):
        #     diff_totecocontrol[ens[i]] = TOTECOSYSCControl_all[ens[i]] - TOTECOSYSCControl_all[ens[i]][0,:,:]
        #     diff_totecofeedback[ens[i]] = TOTECOSYSCFeedback_all[ens[i]] - TOTECOSYSCFeedback_all[ens[i]][0,:,:]
            
            
        # ensMembersTOTECOSYSCControl_50_90_ts = make_timeseries(10,'TOTECOSYSC',
        #                                                        lat,lon,90,49.5,360,0,
        #                                                        diff_totecocontrol)
        # ensMembersTOTECOSYSCFeedback_50_90_ts = make_timeseries(10,'TOTECOSYSC',
        #                                                        lat,lon,90,49.5,360,0,
        #                                                        diff_totecofeedback)
        
        # diff_totecocontrol_mean = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCControl_50_90_ts, 10)
        # diff_totecofeedback_mean = make_ensemble_mean_timeseries(ensMembersTOTECOSYSCFeedback_50_90_ts, 10)
        
        # for ensNum in range(len(ens)):
        #     plt.plot((ensMembersTOTECOSYSCControl_50_90_ts[ens[ensNum]])/(1e15),
        #              color='xkcd:pale red')
        #     plt.plot((ensMembersTOTECOSYSCFeedback_50_90_ts[ens[ensNum]])/(1e15),
        #              color='xkcd:sky blue')
        # plt.plot(diff_totecocontrol_mean/1e15,linewidth=2,color='xkcd:dark red')

        
        # peatlandEnsMeanFeedbackArea = make_ensemble_mean_timeseries(peatlandPfrostFeedbackArea, numEns)
        # peatlandEnsMeanControlArea = make_ensemble_mean_timeseries(peatlandPfrostControlArea, numEns)
        # pfrostEnsMeanFeedbackArea = make_ensemble_mean_timeseries(totalPfrostFeedbackArea, numEns)
        # pfrostEnsMeanControlArea = make_ensemble_mean_timeseries(totalPfrostControlArea, numEns)
        
        
        
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
        
    return peatland



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
