#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:45:07 2022

@author: Ariel L. Morrison
"""
import os
os.chdir('/Users/arielmor/Projects/actm-sai-csu/research/arise_arctic_climate')

def get_colormap(levs):
    from matplotlib import cm
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap
    import numpy as np
    #########################################################
    # create discrete colormaps from existing continuous maps
    # first make default discrete blue-red colormap 
    # replace center colors with white at 0
    #########################################################
    ## brown-blue
    brbg = mpl.colormaps['BrBG'].resampled(levs+3)
    newcolors = brbg(np.linspace(0, 1, 256))
    newcolors[120:136, :] = np.array([1, 1, 1, 1])
    brbg_cmap = ListedColormap(newcolors)
    ## blue-red
    bwr = mpl.colormaps['RdBu_r'].resampled(levs+3)
    newcolors = bwr(np.linspace(0, 1, 256))
    newcolors[122:134, :] = np.array([1, 1, 1, 1])
    rdbu_cmap = ListedColormap(newcolors)
    ## rainbow
    jet = mpl.colormaps['turbo'].resampled(levs)
    ## other
    magma = mpl.colormaps['magma'].resampled(levs)
    reds = mpl.colormaps['Reds'].resampled(levs)
    hot = mpl.colormaps['hot'].resampled(levs)
    seismic = mpl.colormaps['seismic'].resampled(levs)
    seismic = ListedColormap(seismic(np.linspace(0.25, 0.75, 128)))
    return brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic


def circleBoundary():
    ## gives polar stereographic maps a circular border
    import numpy as np
    import matplotlib.path as mpath
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle


def landmask():
    import xarray as xr
    import numpy as np
    #########################################################
    # land mask
    #########################################################
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    landmask = ds.landmask
    ds.close()
    
    landMask = landmask.where(np.isnan(landmask))
    landMask = landmask.copy() + 2
    landMask = np.where(~np.isnan(landmask),landMask, 1)
    return landMask



def make_maps(var1,latitude,longitude,vmins,vmaxs,levs,mycmap,label,title,savetitle,extend1,addPath,seaIce):
    from plottingFunctions import get_colormap, landmask
    brbg_cmap,rdbu_cmap,jet,magma,reds,hot,seismic = get_colormap(levs)
    import matplotlib as mpl
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point, add_cyclic
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import numpy as np
    import xarray as xr
    hfont = {'fontname':'Verdana'}
    from matplotlib import colors as c
    cmapLand = c.ListedColormap(['xkcd:gray','none'])
    
    landMask = landmask()
    
    #########################################################
    # land mask
    #########################################################
    datadir = '/Users/arielmor/Desktop/SAI/data/ARISE/data'
    figureDir = '/Users/arielmor/Desktop/SAI/data/ARISE/figures/'
    ds = xr.open_dataset(datadir + '/b.e21.BW.f09_g17.SSP245-TSMLT-GAUSS-DEFAULT.001.clm2.h0.ALT.203501-206912_NH.nc')
    lat = ds.lat; lon2 = ds.lon
    ds.close()
    
    
    #########################################################
    # make single North Pole stereographic filled contour map
    #########################################################
    var,lon = add_cyclic_point(var1,coord=longitude)
    
    ## Create figure
    fig = plt.figure(figsize=(10,8))
    if vmins < 0. and vmaxs > 0.:
        norm1 = mcolors.TwoSlopeNorm(vmin=vmins, vcenter=0, vmax=vmaxs)
    else:
        norm1 = mcolors.Normalize(vmin=vmins, vmax=vmaxs)
        
    ## Create North Pole Stereo projection map with circle boundary
    ax1 = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax1.set_extent([180, -180, 49.5, 90], crs=ccrs.PlateCarree())
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    ax1.set_facecolor('0.8')
    
    ## field to be plotted
    cf1 = ax1.pcolormesh(lon,latitude,var,transform=ccrs.PlateCarree(), 
                  cmap=mycmap)
    
    ax1.coastlines(linewidth=0.8)
    ## land mask
    ax1.pcolormesh(lon2,lat,landMask,transform=ccrs.PlateCarree(),cmap=cmapLand)
        
    
    ## add lat/lon grid lines
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), 
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
    gl.ylabel_style = {'size': 11, 'rotation':20}
    plt.draw()  # Enable the use of `gl._labels`

    for ea in gl.label_artists:
        # return ea of mpl.text type, e.g. Text(135, 30, '30°N') 
        pos = ea.get_position()
        ## put labels over ocean and not over land
        if pos[0] == 150:
            ea.set_position([0, pos[1]])
    

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm1, cmap=mycmap),
             ax=ax1, orientation='vertical')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(str(label), fontsize=14, fontweight='bold')   


    plt.title(str(title), fontsize=16, fontweight='bold', **hfont, y=1.07)
    ## Save figure
    plt.savefig(figureDir + str(savetitle) + '.pdf', dpi=2000, bbox_inches='tight')
    return fig, ax1

