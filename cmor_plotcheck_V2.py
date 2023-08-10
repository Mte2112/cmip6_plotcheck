import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
import time
import cartopy.crs as ccrs
import sys
import argparse
import warnings

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cmor_plot.cmor_plot.cptools import Tools as cpt
from tabulate import tabulate

# Time entire process
start = time.time()

# Temporary directory being used for preliminary development
runE2 = '/Users/aherron1/Documents/Code/CMIP/Visualization/testdata/css/cmip6/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/historical/r1i1p1f2/'
runE3 = '/Users/aherron1/Documents/Code/CMIP/Visualization/testdata/css/cmip6/CMIP6/CMIP/NASA-GISS/GISS-E3-G/historical/r1i1p1f1/'

# Change directory and set paths for looping
outdir = os.getcwd()
os.chdir(outdir)
allvarsE2 = runE2 + "*/*/*/*"
allvarsE3 = runE3 + "*/*/*/*"

# Set central longitude for plotting
central_lon = 0

# Function to calculate similarity of distributions btw E2 and E3
def KL_divergence(p, q):
    if len(p) > len(q):
        p = np.random.choice(p, len(q))
    elif len(q) > len(p):
        q = np.random.choice(q, len(p))
    kl_div = np.sum(p * np.log(p/q))
    return '{:.2e}'.format(kl_div)

# Loop through the E3 directory 
for direc3 in glob.glob(allvarsE3):
        
    # Call 'get_sample' for E3, save filename and open dataset
    fileE3 = cpt.get_sample(direc3, outdir)
    
    if ('/fx/' in fileE3) | ('/fy/' in fileE3) | ('/Ofx/' in fileE3) | ('/Ofy/' in fileE3):
        None
    else:
        dsE3 = xr.open_dataset(fileE3, decode_times=False)
        
        # Get relevant varname & frequency
        freq = direc3.split("/")[-4]
        varname = direc3.split("/")[-3]
        gride3 = direc3.split("/")[-2]
        modelv2 = runE2.split('CMIP6')[1].split('/')[3]
        modelv3 = direc3.split("/")[-7]
        
        # Try to find the same var/freq in E2
        # start here-- basically try to get the same file using the freq and varname defined earlier. if exception, plot no data
        e3version = direc3.split("/")[-1]
        e2path_short = runE2 + direc3.split(e3version)[0].split(runE3)[1]
        e2path_full = e2path_short + "*"
        direc2 = glob.glob(e2path_full)
        if len(direc2) > 0:
            direc2 = direc2[0]
            os.chdir(direc2) # if this fails, there was no path found (aka no matching data)
            fileE2 = cpt.get_sample(direc2, outdir)

            # Create e2 dataset
            dsE2 = xr.open_dataset(fileE2, decode_times=False)

            # Verify that E2 variable exists, and carry on
            varexist = 1
            
        else:
            # Check if there is a native grid E2 match to E3 regridded
            if gride3 == "gr":
                e2path_full_native = e2path_full.replace("/gr/", "/gn/")
                direc2 = glob.glob(e2path_full_native)
                if len(direc2) > 0:
                    direc2 = direc2[0]
                    os.chdir(direc2) # if this fails, there was no path found (aka no matching data)
                    fileE2 = cpt.get_sample(direc2, outdir)

                    # Create E2 dataset
                    dsE2 = xr.open_dataset(fileE2)

                    # Verify that E2 variable exists, and carry on
                    varexist = 1
            else:   
                print("No E2 match found for " + fileE3.split('/')[-1] + ", skipping...")
                dsE2 = None
                varexist = 0
                
        # Check for num dimensions
        dsE3 = cpt.check_dim(dsE3, varname)
        if varexist == 1:
            dsE2 = cpt.check_dim(dsE2, varname)

        arr_arr = cpt.getstats(dsE3, dsE2, varexist, varname)
        arr_arr_str = ["maxval", "minval", "timemean", "vmax", "vmin"]
        
        # Loop through array and set variables
        for arr, n in zip(arr_arr_str, np.arange(0,len(arr_arr), 1)):
            locals()[arr] = arr_arr[n]
        
        # Title
        years = fileE3.split('_')[-1].split('.')[0]
        m3title = direc3.split("/")[-7] + "\n" + direc3.split("/")[-6] + " " + direc3.split("/")[-5] + " " \
                  + direc3.split("/")[-2] + " " + direc3.split("/")[-1] + " " + "(" + years + ")" + '\n(Daily, Mean)'
        if varexist == 1:
            m2title = direc2.split("/")[-7] + "\n" + direc2.split("/")[-6] + " " + direc2.split("/")[-5] + " " \
                      + direc2.split("/")[-2] + " " + direc2.split("/")[-1] + " " + "(" + years + ")" + '\n(Daily, Mean)'
        
        # Get cbar labels
        try:
            labele3 = 'Mean' + dsE3[varname].attrs["units"]
        except:
            labele3 = 'Mean' + varname
        try:
            labele2 = 'Mean' + dsE2[varname].attrs["units"]
        except:
            labele2 = 'Mean' + varname

        end = time.time()

        # Calculate bounds for colorbars (should be same for both plots)
        cbar_upper = max(maxval)
        cbar_lower = min(minval)

        # E2 plot (left)
        if varexist == 1:
            fig = plt.figure(figsize=(14,5))
            ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_lon))
            gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
            gl1.top_labels = gl1.right_labels = False
            timemean[1].plot(transform=ccrs.PlateCarree(),
                            cbar_kwargs={'orientation':'horizontal','pad': 0.06},
                            vmax=cbar_upper,
                            vmin=cbar_lower)
            ax1.coastlines()
            plt.title(m2title)

        else:
            fig = plt.figure()
            ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_lon))
            ax1.text(0.25, 0.5, 'NO DATA', fontsize=40)
            plt.title(m2title)
            plt.xticks([])
            plt.yticks([])

        # E3 plot (right)
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_lon))
        gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linestyle='--')
        gl2.top_labels = gl2.left_labels = False
        timemean[0].plot(transform=ccrs.PlateCarree(),
                        cbar_kwargs={'orientation':'horizontal','pad': 0.06},
                        vmax=cbar_upper,
                        vmin=cbar_lower)
        ax2.coastlines()
        plt.title(m3title)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks([])
        plt.yticks([])

        # Gather statistics and percent difference between E2 and E3
        E2_mean = list(dsE2.mean().data_vars.items())[-1][1].values.item()
        E3_mean = list(dsE3.mean().data_vars.items())[-1][1].values.item()
        mean_percent_diff = (abs(E2_mean - E3_mean) / E3_mean) * 100

        E2_median = list(dsE2.median().data_vars.items())[-1][1].values.item()
        E3_median = list(dsE3.median().data_vars.items())[-1][1].values.item()
        median_percent_diff = (abs(E2_median - E3_median) / E3_median) * 100

        E2_max = list(dsE2.max().data_vars.items())[-1][1].values.item()
        E3_max = list(dsE3.max().data_vars.items())[-1][1].values.item()
        max_percent_diff = (abs(E2_max - E3_max) / E3_max) * 100

        E2_min = list(dsE2.min().data_vars.items())[-1][1].values.item()
        E3_min = list(dsE3.min().data_vars.items())[-1][1].values.item()
        min_percent_diff = (abs(E2_min - E3_min) / E3_min) * 100

        E2_std = list(dsE2.std().data_vars.items())[-1][1].values.item()
        E3_std = list(dsE3.std().data_vars.items())[-1][1].values.item()
        std_percent_diff = (abs(E2_std - E3_std) / E3_std) * 100

        # Create dataframe for statistics
        variable = list(dsE3.data_vars.items())[-1][0]
        var_title = 'Statistic (for ' + str(variable) + ')'
        df = pd.DataFrame({var_title: ['Mean', 'Median', 'Minimum', 'Maximum', 'Standard Deviation'],
                           'GISS-E2': [E2_mean, E2_median, E2_min, E2_max, E2_std],
                           'GISS-E3': [E3_mean, E3_median, E3_min, E3_max, E3_std],
                           'Percent Difference (%)': [mean_percent_diff, median_percent_diff, min_percent_diff, \
                           max_percent_diff, std_percent_diff]}).set_index([var_title]).round(3)

        # Assign accuracy check (arbitrary for now)
        # (starting with 10% percent difference as requirement to pass check)
        df['Test (within 10%)'] = np.where(df['Percent Difference (%)'] < 10, 'PASS', 'FAIL')

        # Print formatted table
        formatted_df = tabulate(df, headers='keys', tablefmt='fancy_grid')
        print(f'\nE2 File: {m2title}\n\nE3 File: {m3title}')
        print(formatted_df)

        # Print KL divergence between E2 and E3 values
        E2_vals = list(dsE2.data_vars.items())[-1][1].values.flatten()
        E3_vals = list(dsE3.data_vars.items())[-1][1].values.flatten()
        print(f'KL Divergence: {KL_divergence(E2_vals, E3_vals)}')

# Function for saving all plots as a single PDF
def save_image(filename):

    # Create wrapper around pdf file
    p = PdfPages(filename)
      
    # Get list of existing figure numbers
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
      
    # Save files in list
    for fig in figs: 
        fig.savefig(p, format='pdf') 
      
    # Close object
    p.close()  
  
# Name file, call function
filename = 'cmor_plotcheck_V2_plots.pdf'
save_image(filename)

# Calculate overall time
end = time.time()
duration = round(end - start, 3)
print(f'\nTotal time: {duration}')
