import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.backends.backend_pdf import PdfPages
from tabulate import tabulate

class Tools:
    """ Tools (functions) used for cmor_plotcheck
    
    Attributes:
        get_sample: Iterates through E3/2 directories and takes a sample file 
        for each relevant variables
        
        check_dim: Checks for # dimensions. Any files with > 3, selects sample 
        (first index in dimension (i.e. first pressure level))
        
        get_stats: Generates key statistics to be recorded and displayed in a 
        table (i.e. min, max)
            
    """
    
    
    def __init__(self):
        pass
                        
    
    def get_sample(direc, outdir):
        
        """ Funtion to get a sample file to be used in each plot. 
        The sample file will be the first file listed in the directory 
        which is the earliest file of the dataset 
        (i.e. 1850-1900 usually for historical period) 
        
        """
                
        os.chdir(direc)
        file_list = os.listdir()
        file_list.remove('.git')
        firstfile = file_list[0]
        os.chdir(outdir)
        
        # Extract the first file in each directory (i.e. the first N years) to sample each var
        sample = direc + '/' + firstfile 
        
        return sample
    
    def check_dim(ds, varname):
        """ Function to select first pressure/height/etc. level to sample
        If file > 3D, slice at first index of Nth dimension (other than lat/lon/time)
        Might be more interesting for some variables if the sample statistic was
        the integrated column of X. But this can be programmed in later if need be. 
        Please reach out to me if you would like to chat about this-maxwell.t.elling@nasa.gov
        
        TO DO: CONSIDER THE VARS WITH NO TIME DIM
        
        """
        
        dims = ds[varname].dims
        ijt = ["lon", "lat", "time"]
        for dim in dims:
            if dim in ijt:
                pass
            else:
                ds = ds.where(ds[dim] == ds[dim][0].values, drop=True)
        return ds



    def getstats(ds3, ds2, varexist, varname):
        """ Function calculates some basic statistics for plotting
        Output the statistics as an array of arrays
        
        Inputs: 
        ds3: e3 dataset
        ds2: e2 dataset
        varexist: boolean indicating whether e2 matching dataset exists (1) or not (0)
        varname: variable name extracted from directory structure
        tdim: boolean indicating whether time dimension present in file
        
        Outputs: 
        arr_arr = [maxval, minval, timemean, vmax, vmin]
            -maxval: maximum value in dataset
            -minval: minimum value in dataset
            -timemean: time averaged statistic for sample
            -vmax: colorbar max
            -vmin: colorbar min
        """
        
        # Initialize individual arrays
        maxval = []
        minval = []
        timemean = []
        vmax = []
        vmin = []
        tdim = [] 
        
        # Get E23 stats
        maxval.append(ds3[varname].max().values)
        minval.append(ds3[varname].min().values)
        # If time dimension not found in dataset, skipping calculation and plot 2D data
        if "time" in ds3[varname].dims:
            timemean.append(ds3[varname].mean('time'))
            tdim.append(1)
        else:
            timemean.append(ds3[varname])
            tdim.append(0)
            print("No time dimension found in E3 " + varname + " ds, skipping time average...")
        vmax.append(ds3[varname].mean('time').max().values)
        vmin.append(ds3[varname].mean('time').min().values)
        # Get E2 stats (if have matching data)
        if varexist == 1:
            maxval.append(ds2[varname].max().values)
            minval.append(ds2[varname].min().values)
            # If time dimension not found in dataset, skipping calculation and plot 2D data
            if "time" in ds2[varname].dims:
                timemean.append(ds2[varname].mean('time'))
                tdim.append(1)
            else:
                timemean.append(ds2[varname])
                tdim.append(0)
                print("No time dimension found in E2 " + varname + " ds, skipping time average...")
            vmax.append(ds2[varname].mean('time').max().values)
            vmin.append(ds2[varname].mean('time').min().values)
        else:
            maxval.append("NA")
            minval.append("NA")
            timemean.append("NA")
            vmax.append("NA")
            vmin.append("NA")
            
        # Create array of arrays
        arr_arr = [maxval, minval, timemean, vmax, vmin, tdim]
        return arr_arr
        
    def __repr__(self):
    
        """Function to output the characteristics of these Tools
        
        Args:
            None
        
        Returns:
            string: characteristics of the tools
        
        """
        
        return "get_sample(direc, outdir), check_dim(ds), getstats(ds3, ds2, varexist)".\
        format(self.get_sample, self.check_dim, self.getstats)

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

    # Function to calculate similarity of distributions btw E2 and E3
    def KL_divergence(p, q):
        if len(p) > len(q):
            p = np.random.choice(p, len(q))
        elif len(q) > len(p):
            q = np.random.choice(q, len(p))
        kl_div = np.sum(p * np.log(p/q))
        return '{:.2e}'.format(kl_div)

    # Visual comparison of E2 / E3 data
    def histogram(E2_vals, E3_vals, hist_title, varname):

        # Set plotting constants
        num_bins = 25
        alpha = 0.25

        # E2 histogram
        fig, ax = plt.subplots(figsize=(10,7))
        ax.hist(E2_vals, bins=num_bins, color='blue', edgecolor='black', alpha=alpha, label='E2 Data')
        ax.set_title(hist_title)
        ax.set_xlabel(varname)
        ax.set_ylabel("Percentage of E2 Data")
        ax.legend(loc='upper left')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(E2_vals)))
        ax.grid()

        # E3 histogram
        ax_copy = ax.twinx()
        ax_copy.hist(E3_vals, bins=num_bins, color='orange', edgecolor='black', alpha=alpha, label='E3 Data')
        ax_copy.set_ylabel("Percentage of E3 Data")
        ax_copy.legend(loc='upper right')
        ax_copy.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=len(E3_vals)))

    # Function to create neatly formatted table of statistics for E2 and E3
    def stats_df(dsE2, dsE3, threshold):

        # Mean stats
        E2_mean = list(dsE2.mean().data_vars.items())[-1][1].values.item()
        E3_mean = list(dsE3.mean().data_vars.items())[-1][1].values.item()
        mean_percent_diff = (abs(E2_mean - E3_mean) / E3_mean) * 100

        # Median stats
        E2_median = list(dsE2.median().data_vars.items())[-1][1].values.item()
        E3_median = list(dsE3.median().data_vars.items())[-1][1].values.item()
        median_percent_diff = (abs(E2_median - E3_median) / E3_median) * 100

        # Maximum stats
        E2_max = list(dsE2.max().data_vars.items())[-1][1].values.item()
        E3_max = list(dsE3.max().data_vars.items())[-1][1].values.item()
        max_percent_diff = (abs(E2_max - E3_max) / E3_max) * 100

        # Mininum stats
        E2_min = list(dsE2.min().data_vars.items())[-1][1].values.item()
        E3_min = list(dsE3.min().data_vars.items())[-1][1].values.item()
        min_percent_diff = (abs(E2_min - E3_min) / E3_min) * 100

        # Standard deviation stats
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
        df['Test (within 10%)'] = np.where(df['Percent Difference (%)'] < threshold, 'PASS', 'FAIL')

        # Format and return table
        formatted_df = tabulate(df, headers='keys', tablefmt='fancy_grid')
        return formatted_df