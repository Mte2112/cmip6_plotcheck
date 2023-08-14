import os
import sys
import argparse
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

        # Create color coded dataframe
        color_df = df.style.apply(lambda x: ['background: red' if v == 'FAIL' else '' for v in x], axis = 1).apply(lambda x:\
                                      ['background: green' if v == 'PASS' else '' for v in x], axis = 1)

        # Format tabulated dataframe, return both tables
        formatted_df = tabulate(df, headers='keys', tablefmt='fancy_grid')
        return formatted_df, color_df

    # Set command line arguments
    def readOptions(args=sys.argv[1:]):
        parser = argparse.ArgumentParser(description="List of parsing commands")

        parser.add_argument("-r3",
                            "--E3run", 
                            help="Input your E3 run directory (up to variant, i.e. '../r1i1p1f1/'. Please use CSS filesystem or follow directory structure",
                            default='/Users/aherron1/Documents/Code/CMIP/Visualization/testdata/css/cmip6/CMIP6/CMIP/NASA-GISS/GISS-E3-G/historical/r1i1p1f1/')

        parser.add_argument("-rX",
                            "--EXrun",
                            help="Input your comparison file directory (up to variant, i.e. '.../r1i1p1f1/'. Please use CSS filesystem or follow directory structure",
                            default='/Users/aherron1/Documents/Code/CMIP/Visualization/testdata/css/cmip6/CMIP6/CMIP/NASA-GISS/GISS-E2-1-G/historical/r1i1p1f2/')
        
        parser.add_argument("-f",
                            "--figure_name",
                            help="Input your figure name (i.e. 'E3_E07_comp_plots')",
                            default='cmor_plotcheck_V2_plots.pdf')

        parser.add_argument('-hist',
                            '--histogram',
                            help='Option to include histogram in outputted figure (input "-hist" to create histogram in addition to baseline plots)',
                            action='store_true')

        parser.add_argument('-var',
                            '--variable',
                            help='Specific variable of interest for given query (following CSS directory structure and naming conventions)',
                            default=None,
                            nargs='+')
        
        opts = parser.parse_args(args)
        return opts