import os
import xarray as xr
import numpy as np
import pandas as pd

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
