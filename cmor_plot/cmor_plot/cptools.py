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
        
        #global sample # set global variable
        
        os.chdir(direc)
        firstfile = os.listdir()[0] # get first file in directory to sample
        os.chdir(outdir)
        
        # Extract the first file in each directory (i.e. the first N years) to sample each var
        sample = direc + '/' + firstfile 
        
        return sample
    
    def check_dim(ds):
        """ Function to select first pressure/height/etc. level to sample
        If file > 3D, slice at first index of Nth dimension (other than lat/lon/time)
        
        """
        
        try: 
            ds.plev # check if 'plev' is a dimension
            # sample file by masking first plev
            return ds.where(ds['plev'] == ds.plev[0].values) 
        except:
            print("No plev detected")
            return ds

        try:
            ds.height # check if 'height' is a dimension
            height_bin = 1
            # sample file by masking first height
            return ds.where(ds['height'] == ds.height[0].values) 
        except:
            print("No height detected")
            height_bin = 0
            return ds



    def getstats(ds3, ds2, varexist, varname):
        """ Function calculates some basic statistics for plotting
        Output the statistics as an array of arrays
            - arr_arr = [maxval, minval, timemean, vmax, vmin]
        
        """
        
        # Initialize individual arrays
        maxval = []
        minval = []
        timemean = []
        vmax = []
        vmin = []
        
        # Get E23 stats
        maxval.append(ds3[varname].max().values)
        minval.append(ds3[varname].min().values)
        timemean.append(ds3[varname].mean('time'))
        vmax.append(ds3[varname].mean('time').max().values)
        vmin.append(ds3[varname].mean('time').min().values)
        # Get E2 stats (if have matching data)
        if varexist == 1:
            maxval.append(ds2[varname].max().values)
            minval.append(ds2[varname].min().values)
            timemean.append(ds2[varname].mean('time'))
            vmax.append(ds2[varname].mean('time').max().values)
            vmin.append(ds2[varname].mean('time').min().values)
        else:
            maxval.append("NA")
            minval.append("NA")
            timemean.append("NA")
            vmax.append("NA")
            vmin.append("NA")
            
        # Create array of arrays
        arr_arr = [maxval, minval, timemean, vmax, vmin]
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