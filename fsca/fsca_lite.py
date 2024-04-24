from typing import List,Union,Tuple
from pathlib import Path

from .mfmri_core import BaseMFractalMRI
from .const import *

class LiteMFractalMRI(BaseMFractalMRI):
    '''
    pipeline WITHOUT the preprocessing module
    '''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def pipeline(self,
                 scan_file : Union[str,Path],
                 **kwargs):

        '''
        A pipeline function that runs the following sequence of functions:

        1. `load_scan`
            key output : `self.scan`
                an array with a 3D scan
        3. `slice_scan`
            key output : `self.slices`
                a set of 2D slices of `self.scan`
        4. `slice_to_sfc`
            key output : `self.sfcs`
                a set of 1D SFC signals obtained from 2D slices `self.slices`
        5. `calc_mfdfa`
            key output : `self.fqs`
                multifractal fluctuations for scales `self.scales` and orders `self.qorders`
        6. `calc_ghurst`
            key output : `self.ghs`
                generalized Hurst exponents for a set of orders `self.qorders`

        Parameters:
        -----------
        scan_file : str or pathlib.Path
            Path to the NIFTI scan file to load.
        *args, **kwargs : additional arguments and keyword arguments
            Additional arguments and keyword arguments that are passed to the individual functions
            in the pipeline.

        Returns
        -------
        None

        Notes:
        ------
        Each subroutine has specific arguments which can be passed via pipeline **kwargs.
        '''

        kwargs['scan_file'] = scan_file

        self.load_scan(**kwargs)
        self.slice_scan(**kwargs)
        self.slice_to_sfc(**kwargs)
        self.calc_mfdfa(**kwargs)
        self.calc_ghurst(**kwargs)
