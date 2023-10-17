from typing import List,Union,Tuple
from pathlib import Path

from .preprocessing import Preprocessing
from .mfmri_core import BaseMFractalMRI
from .const import *

class MFractalMRI(BaseMFractalMRI):
    '''
    pipeline including a preprocessing module (needs ants)
    '''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        # preprocessing object
        self.prep = None

    def run_preprocessing(self,
                          brain_threshold : float = DEFAULT_BRAIN_TH,
                          n4_bias_corr_kwds : dict = None,
                          **kwargs) -> None:

        '''
        Run the preprocessing pipeline.

        Parameters
        ----------
        brain_threshold : float, optional
            The intensity threshold for the brain extraction step.
        n4_bias_corr_kwds : dict, optional
            Keyword arguments for the N4 bias correction step.
        **kwargs :
            Additional keyword arguments for the `Preprocessing` class.

        Returns
        -------
        None

        Notes
        -----
        The `Preprocessing` object is initialized using the `save_brain_img`
        and `save_bias_corrected` keyword arguments from `kwargs`
        and the resulting object is used to preprocess the scan file.

        The resulting preprocessed scan is stored as a numpy array in `self.scan`.
        '''

        if self.verbose:
            print('run_preprocessing()...')

        keys = ['save_brain_img','save_bias_corrected']
        prep_kwargs = {k:v for k,v in kwargs.items() if k in keys}

        self.prep = Preprocessing(**prep_kwargs)
        self.scan = self.prep.preprocess_structural_img_path(img_path = self.scan_file,
                                                             brain_threshold = brain_threshold,
                                                             n4_bias_corr_kwds = n4_bias_corr_kwds,
                                                             to_numpy = True)

    def pipeline(self,
                 scan_file : Union[str,Path],
                 preprocess_scan : bool = True,
                 **kwargs):

        '''
        A pipeline function that runs the following sequence of functions:

        1. `load_scan` + `run_preprocessing` (optional when preprocess_scan = True)
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
        preprocess_scan : bool, optional
            Whether to preprocess the scan data before running the pipeline.
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

        if preprocess_scan:
            self.load_scan(store_mem = False,**kwargs)
            self.run_preprocessing(**kwargs)
        else:
            self.load_scan(**kwargs)

        self.slice_scan(**kwargs)
        self.slice_to_sfc(**kwargs)
        self.calc_mfdfa(**kwargs)
        self.calc_ghurst(**kwargs)
