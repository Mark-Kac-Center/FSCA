import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib
from typing import List,Union,Tuple
from enum import Enum

from preprocessing import Preprocessing
from sfc import padding, hilbert2d_sfc
from tools import vectorize,mvectorize,mvectorize2
from mfdfa import mfdfa_py, mfdfa_matlab, ghurst
from mfdfa import aut_artefact8

DEFAULT_SLICE_AXIS = 'z'
DEFAULT_NORM_LEVEL = 'slice'
DEFAULT_SCALES = (5,None,30)
DEFAULT_QS = np.concatenate((np.arange(-4,0,.2),np.arange(0.2,4.2,.2)))
DEFAULT_WINDOW_SIZE = 10
DEFAULT_FIT_ORDER = 2
DEFAULT_MFDFA_LIB = 'py'
DEFAULT_BRAIN_TH = 0.99
DEFAULT_SFC_TYPE = 'hilbert'
DEFAULT_SCALE_PRESET = 'small_scales'

class mfractal_mri:
    
    def __init__(self,
                 verbose : bool = False) -> None:

        self.prep = None # preprocessing object 
        
        self.scan = None # scan numpy array shape 3D = (x,y,z)
        self.slices = None # slices numpy array shape 3D = (slice_id, len_1, len_2)
        self.sfcs = None # SFC signals numpy array shape 2D = (slice_id, len_1*len_2)
        self.Fqs = None # multifractal fluctuations numpy array shape 3D = (slice_id, len(self.scales), len(self.qs))
        self.scales = None # scales shape 1D
        self.qs = None # fluctuation function orders shape 1D
        self.Hs = None # generalized Hurst exponents shape 2D = (slice_id, len(self.qs))
        self.Hs_res = None
        
        self.verbose = verbose
        self.scan_file = None
        self.sfc_mode = None
        self.slice_axis = None
        self.window_size_cleaning = None
        self.fit_order = None
    
    @staticmethod
    def _slice_scan(scan : np.ndarray,
                    slice_axis : str = DEFAULT_SLICE_AXIS,
                    norm_level : str = DEFAULT_NORM_LEVEL,
                    max_norm_val : float = 255.0,
                    quantize_val : bool = False,
                    eps : float = 1e-20,
                    *args, **kwargs) -> np.ndarray:
        
        '''
        Slice the given 3D MRI scan along the specified axis. 
        If slice_axis is None, a 3D "slice" will be created instead. 
        Normalization of slices (norm_level) is done either separately in each slice or globally for the whole scan. 
        If `quantize_val` = True, round the normalized values to the nearest integer.

        Parameters:
        ----------
        scan : np.ndarray
            The 3D scan to be sliced.
        slice_axis : str, optional
            The axis along which to slice the scan, one of ['x', 'y', 'z'].
        norm_level : str, optional
            How to normalize the slices, one of ['slice', 'scan'].
        max_norm_val : float, optional
            The value to which the slices are normalized.
        quantize_val : bool, optional
            Whether to round the normalized values to the nearest integer.
        eps : float, optional
            A small positive constant to avoid division by zero.

        Returns:
        -------
        np.ndarray
            The normalized and sliced scan as a numpy array.
        '''

        if slice_axis == 'z':
            slices = np.transpose(scan,axes=(2,0,1))
        elif slice_axis == 'x':
            slices = np.transpose(scan,axes=(0,1,2))
        elif slice_axis == 'y':
            slices = np.transpose(scan,axes=(1,0,2))
        # elif slice_axis == 'none' or slice_axis is None:
            # slices = np.expand_dims(scan,0)
        else:
            print(f'error: unknown slice_axis = {slice_axis}')
    
        if norm_level == 'slice':
            slices = max_norm_val * (slices / (slices.max(axis=(1,2),keepdims=True) + eps))
        elif norm_level == 'scan':
            slices = max_norm_val * (slices / slices.max())
        
        if quantize_val:
            slices = np.round(slices).astype(int)
        
        if (slices < 0).any():
            print('warning: negative values in slices')
        
        return slices
    
    @staticmethod
    def _slices_to_sfc_hilbert2d(slices : np.ndarray,
                                 *args, **kwargs) -> np.ndarray:
        
        '''
        Convert the given 2D slices to a 1D signal by using the 2D Hilbert space-filling curve.
        The input slices should have three dimensions, i.e., shape (n_slices, height, width).
        The function pads each (height,width) slice with zeros to reach a square size.

        Parameters:
        ----------
        slices : np.ndarray
            The 2D slices to be converted to 1D space-filling curves with shape (n_slices, height, width).
            
        Returns:
        -------
        np.ndarray
            The 1D SFC signal of shape (n_slices, height*width)
        '''
    
        if len(slices.shape) != 3:
            print(f'error: wrong slices dims = {slices.shape}')
            
        slices = vectorize(padding)(slices)
        
        return vectorize(hilbert2d_sfc)(slices)

    @staticmethod
    def _clean_sfc(sfc : np.ndarray,
                   window_size : int = DEFAULT_WINDOW_SIZE,
                   *args, **kwargs) -> np.ndarray:
        
        '''
        The 1D SFC signal cleaning function. 
        The algorithm works by dividing the input curve into windows/chunks of size `window_size` 
        and removing those showing no variation.

        Parameters:
        ----------
        sfc : np.ndarray
            The 1D SFC signal to be cleaned.
        window_size : int, optional
            The size of the window to use for the cleaning.

        Returns:
        -------
        np.ndarray
            The cleaned SFC signal.
        '''
    
        return aut_artefact8(series = sfc, window_size = window_size)    
    
    @staticmethod
    def _infer_scales(sfc : np.ndarray,
                      scales_params : tuple = DEFAULT_SCALES,
                      *args, **kwargs) -> np.ndarray:
        
        '''
        The function infers a set of (logarithmic) scales compatible with the given SFC signal. 
        It takes as input a tuple of the form (min_scale,max_scale,n_scales) 
        with the minimum scale, the maximum scale and the number of scales to consider.
        If max_scale = None, the function sets it to a default value of 1/5th of the SFC signal length. 
        The function returns an array of inferred scales.

        Parameters:
        ----------
        sfc : np.ndarray
            The 1D space-filling curve for which scales need to be inferred.
        scales_params : tuple, optional
            A tuple with which the logarithmically equispaced scales compatible with the SFC signal are provided. 
            It should be in form (min_scale, max_scale, and n_scales).

        Returns:
        -------
        np.ndarray
            An array of inferred scales for the given SFC signal.
        '''
        
        # if isinstance(scales,tuple):
        if len(scales_params) == 3:
            min_scale,max_scale,n_scales = scales_params
            if max_scale is None:
                max_scale = np.round(len(sfc)/5).astype(int)

            scales = np.logspace(np.log(min_scale),np.log(max_scale),n_scales+1,base=np.exp(1))
            scales = np.round(scales).astype(int)

            if len(scales) < n_scales:
                scales = np.array([1]*n_scales)
        else:
            print(f'error: wrong scales tuple dim = {len(scales_params)}')
    
        # elif isinstance(scales,List):
        #     scales = scales.astype(int)
        #     min_scale,max_scale,n_scales = scales[0], scales[-1], len(scales)
        # scales_params = (min_scale,max_scale,n_scales)
        
        return scales

    @staticmethod
    def _single_mfdfa(sfc : np.ndarray,
                      scales : Union[tuple,List] = DEFAULT_SCALES,
                      qs : np.ndarray = DEFAULT_QS,
                      window_size_cleaning : int = DEFAULT_WINDOW_SIZE,
                      fit_order : int = DEFAULT_FIT_ORDER,
                      mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                      *args, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        
        '''
        Applies the multifractal detrended fluctuation analysis (MFDFA) to a single SFC signal.
        
        Parameters:
        ----------
        sfc : np.ndarray
            The 1D SFC signal for which the MFDFA algorithm is applied.
        scales : tuple or list, optional
            The scales of the SFC signal provided either as a tuple or a list.
            Tuple should be in the form (min_scale, max_scale, and n_scales) and is used by `_infer_scales` function. 
            The list should contain the scales.
        qs : np.ndarray, optional
            An array of fluctuation function orders for which the MFDFA is computed.
        window_size_cleaning : int, optional
            The window size for the signal cleaning. Used by the `_clean_sfc` function.
        fit_order : int, optional
            The order of polynomial used in the MFDFA detrending fit.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use. Possible values are "py" for an external MFDFA module, "matlab" for
            own routines refactored from MATLAB.
            
        Returns:
        -------
        Fqs : np.ndarray
            the multifractal detrended fluctuations computed across `qs` and `scales`,
        scales : np.ndarray
            the scales.
        '''
        
        # drop null signal from sfc before MFDFA
        # infer scales before cleaning!
        if isinstance(scales,tuple):
            scales = mfractal_mri._infer_scales(sfc = sfc, scales_params = scales)

        sfc = mfractal_mri._clean_sfc(sfc, window_size = window_size_cleaning)
        
        if sfc.shape[0] < window_size_cleaning:
            # print('warning: no signal')
            dfa = np.nan*np.ones((len(scales),len(qs)))
            # scales = np.array([1]*len(scales))

        else:
            
            mfdfa_args = dict(timeseries = sfc, lag = scales, q = qs, order = fit_order)
            if mfdfa_lib == 'py':
                _, dfa = mfdfa_py(**mfdfa_args)
   
            elif mfdfa_lib == 'matlab':
                _, dfa = mfdfa_matlab(**mfdfa_args)
            else:
                print(f'error: unknown mfdfa_lib = {mfdfa_lib}')
                
            # if dfa.shape != (len(scales),len(qs)):
                
        return dfa, scales
    
    @staticmethod
    def _calc_mfdfa(sfcs : np.ndarray,
                     scales : Union[tuple,List] = DEFAULT_SCALES,
                     qs : np.ndarray = DEFAULT_QS,
                     window_size_cleaning : int = DEFAULT_WINDOW_SIZE,
                     fit_order : int = DEFAULT_FIT_ORDER,
                     mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                    *args, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Perform the MFDFA on the SFC signals.
        Accepts two types of arrays. A 2D array of size (N,M) forming a set of N SFC signals each of length M,
        or 1D array of size (M) forming a single SFC signal of length M. Returns the multifractal fluctuations.

        Parameters:
        -----------
        sfcs : np.ndarray
            The 1D SFC signal for which the MFDFA algorithm is applied. Accepts a 2D array of size (N,M) 
            or a 1D array of size (M) where N is the number of SFC signals each of length M.
        scales : tuple or list, optional
            The scales to use in the MFDFA calculation. Either a list/array of scales or a tuple (min_scale,max_scale,n_scales);
            if max_scale = None, infer from the series length M.
        qs : np.ndarray, optional
            An array of fluctuation function orders for which the MFDFA is computed.
        window_size_cleaning : int, optional
            The window size for the signal cleaning. Used by the `_clean_sfc` function.
        fit_order : int, optional
            The order of polynomial used in the MFDFA detrending fit.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use. Possible values are "py" for an external MFDFA module, "matlab" for
            own routines refactored from MATLAB.

        Returns:
        --------
        Fqs : np.ndarray
            An array of size (N, len(qs), len(scales)) containing the multifractal fluctuations for N SFC signals.
        scales : np.ndarray
            An array of inferred scales for the given SFC signals. Has shape (len(scales),) or (len(scales),1) depending
            on whether the input SFC signal was 1D or 2D.

        '''
        
        if len(sfcs.shape) == 2:
            # multiple mfdfas
            calc = lambda sfc : mfractal_mri._single_mfdfa(sfc,
                                                           scales = scales,
                                                           qs = qs,
                                                           window_size_cleaning = window_size_cleaning,
                                                           fit_order = fit_order,
                                                           mfdfa_lib = mfdfa_lib)
            Fqs, scaless = mvectorize(calc)(sfcs)
            
        elif len(sfcs.shape) == 1:
            # single mfdfa
            Fqs, scaless = mfractal_mri._single_mfdfa(sfc = sfcs,
                                                       scales = scales,
                                                       qs = qs,
                                                       window_size_cleaning = window_size_cleaning,
                                                       fit_order = fit_order,
                                                       mfdfa_lib = mfdfa_lib)
            
        else:
            print(f'error: wrong len(sfcs.shape) = {len(sfcs.shape)}')
            
        return Fqs, scaless
    
    @staticmethod
    def _calc_ghurst(Fqs : np.ndarray,
                    scales : np.ndarray,
                    min_scale_ix : int = None,
                    max_scale_ix : int = None,
                    *args, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        
        '''
        Calculates the generalized Hurst exponent from the multifractal fluctuations array calculated for a set of scales and orders of the fluctuation function. 
        Used in conjunction with the `_calc_mfdfa` method. 

        Parameters:
        ----------
        Fqs : np.ndarray
            The 2D array of multifractal fluctuations where the indices correspond to the scales and the fluctuation orders q respectively.
        scales : np.ndarray
            The array of scales for which the multifractal fluctuation array was calculated.
        min_scale_ix : int, optional
            The index of the first scale to be included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale to be included in the fitting of the generalized Hurst exponent.

        Returns:
        -------
        Hs : np.ndarray
            generalized Hurst exponent for each fluctuation order q.
        Hs_res : np.ndarray
            fitting residuals of the generalized Hurst exponent for each fluctuation order q.

        Notes:
        -----
        For a default array of orders qs spanned over (-4,4) with step 0.2 without q = 0.0, the classic Hurst exponent for q=2 is given by Hs[29].
        '''
        
        if len(Fqs.shape) == 3:
            calc = lambda Fqs,scales: ghurst(Fq = Fqs,
                                             scales = scales,
                                             min_scale_ix = min_scale_ix,
                                             max_scale_ix = max_scale_ix)
        
            Hs,Hs_res = mvectorize2(calc)(Fqs,scales)
            
        elif len(Fqs.shape) == 2:
            Hs,Hs_res = ghurst(Fq = Fqs,
                                scales = scales,
                                min_scale_ix = min_scale_ix,
                                max_scale_ix = max_scale_ix)
            
        elif len(Fqs.shape) == 1:
            Fqs = Fqs.reshape(-1,1)
            Hs,Hs_res = ghurst(Fq = Fqs,
                                scales = scales,
                                min_scale_ix = min_scale_ix,
                                max_scale_ix = max_scale_ix)
            
        else:
            print(f'error: wrong len(Fqs.shape) = {len(Fqs.shape)}')
            
        return Hs, Hs_res
    
    
    
    def load_scan(self,
                  scan_file : Union[str,Path],
                  store_mem : bool = True,
                  *args, **kwargs) -> None:

        '''
        Load a NIFTI scan file. Returns errors if the file does not exist or is not a NIFTI file.

        Parameters
        ----------
        scan_file : str or pathlib.Path
            Path to the NIFTI scan file.
        store_mem : bool, optional
            Store NIFTI file in memory.

        Raises
        ------
        FileNotFoundError
            If the scan_file path does not exist.
        ValueError
            If the scan_file is not a NIFTI file.

        Returns
        -------
        None
        
        Notes
        -----
        If store_mem = True, the resulting scan is stored as a numpy array in `self.scan`.
        '''
        
        scan_file = Path(scan_file)
        
        if not scan_file.exists():
            print(f'error: no scan_file = {scan_file}')
        else:  
            if ''.join(scan_file.suffixes) not in ['.nii.gz','.nii']:
                print(f'error: scan_file = {scan_file} is not a NIFTI file')
            
        self.scan_file = scan_file
        
        if store_mem:
            self.scan = Preprocessing.read_img(self.scan_file).numpy()
        
            
    def run_preprocessing(self,
                          brain_threshold : float = DEFAULT_BRAIN_TH,
                          n4_bias_corr_kwds : dict = None,
                          *args, **kwargs) -> None:
        
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
        The `Preprocessing` object is initialized using the `save_brain_img` and `save_bias_corrected`
        keyword arguments from `kwargs`, and the resulting object is used to preprocess the scan file.

        The resulting preprocessed scan is stored as a numpy array in `self.scan`.
        '''
    
        prep_kwargs = {k:v for k,v in kwargs.items() if k in ['save_brain_img','save_bias_corrected']}
        self.prep = Preprocessing(**prep_kwargs)
        self.scan = self.prep.preprocess_structural_img_path(img_path = self.scan_file,
                                                             brain_threshold = brain_threshold,
                                                             n4_bias_corr_kwds = n4_bias_corr_kwds,
                                                             to_numpy = True)
    
    def slice_scan(self,
                   slice_axis : str = DEFAULT_SLICE_AXIS,
                   *args, **kwargs) -> None:
        '''
        Slice the loaded scan along a specified axis.

        Parameters
        ----------
        slice_axis : str, optional
            The axis along which to slice the scan. Must be one of 'x', 'y', or 'z',
            corresponding to the x-axis, y-axis, or z-axis, respectively.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the `_slice_scan` function.

        Returns
        -------
        None

        Notes
        -----
        The sliced scans are stored in the variable `self.slices`.
        '''
        
        self.slices = mfractal_mri._slice_scan(scan = self.scan,
                                               slice_axis = slice_axis,
                                               *args, **kwargs)
        
        # # set sfc_mode
        # if len(self.slices.shape) == 3:
        #     self.sfc_mode = '2d'
        # elif len(self.slices.shape) == 4:
        #     self.sfc_mode = '3d'
        # else:
        #     print(f'error: incorrect self.slices.shape = {self.slices.shape}')

    def slice_to_sfc(self,
                   sfc_type : str = DEFAULT_SFC_TYPE,
                   *args, **kwargs) -> None:
        '''
        Convert slices to SFC.

        Parameters
        ----------
        sfc_type : str, optional
            The type of space-filling algorithm to use. Possible values are ['hilbert','gilbert'].

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `sfc_type` is invalid.

        Notes
        -----
        Calculates SFC signals from `self.slices` and saves into `self.sfcs`.

        '''
        
        if sfc_type == 'hilbert':
            # if self.sfc_mode == '2d':
            self.sfcs = mfractal_mri._slices_to_sfc_hilbert2d(slices = self.slices)
            
            # elif self.sfc_mode == '3d':
                # print('error: hilbert3d not implemented')
            
        elif sfc_type == 'gilbert':
            print('error: gilbert not implemented')
 

    def calc_mfdfa(self,
                   scales : Union[tuple,List] = DEFAULT_SCALES,
                   qs : np.ndarray = DEFAULT_QS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   *args, **kwargs) -> None:
        
        '''
        Calculate multifractal detrended fluctuation analysis (MFDFA) for sets of `scales` and fluctuation orders `qs` using the `mfdfa_lib` function.

        Parameters:
        -----------
        scales : tuple or list, optional
            The range of scales to use when calculating the MFDFA, used by `_calc_mfdfa` function.
        qs : ndarray, optional
            The list of fluctuation orders q, used by `_calc_mfdfa` function.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use. Possible values are "py" for an external MFDFA module, "matlab" for
            own routines refactored from MATLAB.

        Returns:
        --------
        None

        Notes:
        ------
        The MFDFA result are stored in `self.Fqs` (multifractal fluctuations), `self.qs` and `self.scales` (orders and scales respectively).
        '''
        
        self.qs = qs
        self.Fqs, self.scales = mfractal_mri._calc_mfdfa(sfcs = self.sfcs,
                                                         scales = scales,
                                                         qs = qs,
                                                         mfdfa_lib = mfdfa_lib,
                                                         *args, **kwargs)
        
        
    def calc_ghurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = DEFAULT_SCALE_PRESET,
                   *args, **kwargs) -> None:
        
        '''
        Calculate the generalized Hurst exponents for the previously computed MFDFA results.

        Parameters
        ----------
        min_scale_ix : int, optional
            The index of the first scale to be included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale to be included in the fitting of the generalized Hurst exponent.
        scale_preset : str, optional
            Preset to use when setting min_scale_ix and max_scale_ix automatically.
            Valid options are "all_scales", "small_scales" and "large_scales".

        Returns
        -------
        None
        
        Notes:
        ------
        Generalized Hurst exponents are stored in `self.Hs` (values) and `self.Hs_res` (fitting residuals).
        For a default array of orders qs spanned over (-4,4) with step 0.2 without q = 0.0, the classic Hurst exponent for q=2 is given by self.Hs[:,29].
        '''
        
        if scale_preset == 'all_scales':
            min_scale_ix = None
            max_scale_ix = None
            
        elif scale_preset == 'small_scales':
            min_scale_ix = None
            # max_scale_ix = 11
            max_scale_ix = 5
            
        elif scale_preset == 'large_scales':
            min_scale_ix = 20
            max_scale_ix = None
        
        else:
            print(f'warning: unknown scale_preset = {scale_preset}')
        
        Hs, Hs_res = mfractal_mri._calc_ghurst(Fqs = self.Fqs,
                                              scales = self.scales,
                                              min_scale_ix = min_scale_ix,
                                              max_scale_ix = max_scale_ix)
        
        self.Hs, self.Hs_res = Hs, Hs_res

    def pipeline(self,
                 scan_file : Union[str,Path],
                 preprocess_scan : bool = True,
                 *args,**kwargs):
        
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
            key output : `self.Fqs` 
                multifractal fluctuations for a set of scales `self.scales` and orders `self.qs`
        6. `calc_ghurst` 
            key output : `self.Hs` 
                generalized Hurst exponents for a set of orders `self.qs`

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
        Each subroutine has specific arguments which can be passed via pipeline **kwargs arguments to matching functions.
        '''
        
        kwargs['scan_file'] = scan_file
        
        if preprocess_scan:
            if self.verbose: print('load_scan()...')
            self.load_scan(store_mem = False,*args,**kwargs)
            if self.verbose: print('run_preprocessing()...')
            self.run_preprocessing(*args,**kwargs)
        else:
            if self.verbose: print('load_scan()...')
            self.load_scan(*args,**kwargs)
        if self.verbose: print('slice_scan()...')
        self.slice_scan(*args,**kwargs)
        if self.verbose: print('slice_to_sfc()...')
        self.slice_to_sfc(*args,**kwargs)
        if self.verbose: print('calc_mfdfa()...')
        self.calc_mfdfa(*args,**kwargs)
        if self.verbose: print('calc_ghurst()...')
        self.calc_ghurst(*args,**kwargs)
        