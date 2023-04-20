import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib
from typing import List,Union,Tuple
from enum import Enum

from sfc import padding, hilbert2d_sfc
from tools import vectorize,mvectorize,mvectorize2
from mfdfa import mfdfa_py, mfdfa_matlab, ghurst
from mfdfa import aut_artefact8

DEFAULT_SLICE_AXIS = 'z'
DEFAULT_SCALES = (5,None,30)
DEFAULT_QS = np.concatenate((np.arange(-4,0,.2),np.arange(0.2,4.2,.2)))
DEFAULT_WINDOW_SIZE = 10
DEFAULT_FIT_ORDER = 2
DEFAULT_MFDFA_LIB = 'py'

class mfractal_mri:
    
    def __init__(self,
                 verbose : bool = False):
        
        self._state = 0
        
        self.prep = None
        
        self.scan_file = None
        self.slices = None
        self.sfcs = None
        self.Fqs = None
        self.scales = None
        self.qs = None
        self.Hs = None
        self.Hs_res = None
        
        self.sfc_mode = None
        self.slice_axis = None
        self.window_size_cleaning = None
        self.fit_order = None
        
        self.verbose = verbose
        
    @staticmethod
    def _load_scan(nii_file : Union[str,Path],
                   *args, **kwargs) -> nib.spatialimages.SpatialImage:
        
        return nib.load(nii_file)
    
    @staticmethod
    def _slice_scan(scan_file : nib.spatialimages.SpatialImage,
                    slice_axis : str = DEFAULT_SLICE_AXIS,
                    norm_level : str = 'slice',
                    max_norm_val : float = 255.0,
                    quantize_slices : bool = False,
                    eps : float = 1e-20,
                    *args, **kwargs) -> np.ndarray:
        '''
        scan_file -
        norm_level - 
        slice_axis - along which axis to slice the scan for further application of 2D SFC; 
                    if none then a 3D SFC will be applied; [x,y,z,none]; default = z
        norm_level - how to normalize the slices; [slice,scan]; default = slice
        max_norm_val - value to which slices are normalized; default = 255
        '''
        if slice_axis == 'z':
            slices = np.transpose(scan_file.dataobj,axes=(2,0,1))
        elif slice_axis == 'x':
            slices = np.transpose(scan_file.dataobj,axes=(0,1,2))
        elif slice_axis == 'y':
            slices = np.transpose(scan_file.dataobj,axes=(1,0,2))
        elif slice_axis == 'none' or slice_axis == None:
            slices = np.expand_dims(scan_file.dataobj,0)
        else:
            print(f'error: unknown slice_axis = {slice_axis}')
    
        if norm_level == 'slice':
            slices = max_norm_val * (slices / (slices.max(axis=(1,2),keepdims=True) + eps))
        elif norm_level == 'scan':
            slices = max_norm_val * (slices / slices.max())
        
        if quantize_slices:
            slices = np.round(slices).astype(int)
        
        if (slices < 0).any():
            print('warning: negative values in slices')
        
        return slices
    
    @staticmethod
    def _slices_to_sfc_hilbert2d(slices : np.ndarray,
                                 *args, **kwargs) -> np.ndarray:
        if len(slices.shape) != 3:
            print(f'error: wrong slices dims = {slices.shape}')
            
        slices = vectorize(padding)(slices)
        
        return vectorize(hilbert2d_sfc)(slices)

    @staticmethod
    def _clean_sfc(sfc : np.ndarray,
                   window_size : int = DEFAULT_WINDOW_SIZE,
                   *args, **kwargs) -> np.ndarray:
        '''
        clean sfc curve by erasing null parts of the signal
        '''
        
        return aut_artefact8(series = sfc, window_size = window_size)    
    
    @staticmethod
    def _infer_scales(scales : Union[tuple,List],
                      sfc : np.ndarray,
                      *args, **kwargs) -> np.ndarray:
        
        if isinstance(scales,tuple):
            if len(scales) == 3:
                min_scale,max_scale,n_scales = scales
                if max_scale is None:
                    max_scale = np.round(len(sfc)/5).astype(int)
                
                scales = np.logspace(np.log(min_scale),np.log(max_scale),n_scales+1,base=np.exp(1))
                scales = np.round(scales).astype(int)
                
                if len(scales) < n_scales:
                    scales = np.array([1]*n_scales)
            else:
                print(f'error: wrong scales tuple dim = {len(scales)}')
    
        elif isinstance(scales,List):
            scales = scales.astype(int)
            min_scale,max_scale,n_scales = scales[0], scales[-1], len(scales)
            
        scales_params = (min_scale,max_scale,n_scales)
        
        return scales

    @staticmethod
    def _single_mfdfa(sfc : np.ndarray,
                      scales : Union[tuple,List] = DEFAULT_SCALES,
                      qs : np.ndarray = DEFAULT_QS,
                      window_size_cleaning : int = DEFAULT_WINDOW_SIZE,
                      fit_order : int = DEFAULT_FIT_ORDER,
                      mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                      *args, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        f'''
        help
        scales - {DEFAULT_SCALES}
        '''
        
        # drop null signal from sfc before MFDFA
        # infer scales before cleaning!
        scales = mfractal_mri._infer_scales(scales,sfc)
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
                dfa = mfdfa_matlab(**mfdfa_args)
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
        MFDFA function
        sfcs - array of size (N,M) or (M,), a set of N sfc series of length M
        scales - list/array of scales or a tuple (min_scale,max_scale,n_scales); 
                if max_scale = None, infer from the series length M; 
                default = (5,None,30)
        qs - list/array of generalized Hurst exponents; 
                default = [-4,4,.2] without q=0
        window_size_cleaning - size of the null signal window; used in sfc cleaning; 
                                default = 10
        fit_order - order of detrending polynomial; 
                    default = 2
        mfdfa_lib - type of MFDFA procedure; 
                        "py" uses an external MFDFA module;
                        "matlab" uses own routines refactored from matlab; 
                        defualt = "py"
        '''
        
        if len(sfcs.shape) == 2:
            
            calc = lambda sfc : mfractal_mri._single_mfdfa(sfc,
                                                           scales = scales,
                                                           qs = qs,
                                                           window_size_cleaning = window_size_cleaning,
                                                           fit_order = fit_order,
                                                           mfdfa_lib = mfdfa_lib)
            Fqs, scaless = mvectorize(calc)(sfcs)
            
        elif len(sfcs.shape) == 1:
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
    def _calc_hurst(Fqs : np.ndarray,
                    scales : np.ndarray,
                    min_scale_ix : int = None,
                    max_scale_ix : int = None,
                    *args, **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        
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
                  nii_file : str,
                  *args, **kwargs) -> None:
        '''
        TODO: load_scan help
        '''
        
        self.scan_file = mfractal_mri._load_scan(nii_file = nii_file)   
    
    def slice_scan(self,
                   slice_axis : str = DEFAULT_SLICE_AXIS,
                   *args, **kwargs) -> None:
        '''
        TODO: slice scan help
        '''
        
        self.slices = mfractal_mri._slice_scan(scan_file = self.scan_file,
                                               slice_axis = slice_axis,
                                               *args, **kwargs)
        
        # set sfc_mode
        if len(self.slices.shape) == 3:
            self.sfc_mode = '2d'
        elif len(self.slices.shape) == 4:
            self.sfc_mode = '3d'
        else:
            print(f'error: incorrect self.slices.shape = {self.slices.shape}')

    def slice_to_sfc(self,
                   sfc_type : str = 'hilbert',
                   *args, **kwargs) -> None:
        '''
        TODO: slice_to_sfc help
        '''
        
        if sfc_type == 'hilbert':
            if self.sfc_mode == '2d':
                self.sfcs = mfractal_mri._slices_to_sfc_hilbert2d(slices = self.slices)
            
            elif self.sfc_mode == '3d':
                print('error: hilbert3d not implemented')
            
        elif sfc_type == 'gilbert':
            print('error: gilbert not implemented')
 

    def calc_mfdfa(self,
                   scales : Union[tuple,List] = DEFAULT_SCALES,
                   qs : np.ndarray = DEFAULT_QS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   *args, **kwargs) -> None:
        '''
        TODO: calc_mfdfa help
        '''
        
        self.qs = qs
        self.Fqs, self.scales = mfractal_mri._calc_mfdfa(sfcs = self.sfcs,
                                                         scales = scales,
                                                         qs = qs,
                                                         mfdfa_lib = mfdfa_lib,
                                                         *args, **kwargs)
        
        
    def calc_hurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = 'small_scales',
                   *args, **kwargs):
        '''
        TODO: calc_hurst help
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
        
        Hs, Hs_res = mfractal_mri._calc_hurst(Fqs = self.Fqs,
                                              scales = self.scales,
                                              min_scale_ix = min_scale_ix,
                                              max_scale_ix = max_scale_ix)
        
        self.Hs, self.Hs_res = Hs, Hs_res

    def pipeline(self,*args,**kwargs):
        self.load_scan(*args,**kwargs)
        self.slice_scan(*args,**kwargs)
        self.slice_to_sfc(*args,**kwargs)
        self.calc_mfdfa(*args,**kwargs)
        self.calc_hurst(*args,**kwargs)
        