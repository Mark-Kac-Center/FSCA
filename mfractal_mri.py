import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import nibabel as nib


from sfc import padding, hilbert2d_sfc
from tools import vectorize
from mfdfa import mfdfa_py, mfdfa_matlab, ghurst

class mfractal_mri:
    
    def __init__(self,
                 verbose : bool = False):
        self.scan_file = None
        self.slice_axis = None
        self.slices = None
        self.sfcs = None
        self.Fqs = None
        self.qs = None
        self.scales = None
        
        self.verbose = verbose
        
    def load_scan(self,
                  nii_file : str, *args,**kwargs):
        # TODO: check if file exists and is a nii.gz file
        self.scan_file = nib.load(nii_file)
    
    def _slice_scan(self,
                    slice_axis : str = 'z',
                    norm_level : str = 'slice',
                    max_norm_val : float = 255.0, *args,**kwargs):
        '''
        slice_axis - along which axis to slice the scan for further application of 2D SFC; 
                    if none then a 3D SFC will be applied; [x,y,z,none]; default = z
        norm_level - how to normalize the slices; [slice,scan]; default = slice
        max_norm_val - value to which slices are normalized; default = 255
        '''
        if slice_axis == 'z':
            self.slices = np.transpose(self.scan_file.dataobj,axes=(2,0,1))
        elif slice_axis == 'x':
            self.slices = np.transpose(self.scan_file.dataobj,axes=(0,1,2))
        elif slice_axis == 'y':
            self.slices = np.transpose(self.scan_file.dataobj,axes=(1,0,2))
        elif slice_axis == 'none' or slice_axis == None:
            self.slices = np.expand_dims(self.scan_file.dataobj,0)
        else:
            print(f'error: unknown slice_axis = {slice_axis}')
    
        if norm_level == 'slice':
            eps = 1e-20
            self.slices = max_norm_val * (self.slices / (self.slices.max(axis=(1,2),keepdims=True) + eps))
        elif norm_level == 'scan':
            self.slices = max_norm_val * (self.slices / self.slices.max())
        
        if (self.slices < 0).any():
            print('warning: negative values in slices')
        
        # set sfc_mode
        if len(self.slices.shape) == 3:
            self.sfc_mode = '2d'
        elif len(self.slices.shape) == 4:
            self.sfc_mode = '3d'
        else:
            print(f'error: incorrect self.slices.shape = {self.slices.shape}')
            
    def _map_to_sfc(self,
                    sfc_type = 'hilbert', *args,**kwargs):
        
        if sfc_type == 'hilbert':
            if self.sfc_mode == '2d':
                self.slices = vectorize(padding)(self.slices) # ensure slices are padded
                self.sfcs = vectorize(hilbert2d_sfc)(self.slices)
            
            elif self.sfc_mode == '3d':
                print('not implemented')
            
        elif sfc_type == 'gilbert':
            pass
        
        pass
    
    def _clean_sfc(sfc, n: int = 10):
        '''
        drops null bins of length n in the sfc signal (aut_artefact8.m)
        '''
        
        imax = np.floor(len(sfc)/n).astype(int)

        swv = np.lib.stride_tricks.sliding_window_view
        out = swv(sfc,window_shape=(n,))

        subout = out[::n]

        subout2 = subout[list(map(lambda x: len(set(x))>1,subout))]
        
        sfc_clean = subout2.reshape(-1)
        
        ix_post = (len(sfc) % n)
        
        if ix_post > 0:
            sfc_clean = np.append(sfc_clean,sfc[-ix_post:])
        
        return sfc_clean
    
    def _run_mfdfa(self,
                   sfcs = None,
                   clean_bin_size : int = 10,
                   min_scale : int = 5,
                   n_scales : int = 30,
                   fit_order : int = 2,
                   qs = None,
                   max_scale = None,
                   scales = None,
                   hurst_type : str = 'small_scales',
                   mfdfa_type = 'py', *args,**kwargs):
        '''
        MFDFA
        max_scale = None - infer from data
        '''

        if sfcs is None:
            sfcs = self.sfcs
        
        qs = np.concatenate((np.arange(-4,0,.2),np.arange(0.2,4.2,.2)))
        self.qs = qs
        
        def calc_MFDFA(sfc):
            '''
            calc using external MFDFA module
            '''
            
            max_scale = np.round(len(sfc)/5).astype(int)

            ds = (np.log(max_scale)-np.log(min_scale))/n_scales
            scales = np.arange(np.log(min_scale),
                               np.log(max_scale),
                               ds)
            scales = np.append(scales,scales[-1]+ds)
            scales = np.round(np.exp(scales)).astype(int)
            self.scales = scales
            
            # drop null signal from sfc before MFDFA
            sfc = mfractal_mri._clean_sfc(sfc, n = clean_bin_size)
            
            if sfc.shape[0] < clean_bin_size:
                # print('warning: no signal')
                dfa = np.nan*np.ones((len(scales),len(qs)))
            
            else:
                lag, dfa = mfdfa_py(sfc, lag = scales, q = qs, order = fit_order)
            
            return dfa

        def calc_aut_MFDFA(sfc):
            '''
            calc using own routines refactored from aut_MFDFA.m matlab script
            '''
            pass
            
        calc_dict = {'py': calc_MFDFA, 'matlab': calc_aut_MFDFA}
        if mfdfa_type in calc_dict.keys():
            calc = calc_dict[mfdfa_type]
        else:
            print(f'error: unknown mfdfa_type = {mfdfa_type}')
            
        self.Fqs = vectorize(calc)(sfcs)
    
    def _get_hursts(self,nmin=0,nmax=0, *args,**kwargs):
        out = vectorize(ghurst)(self.Fqs,self.scales,nmin=nmin,nmax=nmax)
        self.H, self.H_errs = out[:,0,:], out[:,1,:]
        return self.H
    
    def pipeline(self,*args,**kwargs):
        self.load_scan(*args,**kwargs)
        self._slice_scan(*args,**kwargs)
        self._map_to_sfc(*args,**kwargs)
        self._run_mfdfa(*args,**kwargs)
        return self._get_hursts(*args,**kwargs)