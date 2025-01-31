from typing import Union, List, Tuple
import numpy as np

from fsca.const import *
from fsca.sfc import padding, hilbert2d_sfc,hilbert3d_sfc, ddsfc2d
from fsca.mfdfa import mfdfa_py, mfdfa_matlab, ghurst
from fsca.mfdfa import aut_artefact8
from fsca.mfdfa import spectrum, spectrum_params
from fsca.tools import vectorize,mvectorize,mvectorize2

class base_pipeline:

    def __init__(self,
                 output : str = None,
                 verbose : bool = False) -> None:

        self.data = None
        self.data_dim = 0
        self.sfcs = None
        self.fqs = None
        self.scales = None
        self.qorders = None
        self.ghs = None
        self.ghs_res = None
        
        self.alphas = None
        self.fs = None
        self.Ds = None
        self.As = None
        
        self._proc_mode = None

        self.output = output
        self.verbose = verbose

    def run(self, data : np.ndarray) -> None:
        pass

    def get_result(self):
        h_ix = np.argwhere(self.qorders == 2)
        if len(h_ix) > 0:
            h_ix = h_ix.item()
            return self.ghs[...,h_ix]

        print('error: cannot extract the Hurst exponent from given qorders')
        return None

    @staticmethod
    def _is_data_correct(data : np.ndarray,
                         data_dim : int) -> bool:

        if len(data.shape) == data_dim or len(data.shape) == data_dim+1:
            return True
        return False

    def load_data(self,
                  data : np.ndarray, **kwargs) -> None:
        if base_pipeline._is_data_correct(data, data_dim = self.data_dim):
            self.data = data
        else:
            raise ValueError(f'wrong data shape {data.shape}')

    def set_proc_mode(self, **kwargs) -> None:
        if len(self.data.shape) == self.data_dim:
            self._proc_mode = 'single'
        elif len(self.data.shape) == self.data_dim + 1:
            self._proc_mode = 'batch'
        else:
            raise ValueError(f'cannot set _proc_mode; wrong data.shape = {self.data.shape}')

    @staticmethod
    def pack_array(array: np.ndarray,
                   final_dims: int = 2) -> Tuple[tuple,np.ndarray]:
        array_shape = array.shape
        array = array.reshape(-1,*array_shape[-final_dims:])
        return array, array_shape

    @staticmethod
    def unpack_array(array: np.ndarray, array_shape: tuple) -> np.ndarray:
        return array.reshape(*array_shape)

    @staticmethod
    def _slice_single_data(data : np.ndarray,
                           slice_axis : str = DEFAULT_SLICE_AXIS, **kwargs) -> np.ndarray:

        f'''
        Slice the given 3D data along the specified axis.
        If slice_axis is 'none', a 3D hyper-slice will be created instead.
        Normalization of slices is conducted a) separately in each slice,
        or b) globally for the whole scan.
        If `quantize_val` = True, round the normalized values to the nearest integer.

        Parameters:
        ----------
        data : np.ndarray
            The 3D data to be sliced.
        slice_axis : str, optional (default = {DEFAULT_SLICE_AXIS})
            The axis along which to slice the scan, one of ['x', 'y', 'z', 'none'].

        Returns:
        -------
        np.ndarray
            The sliced data as a numpy array.
        '''



        if slice_axis == 'z':
            slices = np.transpose(data,axes=(2,0,1))
        elif slice_axis == 'x':
            slices = np.transpose(data,axes=(0,1,2))
        elif slice_axis == 'y':
            slices = np.transpose(data,axes=(1,0,2))
        elif slice_axis in NONE_VALS:
            slices = data
        else:
            print(f'error: unknown slice_axis = {slice_axis}')

        return slices

    @staticmethod
    def _slice_batch_data(data : np.ndarray,
                          slice_axis : str = DEFAULT_SLICE_AXIS, **kwargs) -> np.ndarray:

        def calc(data):
            return base_pipeline._slice_single_data(data, slice_axis = slice_axis, **kwargs)

        return vectorize(calc)(data)

    @staticmethod
    def _norm_single_slice(slice : np.ndarray,
                            min_norm_val : float = 0.0,
                            max_norm_val : float = 255.0,
                            quantize_val : bool = False,
                            eps : float = 1e-20,
                            **kwargs) -> np.ndarray:

        f'''
        Slice the given 3D MRI scan along the specified axis.
        If slice_axis is 'none', a 3D hyper-slice will be created instead.
        Normalization of slices is conducted a) separately in each slice,
        or b) globally for the whole scan.
        If `quantize_val` = True, round the normalized values to the nearest integer.

        Parameters:
        ----------
        slice : np.ndarray
            The 2D/3D slice to be sliced.
        min_norm_val : float, optional (default = 0.0)
            The minimum value to which the slices are normalized.
        max_norm_val : float, optional (default = 255.0)
            The maximum value to which the slices are normalized.
        quantize_val : bool, optional (default = False)
            Whether to round the normalized values to the nearest integer.
        eps : float, optional (default = 1e-20)
            A small positive constant to avoid division by zero.

        Returns:
        -------
        np.ndarray
            The normalized slice as a numpy array.
        '''

        minv = slice.min(keepdims=True)
        maxv = slice.max(keepdims=True)
        x = (slice - minv) / (maxv - minv + eps)
        slice = max_norm_val * x + min_norm_val * (1-x)

        if quantize_val:
            slice = np.round(slice).astype(int)

        if (slice < 0).any():
            print('warning: negative values in slices')

        return slice

    @staticmethod
    def _norm_batch_slices(slices : np.ndarray,
                            min_norm_val : float = 0.0,
                            max_norm_val : float = 255.0,
                            quantize_val : bool = False,
                            eps : float = 1e-20,
                            **kwargs) -> np.ndarray:

        def calc(slice):
            return base_pipeline._norm_single_slice(slice = slice,
                                                    min_norm_val = min_norm_val,
                                                    max_norm_val = max_norm_val,
                                                    quantize_val = quantize_val,
                                                    eps = eps)

        return vectorize(calc)(slices)

    @staticmethod
    def _map_data_to_sfc_hilbert2d(data : np.ndarray) -> np.ndarray:
        '''
        Convert the given 2D data to a 1D signal by using the 2D Hilbert space-filling curve.
        The input slices should have three dimensions, i.e., shape (n_data, height, width).
        The function pads each (height,width) slice with zeros to reach a square size.

        Parameters:
        ----------
        data : np.ndarray
            The 2D data to be converted to 1D space-filling curves
            with shape (n_data, height, width).

        Returns:
        -------
        np.ndarray
            The 1D SFC signal of shape (n_data, height*width)
        '''

        data = vectorize(padding)(data)
        return vectorize(hilbert2d_sfc)(data)

    @staticmethod
    def _map_data_to_sfc_hilbert3d(data: np.ndarray) -> np.ndarray:
        '''
        TODO
        '''

        data = vectorize(padding)(data)
        return vectorize(hilbert3d_sfc)(data)

    @staticmethod
    def _map_data_to_sfc_ddsfc2d(data : np.ndarray) -> np.ndarray:
        '''
        TODO
        '''
        data = vectorize(padding)(data)
        return vectorize(ddsfc2d)(data)

    @staticmethod
    def _clean_sfc(sfc : np.ndarray,
                   window_size : int = DEFAULT_WINDOW_SIZE) -> np.ndarray:

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
                      scales_params : tuple = DEFAULT_SCALES) -> np.ndarray:

        '''
        The function infers a set of (logarithmic) scales compatible with the given SFC signal.
        It takes as input a tuple of the form (min_scale,max_scale,n_scales)
        with the minimum scale, the maximum scale and the number of scales to consider.
        If max_scale = None, the function sets it to a default value
        of 1/5th of the SFC signal length. Function returns an array of inferred scales.

        Parameters:
        ----------
        sfc : np.ndarray
            The 1D space-filling curve for which scales need to be inferred.
        scales_params : tuple, optional
            A tuple with which the logarithmically equispaced scales
            compatible with the SFC signal are provided.
            Function accepts a 3-tuple (min_scale, max_scale, and n_scales).

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
    def _calc_single_mfdfa(sfc : np.ndarray,
                      scales : Union[tuple,List] = DEFAULT_SCALES,
                      qorders : np.ndarray = DEFAULT_QORDERS,
                      window_size_cleaning : int = DEFAULT_WINDOW_SIZE,
                      fit_order : int = DEFAULT_FIT_ORDER,
                      mfdfa_lib : str = DEFAULT_MFDFA_LIB) -> Tuple[np.ndarray,np.ndarray]:

        '''
        Applies the multifractal detrended fluctuation analysis (MFDFA) to a single SFC signal.

        Parameters:
        ----------
        sfc : np.ndarray
            The 1D SFC signal for which the MFDFA algorithm is applied.
        scales : tuple or list, optional
            The scales of the SFC signal provided either as a tuple or a list.
            Tuple should be in the form (min_scale, max_scale, and n_scales)
            and is used by `_infer_scales` function. The list should contain the scales.
        qorders : np.ndarray, optional
            An array of fluctuation function orders for which the MFDFA is computed.
        window_size_cleaning : int, optional
            The window size for the signal cleaning. Used by the `_clean_sfc` function.
        fit_order : int, optional
            The order of polynomial used in the MFDFA detrending fit.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use.
            Possible values are "py" for an external MFDFA module,
            "matlab" for own routines refactored from MATLAB.

        Returns:
        -------
        fqs : np.ndarray
            the multifractal detrended fluctuations computed across `qorders` and `scales`,
        scales : np.ndarray
            the scales.
        '''

        # drop null signal from sfc before MFDFA
        # infer scales before cleaning!
        if isinstance(scales,tuple):
            scales = pipeline_2d._infer_scales(sfc = sfc, scales_params = scales)

        sfc = pipeline_2d._clean_sfc(sfc, window_size = window_size_cleaning)

        if sfc.shape[0] < window_size_cleaning:
            # print('warning: no signal')
            dfa = np.nan*np.ones((len(scales),len(qorders)))
            # scales = np.array([1]*len(scales))

        else:

            mfdfa_args = dict(timeseries = sfc, lag = scales, q = qorders, order = fit_order)

            if mfdfa_lib == 'py':
                _, dfa = mfdfa_py(**mfdfa_args)

            elif mfdfa_lib == 'matlab':
                _, dfa = mfdfa_matlab(**mfdfa_args)
            else:
                print(f'error: unknown mfdfa_lib = {mfdfa_lib}')

        dfa = dfa.astype('float')
        scales = scales.astype('int')

        return dfa, scales

    @staticmethod
    def _calc_batch_mfdfa(sfcs : np.ndarray,
                     scales : Union[tuple,List] = DEFAULT_SCALES,
                     qorders : np.ndarray = DEFAULT_QORDERS,
                     window_size_cleaning : int = DEFAULT_WINDOW_SIZE,
                     fit_order : int = DEFAULT_FIT_ORDER,
                     mfdfa_lib : str = DEFAULT_MFDFA_LIB) -> Tuple[np.ndarray,np.ndarray]:
        '''
        Perform the MFDFA on the SFC signals.
        Accepts two types of arrays. A 2D array of size (N,M)
        forming a set of N SFC signals each of length M.
        Returns the multifractal fluctuations.

        Parameters:
        -----------
        sfcs : np.ndarray
            The 1D SFC signal for which the MFDFA algorithm is applied.
            Accepts a 2D array of size (N,M)
            where N is the number of SFC signals each of length M.
        scales : tuple or list, optional
            The scales to use in the MFDFA calculation.
            Either a list/array of scales or a tuple (min_scale,max_scale,n_scales).
            If max_scale = None, infer from the series length M.
        qorders : np.ndarray, optional
            An array of fluctuation function orders for which the MFDFA is computed.
        window_size_cleaning : int, optional
            The window size for the signal cleaning. Used by the `_clean_sfc` function.
        fit_order : int, optional
            The order of polynomial used in the MFDFA detrending fit.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use.
            Possible values are "py" for an external MFDFA module,
            "matlab" for own routines refactored from MATLAB.

        Returns:
        --------
        fqs : np.ndarray
            An array of size (N, len(qorders), len(scales)),
            containing the multifractal fluctuations for N SFC signals.
        scales : np.ndarray
            An array of inferred scales for the given SFC signals.
            Has shapes (len(scales),) or (len(scales),1)
            depending on whether the input SFC signal was 1D or 2D.
        '''

        def calc(sfc):
            return pipeline_2d._calc_single_mfdfa(sfc,
                                             scales = scales,
                                             qorders = qorders,
                                             window_size_cleaning = window_size_cleaning,
                                             fit_order = fit_order,
                                             mfdfa_lib = mfdfa_lib)

        fqs, scaless = mvectorize(calc)(sfcs)
        return fqs, scaless

    @staticmethod
    def _calc_single_ghurst(fq : np.ndarray,
                    scales : np.ndarray,
                    min_scale_ix : int = None,
                    max_scale_ix : int = None,
                    **kwargs) -> Tuple[np.ndarray,np.ndarray]:

        '''
        Calculates the generalized Hurst exponent from the multifractal fluctuations array.
        Accepts 1D, 2D and 3D arrays.
        For 2D and 3D arrays, Hursts are calculated wrt. the first index.
        Used in conjunction with the `_calc_mfdfa` method.

        Parameters:
        ----------
        fq : np.ndarray
            The 1D array of multifractal fluctuations.
        scales : np.ndarray
            The array of scales for which the multifractal fluctuation array was calculated.
        min_scale_ix : int, optional
            The index of the first scale included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale included in the fitting of the generalized Hurst exponent.

        Returns:
        -------
        gh : np.ndarray
            generalized Hurst exponent for each fluctuation order q.
        gh_res : np.ndarray
            fitting residuals of the generalized Hurst exponent for each fluctuation order q.

        Notes:
        -----
        For a default array of orders `qorders` spanned over (-4,4) with step 0.2 without q = 0.0,
        the classic Hurst exponent for q=2 is given by ghs[29].
        '''

        # fq = fq.reshape(-1,1)
        gh,gh_res = ghurst(Fq = fq,
                            scales = scales,
                            min_scale_ix = min_scale_ix,
                            max_scale_ix = max_scale_ix)

        return gh, gh_res

    @staticmethod
    def _calc_batch_ghurst(fqs : np.ndarray,
                    scales : np.ndarray,
                    min_scale_ix : int = None,
                    max_scale_ix : int = None,
                    **kwargs) -> Tuple[np.ndarray,np.ndarray]:
        '''
        fqs: (n_batch, n_scales, n_orders)
        '''

        def calc(fqs,scales):
            return ghurst(Fq = fqs,
                            scales = scales,
                            min_scale_ix = min_scale_ix,
                            max_scale_ix = max_scale_ix)

        ghs,ghs_res = mvectorize2(calc)(fqs,scales)

        return ghs, ghs_res

    @staticmethod
    def _calc_single_spectrum(qorders: np.ndarray, 
                              gh: np.ndarray):
        '''
        
        '''
        alpha, f = spectrum(qorders,gh)
        return alpha, f     

    @staticmethod
    def _calc_batch_spectrum(qorders: np.ndarray, ghs: np.ndarray):
        '''
        '''
        if len(qorders.shape) == 1:
            qorders = np.stack([qorders for _ in range(ghs.shape[0])])
            
        alphas, fs = mvectorize2(spectrum)(qorders,ghs)
        return alphas, fs

    @staticmethod
    def _calc_single_spectrum_params(alpha: np.ndarray, 
                                     f: np.ndarray):
        '''
        
        '''
        D,A = spectrum_params(alpha, f)
        return alpha, f     

    @staticmethod
    def _calc_batch_spectrum_params(alphas: np.ndarray, 
                                    fs: np.ndarray):
        '''
        
        '''
        Ds, As = mvectorize2(spectrum_params)(alphas,fs)
        return Ds, As

    def calc_mfdfa(self):
        pass

    def calc_ghurst(self):
        pass
        
    def calc_falpha(self, **kwargs) -> None:
        print('warning: calc_falpha implementation is in an alpha stage')
        self.alphas, self.fs = self._calc_batch_spectrum(qorders = self.qorders, ghs = self.ghs)
        self.Ds, self.As = self._calc_batch_spectrum_params(alphas = self.alphas, fs = self.fs)
        
    def run(self):
        pass
        
class pipeline_2d(base_pipeline):
    '''
    2d process pipeline
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_dim = 2

    def map_data_to_sfc2d(self,
                          sfc_type : str = DEFAULT_SFC_TYPE,
                          **kwargs) -> None:
        '''
        Convert data to 2D SFC.

        Parameters
        ----------
        sfc_type : str, optional
            The type of space-filling algorithm to use. Possible values are ['hilbert','hilbert3d','gilbert','data-driven'].

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

        if self._proc_mode == 'single':
            data = np.expand_dims(self.data,0)
        else:
            data = self.data

        if sfc_type == 'hilbert':
            sfcs = pipeline_2d._map_data_to_sfc_hilbert2d(data = data)

        elif sfc_type == 'gilbert':
            raise ValueError('error: gilbert not implemented')

        elif sfc_type == 'data-driven':
            sfcs = pipeline_2d._map_data_to_sfc_ddsfc2d(data = data)

        else:
            raise ValueError(f'error: wrong sfc_type = {sfc_type}')

        if self._proc_mode == 'single':
            sfcs = sfcs.squeeze()

        self.sfcs = sfcs

    def calc_mfdfa(self,
                   scales : np.ndarray = DEFAULT_SCALES,
                   qorders : np.ndarray = DEFAULT_QORDERS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   **kwargs) -> None:
        '''
        Calculate multifractal detrended fluctuation analysis (MFDFA)
        for sets of `scales` and fluctuation orders `qorders` using the `mfdfa_lib` function.

        Parameters:
        -----------
        scales : tuple or list, optional
            The range of scales to use when calculating the MFDFA, used by `_calc_mfdfa` function.
        qorders : ndarray, optional
            The list of fluctuation orders q, used by `_calc_mfdfa` function.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use.
            Possible values are "py" for an external MFDFA module,
            "matlab" for own routines refactored from MATLAB.

        Returns:
        --------
        None

        Notes:
        ------
        The MFDFA result are stored in `self.fqs` (multifractal fluctuations),
        `self.qorders` and `self.scales` (orders and scales respectively).
        '''

        if self._proc_mode == 'single':
            sfcs = np.expand_dims(self.sfcs,0)
        else:
            sfcs = self.sfcs

        fqs, scales = pipeline_2d._calc_batch_mfdfa(sfcs = sfcs,
                                              scales = scales,
                                              qorders = qorders,
                                              mfdfa_lib = mfdfa_lib,
                                              **kwargs)

        if self._proc_mode == 'single':
            fqs = fqs.squeeze()
            scales = scales.squeeze()

        self.fqs = fqs
        self.scales = scales
        self.qorders = qorders

    def calc_ghurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = DEFAULT_SCALE_PRESET,
                   **kwargs) -> None:

        '''
        Calculate the generalized Hurst exponents for the previously computed MFDFA results.

        Parameters
        ----------
        min_scale_ix : int, optional
            The index of the first scale included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale included in the fitting of the generalized Hurst exponent.
        scale_preset : str, optional
            Preset to use when setting min_scale_ix and max_scale_ix automatically.
            Valid options are "all_scales", "small_scales" and "large_scales".

        Returns
        -------
        None

        Notes:
        ------
        Generalized Hurst exponents are stored in:
        `self.ghs` (values) and `self.ghs_res` (fitting residuals).
        For a default array of orders `qorders` spanned over (-4,4) with step 0.2 without q = 0.0,
        the classic Hurst exponent for q=2 is given by self.ghs[:,29].
        '''

        if self.verbose:
            print('calc_ghurst()...')

        if scale_preset == 'all_scales':
            min_scale_ix = min_scale_ix
            max_scale_ix = max_scale_ix

        elif scale_preset == 'small_scales':
            min_scale_ix = None
            # max_scale_ix = 11
            max_scale_ix = DEFAULT_SMALL_SCALES_INTERVAL_SIZE if max_scale_ix is None else max_scale_ix

        elif scale_preset == 'large_scales':
            min_scale_ix = -DEFAULT_LARGE_SCALES_INTERVAL_SIZE if min_scale_ix is None else min_scale_ix
            max_scale_ix = None

        else:
            print(f'warning: unknown scale_preset = {scale_preset}')

        if self._proc_mode == 'single':
            fqs = np.expand_dims(self.fqs,0)
            scales = np.expand_dims(self.scales,0)
        else:
            fqs = self.fqs
            scales = self.scales

        ghs, ghs_res = pipeline_2d._calc_batch_ghurst(fqs = fqs,
                                                      scales = scales,
                                                      min_scale_ix = min_scale_ix,
                                                      max_scale_ix = max_scale_ix)

        if self._proc_mode == 'single':
            ghs = ghs.squeeze()
            ghs_res = ghs_res.squeeze()

        self.ghs, self.ghs_res = ghs, ghs_res
        
    def run(self, data: np.ndarray, **kwargs) -> None:
        self.load_data(data, **kwargs)
        self.set_proc_mode(**kwargs) # set processing mode (single or batch)

        self.map_data_to_sfc2d(**kwargs) # output -> self.sfcs
        self.calc_mfdfa(**kwargs) # output -> self.fqs, self.scales, self.qorders
        self.calc_ghurst(**kwargs) # output -> self.ghs, self.ghs_res

class pipeline_2x1d(pipeline_2d):
    '''
    2d + 1d process pipeline
    '''

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.data_dim = 3

    def map_data_to_sfc2d(self,
                          sfc_type : str = DEFAULT_SFC_TYPE,
                          **kwargs) -> None:
        '''
        Convert data to 2D SFC.

        Parameters
        ----------
        sfc_type : str, optional
            The type of space-filling algorithm to use. Possible values are ['hilbert','hilbert3d','gilbert','data-driven'].

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

        if self._proc_mode == 'batch':
            data, data_shape = pipeline_2x1d.pack_array(self.data)
        else:
            data = self.data

        if sfc_type == 'hilbert':
            sfcs = pipeline_2d._map_data_to_sfc_hilbert2d(data = data)

        elif sfc_type == 'gilbert':
            raise ValueError('error: gilbert not implemented')

        elif sfc_type == 'data-driven':
            sfcs = pipeline_2d._map_data_to_sfc_ddsfc2d(data = data)

        else:
            raise ValueError(f'error: wrong sfc_type = {sfc_type}')

        if self._proc_mode == 'batch':
            sfcs_shape = sfcs.shape
            sfcs = pipeline_2x1d.unpack_array(sfcs,data_shape[:2]+sfcs.shape[-1:])

        self.sfcs = sfcs

    def calc_mfdfa(self,
                   scales : np.ndarray = DEFAULT_SCALES,
                   qorders : np.ndarray = DEFAULT_QORDERS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   **kwargs) -> None:
        '''
        Calculate multifractal detrended fluctuation analysis (MFDFA)
        for sets of `scales` and fluctuation orders `qorders` using the `mfdfa_lib` function.

        Parameters:
        -----------
        scales : tuple or list, optional
            The range of scales to use when calculating the MFDFA, used by `_calc_mfdfa` function.
        qorders : ndarray, optional
            The list of fluctuation orders q, used by `_calc_mfdfa` function.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use.
            Possible values are "py" for an external MFDFA module,
            "matlab" for own routines refactored from MATLAB.

        Returns:
        --------
        None

        Notes:
        ------
        The MFDFA result are stored in `self.fqs` (multifractal fluctuations),
        `self.qorders` and `self.scales` (orders and scales respectively).
        '''

        if self._proc_mode == 'batch':
            sfcs, sfcs_shape = pipeline_2x1d.pack_array(self.sfcs, final_dims = 1)
        else:
            sfcs = self.sfcs

        fqs, scales = pipeline_2x1d._calc_batch_mfdfa(sfcs = sfcs,
                                              scales = scales,
                                              qorders = qorders,
                                              mfdfa_lib = mfdfa_lib,
                                              **kwargs)

        if self._proc_mode == 'batch':
            fqs_shape = sfcs_shape[:2]+fqs.shape[-2:]
            fqs = fqs.reshape(fqs_shape)
            scales_shape = sfcs_shape[:2]+scales.shape[-1:]
            scales = scales.reshape(scales_shape)
            # fqs = fqs.squeeze()
            # scales = scales.squeeze()

        self.fqs = fqs
        self.scales = scales
        self.qorders = qorders

    def calc_ghurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = DEFAULT_SCALE_PRESET,
                   **kwargs) -> None:

        '''
        Calculate the generalized Hurst exponents for the previously computed MFDFA results.

        Parameters
        ----------
        min_scale_ix : int, optional
            The index of the first scale included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale included in the fitting of the generalized Hurst exponent.
        scale_preset : str, optional
            Preset to use when setting min_scale_ix and max_scale_ix automatically.
            Valid options are "all_scales", "small_scales" and "large_scales".

        Returns
        -------
        None

        Notes:
        ------
        Generalized Hurst exponents are stored in:
        `self.ghs` (values) and `self.ghs_res` (fitting residuals).
        For a default array of orders `qorders` spanned over (-4,4) with step 0.2 without q = 0.0,
        the classic Hurst exponent for q=2 is given by self.ghs[:,29].
        '''

        if self.verbose:
            print('calc_ghurst()...')

        if scale_preset == 'all_scales':
            min_scale_ix = min_scale_ix
            max_scale_ix = max_scale_ix

        elif scale_preset == 'small_scales':
            min_scale_ix = None
            # max_scale_ix = 11
            max_scale_ix = DEFAULT_SMALL_SCALES_INTERVAL_SIZE if max_scale_ix is None else max_scale_ix

        elif scale_preset == 'large_scales':
            min_scale_ix = -DEFAULT_LARGE_SCALES_INTERVAL_SIZE if min_scale_ix is None else min_scale_ix
            max_scale_ix = None

        else:
            print(f'warning: unknown scale_preset = {scale_preset}')

        if self._proc_mode == 'batch':
            fqs, fqs_shape = pipeline_2x1d.pack_array(self.fqs)
            scales, scales_shape = pipeline_2x1d.pack_array(self.scales, final_dims = 1)
            # fqs = np.expand_dims(self.fqs,0)
            # scales = np.expand_dims(self.scales,0)
        else:
            fqs = self.fqs
            scales = self.scales

        ghs, ghs_res = pipeline_2x1d._calc_batch_ghurst(fqs = fqs,
                                                      scales = scales,
                                                      min_scale_ix = min_scale_ix,
                                                      max_scale_ix = max_scale_ix)

        if self._proc_mode == 'batch':
            ghs = ghs.reshape(fqs_shape[:2]+ghs.shape[-1:])
            ghs_res = ghs.reshape(fqs_shape[:2]+ghs_res.shape[-1:])

        self.ghs, self.ghs_res = ghs, ghs_res

class pipeline_3d(base_pipeline):
    '''
    3d process pipeline
    '''

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_dim = 3
        self.slices = None
        self.slice_axis = ''
        self.sfc_dim = None

    def slice_data(self,
                   slice_axis : str = DEFAULT_SLICE_AXIS, **kwargs) -> None:

        if self._proc_mode == 'single':
            data = np.expand_dims(self.data,0)
        else:
            data = self.data

        slices = super()._slice_batch_data(data = data,
                                            slice_axis = slice_axis, **kwargs)

        if self._proc_mode == 'single':
            slices = slices.squeeze()
        else:
            pass

        self.slices = slices
        self.slice_axis = slice_axis
        self.sfc_dim = {'x':2,'y':2,'z':2,None:3,'none':3,'None':3}[slice_axis]

    def norm_slices(self, **kwargs):

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            slices = self.slices
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            slices, slices_shape = super().pack_array(self.slices,final_dims=2)
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            slices = np.expand_dims(self.slices,0)
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            slices = self.slices

        slices = super()._norm_batch_slices(slices,**kwargs)

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            pass
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            slices = slices.reshape(slices_shape[:2]+slices.shape[-2:])
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            slices = slices.squeeze()
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            pass

        self.slices = slices

    def map_data_to_sfc(self,
                        sfc_type : str = DEFAULT_SFC_TYPE,
                        **kwargs) -> None:

        if self.sfc_dim == 2:

            if self._proc_mode == 'batch':
                slices, slices_shape = super().pack_array(self.slices,final_dims=2)
            else:
                slices = self.slices

            if sfc_type == 'hilbert':
                sfcs = super()._map_data_to_sfc_hilbert2d(data = slices)

            elif sfc_type == 'data-driven':
                sfcs = super()._map_data_to_sfc_ddsfc2d(data = slices)

            else:
                raise ValueError(f'error: wrong sfc_type = {sfc_type}')

            if self._proc_mode == 'batch':
                sfcs_shape = slices_shape[:2]+sfcs.shape[-1:]
                sfcs = super().unpack_array(sfcs,sfcs_shape)
            else:
                pass

        elif self.sfc_dim == 3:

            if self._proc_mode == 'batch':
                slices = self.slices
            else:
                slices = np.expand_dims(self.slices,0)

            # print(slices.shape)
            if sfc_type == 'hilbert':
                sfcs = super()._map_data_to_sfc_hilbert3d(data = slices)
                # print(sfcs.shape)

            elif sfc_type == 'data-driven':
                raise ValueError('error: data-driven in 3d not implemented')

            else:
                raise ValueError(f'error: wrong sfc_type = {sfc_type}')

            if self._proc_mode == 'batch':
                pass
            else:
                sfcs = sfcs.squeeze()
            # print(sfcs.shape)

        self.sfcs = sfcs

    def calc_mfdfa(self,
                   scales : np.ndarray = DEFAULT_SCALES,
                   qorders : np.ndarray = DEFAULT_QORDERS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   **kwargs) -> None:
        '''
        Calculate multifractal detrended fluctuation analysis (MFDFA)
        for sets of `scales` and fluctuation orders `qorders` using the `mfdfa_lib` function.

        Parameters:
        -----------
        scales : tuple or list, optional
            The range of scales to use when calculating the MFDFA, used by `_calc_mfdfa` function.
        qorders : ndarray, optional
            The list of fluctuation orders q, used by `_calc_mfdfa` function.
        mfdfa_lib : str, optional
            The type of MFDFA procedure to use.
            Possible values are "py" for an external MFDFA module,
            "matlab" for own routines refactored from MATLAB.

        Returns:
        --------
        None

        Notes:
        ------
        The MFDFA result are stored in `self.fqs` (multifractal fluctuations),
        `self.qorders` and `self.scales` (orders and scales respectively).
        '''

        if self.sfc_dim == 2:

            if self._proc_mode == 'batch':
                sfcs, sfcs_shape = super().pack_array(self.sfcs,final_dims=1)
            else:
                sfcs = self.sfcs

        elif self.sfc_dim == 3:

            if self._proc_mode == 'batch':
                sfcs = self.sfcs
            else:
                sfcs = np.expand_dims(self.sfcs,0)


        fqs, scales = super()._calc_batch_mfdfa(sfcs = sfcs,
                                              scales = scales,
                                              qorders = qorders,
                                              mfdfa_lib = mfdfa_lib)

        if self.sfc_dim == 2:

            if self._proc_mode == 'batch':
                fqs = fqs.reshape(sfcs_shape[:-1]+fqs.shape[-2:])
                scales = scales.reshape(sfcs_shape[:-1]+scales.shape[-1:])
            else:
                pass

        elif self.sfc_dim == 3:

            if self._proc_mode == 'batch':
                pass
            else:
                fqs = fqs.squeeze()
                scales = scales.squeeze()

        self.fqs = fqs
        self.scales = scales
        self.qorders = qorders

    def calc_ghurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = DEFAULT_SCALE_PRESET,
                   **kwargs) -> None:

        '''
        Calculate the generalized Hurst exponents for the previously computed MFDFA results.

        Parameters
        ----------
        min_scale_ix : int, optional
            The index of the first scale included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale included in the fitting of the generalized Hurst exponent.
        scale_preset : str, optional
            Preset to use when setting min_scale_ix and max_scale_ix automatically.
            Valid options are "all_scales", "small_scales" and "large_scales".

        Returns
        -------
        None

        Notes:
        ------
        Generalized Hurst exponents are stored in:
        `self.ghs` (values) and `self.ghs_res` (fitting residuals).
        For a default array of orders `qorders` spanned over (-4,4) with step 0.2 without q = 0.0,
        the classic Hurst exponent for q=2 is given by self.ghs[:,29].
        '''

        if self.verbose:
            print('calc_ghurst()...')

        if scale_preset == 'all_scales':
            min_scale_ix = min_scale_ix
            max_scale_ix = max_scale_ix

        elif scale_preset == 'small_scales':
            min_scale_ix = None
            # max_scale_ix = 11
            max_scale_ix = DEFAULT_SMALL_SCALES_INTERVAL_SIZE if max_scale_ix is None else max_scale_ix

        elif scale_preset == 'large_scales':
            min_scale_ix = -DEFAULT_LARGE_SCALES_INTERVAL_SIZE if min_scale_ix is None else min_scale_ix
            max_scale_ix = None

        else:
            print(f'warning: unknown scale_preset = {scale_preset}')

        if self.sfc_dim == 2:

            if self._proc_mode == 'batch':
                fqs, fqs_shape = super().pack_array(self.fqs)
                scales, scales_shape = super().pack_array(self.scales,final_dims=1)
            else:
                fqs = self.fqs
                scales = self.scales

        elif self.sfc_dim == 3:

            if self._proc_mode == 'batch':
                fqs = self.fqs
                scales = self.scales
            else:
                fqs = np.expand_dims(self.fqs,0)
                scales = np.expand_dims(self.scales,0)

        ghs, ghs_res = super()._calc_batch_ghurst(fqs = fqs,
                                                      scales = scales,
                                                      min_scale_ix = min_scale_ix,
                                                      max_scale_ix = max_scale_ix)

        if self.sfc_dim == 2:

            if self._proc_mode == 'batch':
                ghs = ghs.reshape(fqs_shape[:2]+ghs.shape[-1:])
                ghs_res = ghs_res.reshape(fqs_shape[:2]+ghs_res.shape[-1:])
            else:
                pass

        elif self.sfc_dim == 3:

            if self._proc_mode == 'batch':
                pass
            else:
                ghs = ghs.squeeze()
                ghs_res = ghs_res.squeeze()

        self.ghs, self.ghs_res = ghs, ghs_res

    def run(self, data: np.ndarray, **kwargs) -> None:
        self.load_data(data,**kwargs)
        self.set_proc_mode(**kwargs) # set processing mode (single or batch)
        self.slice_data(**kwargs) # output -> self.slices
        self.norm_slices(**kwargs) # output -> normalized self.slices
        self.map_data_to_sfc(**kwargs) # output -> self.sfcs
        self.calc_mfdfa(**kwargs) # output -> self.fqs, self.scales, self.qorders
        self.calc_ghurst(**kwargs) # output -> self.ghs, self.ghs_res
        # self.calc_falpha(**kwargs) # output -> self.alphas, self.fs, self.Ds, self.As
        
class pipeline_3x1d(pipeline_3d):

    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args,**kwargs)
        self.data_dim = 4

    def slice_data(self,
                   slice_axis : str = DEFAULT_SLICE_AXIS, **kwargs) -> None:

        if self._proc_mode == 'single':
            data = self.data
        else:
            data, data_shape = base_pipeline.pack_array(self.data,final_dims = 3)

        slices = super()._slice_batch_data(data = data,
                                            slice_axis = slice_axis, **kwargs)

        if self._proc_mode == 'single':
            pass
        else:
            slices = slices.reshape(data_shape[:2] + slices.shape[-3:])

        self.slices = slices
        self.slice_axis = slice_axis
        self.sfc_dim = {'x':2,'y':2,'z':2,None:3,'none':3,'None':3}[slice_axis]

    def norm_slices(self, **kwargs):

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            slices, slices_shape = super().pack_array(self.slices,final_dims=2)
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            slices, slices_shape = super().pack_array(self.slices,final_dims=2)
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            slices = self.slices
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            slices, slices_shape = super().pack_array(self.slices,final_dims=3)

        slices = super()._norm_batch_slices(slices,**kwargs)

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            slices = slices.reshape(slices_shape[:2]+slices.shape[-2:])
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            slices = slices.reshape(slices_shape[:3]+slices.shape[-2:])
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            pass
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            slices = slices.reshape(slices_shape[:2]+slices.shape[-3:])


        self.slices = slices

    def map_data_to_sfc(self,
                        sfc_type : str = DEFAULT_SFC_TYPE,
                        **kwargs) -> None:

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            slices, slices_shape = super().pack_array(self.slices,final_dims=2)
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            slices, slices_shape = super().pack_array(self.slices,final_dims=2)
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            slices, slices_shape = super().pack_array(self.slices,final_dims=3)
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            slices, slices_shape = super().pack_array(self.slices,final_dims=3)

        if self.sfc_dim == 2:

            if sfc_type == 'hilbert':
                sfcs = super()._map_data_to_sfc_hilbert2d(data = slices)

            elif sfc_type == 'data-driven':
                sfcs = super()._map_data_to_sfc_ddsfc2d(data = slices)

            else:
                raise ValueError(f'error: wrong sfc_type = {sfc_type}')

        elif self.sfc_dim == 3:

            if sfc_type == 'hilbert':
                sfcs = super()._map_data_to_sfc_hilbert3d(data = slices)

            elif sfc_type == 'data-driven':
                raise ValueError('error: data-driven in 3d not implemented')

            else:
                raise ValueError(f'error: wrong sfc_type = {sfc_type}')

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            sfcs = sfcs.reshape(slices_shape[:2]+sfcs.shape[-1:])
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            sfcs = sfcs.reshape(slices_shape[:3]+sfcs.shape[-1:])
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            pass
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            sfcs = sfcs.reshape(slices_shape[:2]+sfcs.shape[-1:])

        self.sfcs = sfcs

    def calc_mfdfa(self,
                   scales : np.ndarray = DEFAULT_SCALES,
                   qorders : np.ndarray = DEFAULT_QORDERS,
                   mfdfa_lib : str = DEFAULT_MFDFA_LIB,
                   **kwargs) -> None:

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            sfcs, sfcs_shape = super().pack_array(self.sfcs,final_dims=1)
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            sfcs, sfcs_shape = super().pack_array(self.sfcs,final_dims=1)
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            sfcs = self.sfcs
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            sfcs, sfcs_shape = super().pack_array(self.sfcs,final_dims=1)

        fqs, scales = super()._calc_batch_mfdfa(sfcs = sfcs,
                                              scales = scales,
                                              qorders = qorders,
                                              mfdfa_lib = mfdfa_lib,
                                              **kwargs)

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            fqs = fqs.reshape(sfcs_shape[:2]+fqs.shape[-2:])
            scales = scales.reshape(sfcs_shape[:2]+scales.shape[-1:])
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            fqs = fqs.reshape(sfcs_shape[:3]+fqs.shape[-2:])
            scales = scales.reshape(sfcs_shape[:3]+scales.shape[-1:])
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            pass
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            fqs = fqs.reshape(sfcs_shape[:2]+fqs.shape[-2:])
            scales = scales.reshape(sfcs_shape[:2]+scales.shape[-1:])

        self.fqs = fqs
        self.scales = scales
        self.qorders = qorders

    def calc_ghurst(self,
                   min_scale_ix : int = None,
                   max_scale_ix : int = None,
                   scale_preset : str = DEFAULT_SCALE_PRESET,
                   **kwargs) -> None:

        '''
        Calculate the generalized Hurst exponents for the previously computed MFDFA results.

        Parameters
        ----------
        min_scale_ix : int, optional
            The index of the first scale included in the fitting of the generalized Hurst exponent.
        max_scale_ix : int, optional
            The index of the last scale included in the fitting of the generalized Hurst exponent.
        scale_preset : str, optional
            Preset to use when setting min_scale_ix and max_scale_ix automatically.
            Valid options are "all_scales", "small_scales" and "large_scales".

        Returns
        -------
        None

        Notes:
        ------
        Generalized Hurst exponents are stored in:
        `self.ghs` (values) and `self.ghs_res` (fitting residuals).
        For a default array of orders `qorders` spanned over (-4,4) with step 0.2 without q = 0.0,
        the classic Hurst exponent for q=2 is given by self.ghs[:,29].
        '''

        if self.verbose:
            print('calc_ghurst()...')

        if scale_preset == 'all_scales':
            min_scale_ix = min_scale_ix
            max_scale_ix = max_scale_ix

        elif scale_preset == 'small_scales':
            min_scale_ix = None
            # max_scale_ix = 11
            max_scale_ix = DEFAULT_SMALL_SCALES_INTERVAL_SIZE if max_scale_ix is None else max_scale_ix

        elif scale_preset == 'large_scales':
            min_scale_ix = -DEFAULT_LARGE_SCALES_INTERVAL_SIZE if min_scale_ix is None else min_scale_ix
            max_scale_ix = None

        else:
            print(f'warning: unknown scale_preset = {scale_preset}')

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            fqs, fqs_shape = base_pipeline.pack_array(self.fqs,final_dims=2)
            scales, scales_shape = base_pipeline.pack_array(self.scales,final_dims=1)
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            fqs, fqs_shape = base_pipeline.pack_array(self.fqs,final_dims=2)
            scales, scales_shape = base_pipeline.pack_array(self.scales,final_dims=1)
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            fqs = self.fqs
            scales = self.scales
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            fqs, fqs_shape = base_pipeline.pack_array(self.fqs,final_dims=2)
            scales, scales_shape = base_pipeline.pack_array(self.scales,final_dims=1)

        ghs, ghs_res = super()._calc_batch_ghurst(fqs = fqs,
                                                      scales = scales,
                                                      min_scale_ix = min_scale_ix,
                                                      max_scale_ix = max_scale_ix)

        if self._proc_mode == 'single' and self.sfc_dim == 2:
            ghs = ghs.reshape(fqs_shape[:2]+ghs.shape[-1:])
            ghs_res = ghs_res.reshape(fqs_shape[:2]+ghs_res.shape[-1:])
        elif self._proc_mode == 'batch' and self.sfc_dim == 2:
            ghs = ghs.reshape(fqs_shape[:3]+ghs.shape[-1:])
            ghs_res = ghs_res.reshape(fqs_shape[:3]+ghs_res.shape[-1:])
        elif self._proc_mode == 'single' and self.sfc_dim == 3:
            pass
        elif self._proc_mode == 'batch' and self.sfc_dim == 3:
            ghs = ghs.reshape(fqs_shape[:2]+ghs.shape[-1:])
            ghs_res = ghs_res.reshape(fqs_shape[:2]+ghs_res.shape[-1:])

        self.ghs, self.ghs_res = ghs, ghs_res
