from typing import List,Union,Tuple
from pathlib import Path
import numpy as np

from fsca.const import *
from fsca.neuro.preprocessing import Preprocessing
from fsca.neuro.pipeline import pipeline_mri

class pipeline_mri_preprocessing(pipeline_mri):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
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

    def run(self, scan_file : Union[str,Path], **kwargs):
        if isinstance(scan_file,np.ndarray):
            raise ValueError('preprocessing module needs a nii file')

        self.load_scan(scan_file,**kwargs)
        self.run_preprocessing(**kwargs)
        self.scan_norm(**kwargs)
        super().run(data = self.scan,**kwargs)

