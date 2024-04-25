from typing import Union
from pathlib import Path
import nibabel as nib
import numpy as np

from fsca.pipeline_basic import pipeline_3d
from fsca.pipeline_basic import pipeline_3x1d

class pipeline_mri(pipeline_3d):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.scan_file = None
        self.scan = None

    def load_scan(self,
                  scan_file : Union[str,Path,np.ndarray],
                  store_mem : bool = True,
                  **kwargs) -> None:

        '''
        Loads a scan.

        Possible formats:
        - NIFTI file
        - numpy array

        Parameters
        ----------
        scan_file : str, pathlib.Path or numpy.array
            Path to the NIFTI scan file or a numpy array.
        store_mem : bool, optional (default = True)
            Store NIFTI file in memory.
        Raises
        ------
        FileNotFoundError
            If the scan_file path does not exist.
        ValueError
            If the scan_file is of incorrect type.

        Returns
        -------
        None

        Notes
        -----
        If store_mem = True, the resulting scan is stored as a numpy array in `self.scan`.
        '''

        if self.verbose:
            print('load_scan()...')

        if isinstance(scan_file,(str,Path)):
            scan_file = Path(scan_file)
            if not scan_file.exists():
                raise FileNotFoundError(f'error: no scan_file = {scan_file}')
            else:
                if ''.join(scan_file.suffixes) not in ['.nii.gz','.nii']:
                    raise ValueError(f'error: scan_file = {scan_file} is not a NIFTI file')
            self.scan_file = scan_file

            if store_mem:
                self.scan = nib.load(self.scan_file).get_fdata()

        elif isinstance(scan_file,np.ndarray):
            self.scan = scan_file

    def run(self, scan_file : Union[str,Path,np.ndarray], **kwargs):

        self.load_scan(scan_file,**kwargs)
        super().run(data = self.scan,**kwargs)

class pipeline_fmri(pipeline_3x1d):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def load_scan(self,
                  scan_file : Union[str,Path,np.ndarray],
                  store_mem : bool = True,
                  **kwargs) -> None:
        '''
        Loads a scan.

        Possible formats:
        - NIFTI file
        - numpy array

        Parameters
        ----------
        scan_file : str, pathlib.Path or numpy.array
            Path to the NIFTI scan file or a numpy array.
        store_mem : bool, optional (default = True)
            Store NIFTI file in memory.
        Raises
        ------
        FileNotFoundError
            If the scan_file path does not exist.
        ValueError
            If the scan_file is of incorrect type.

        Returns
        -------
        None

        Notes
        -----
        If store_mem = True, the resulting scan is stored as a numpy array in `self.scan`.
        '''

        if self.verbose:
            print('load_scan()...')

        if isinstance(scan_file,(str,Path)):
            scan_file = Path(scan_file)
            if not scan_file.exists():
                raise FileNotFoundError(f'error: no scan_file = {scan_file}')
            else:
                if ''.join(scan_file.suffixes) not in ['.nii.gz','.nii']:
                    raise ValueError(f'error: scan_file = {scan_file} is not a NIFTI file')
            self.scan_file = scan_file

            if store_mem:
                self.scan = nib.load(self.scan_file).get_fdata()
                self.scan = np.transpose(self.scan,[3,0,1,2])

        elif isinstance(scan_file,np.ndarray):
            self.scan = scan_file

    def run(self, scan_file : Union[str,Path,np.ndarray], **kwargs):

        self.load_scan(scan_file,**kwargs)
        super().run(data = self.scan,**kwargs)
