import pytest
import numpy as np
import re

from mfmri.mfmri_core import BaseMFractalMRI

@pytest.fixture
def BaseMFractalMRI_instance():
    return BaseMFractalMRI()

def test__load_scan_success(BaseMFractalMRI_instance):
    scan_file = 'test-data/scanfile.nii.gz'
    BaseMFractalMRI_instance.load_scan(scan_file)
    
    assert isinstance(BaseMFractalMRI_instance.scan,np.ndarray)
    
    scan_file = np.random.randn(10,10,10)
    BaseMFractalMRI_instance.load_scan(scan_file)
    assert isinstance(BaseMFractalMRI_instance.scan,np.ndarray)

def test__load_scan_file_missing(BaseMFractalMRI_instance):
    scan_file = 'test-data/scanfile_missing.nii.gz'
    with pytest.raises(FileNotFoundError, match=f'error: no scan_file = {scan_file}'):
        BaseMFractalMRI_instance.load_scan(scan_file)
    
@pytest.fixture
def scan_file_npz(tmp_path_factory):
    scan_file_npz = tmp_path_factory.mktemp('test-data') / 'scanfile.npz'
    np.savez(scan_file_npz,[])
    return scan_file_npz
    
def test__load_scan_file_wrong_filetype(BaseMFractalMRI_instance,scan_file_npz):
    # scan_file = 'test-data/scanfile.mat'
    with pytest.raises(ValueError, match=f'error: scan_file = {scan_file_npz} is not a NIFTI file'):
        BaseMFractalMRI_instance.load_scan(scan_file_npz)
    
def test__load_scan_wrong_shape_array(BaseMFractalMRI_instance):
    scan_file = np.random.randn(10,10)
    match = re.escape(f'error: provided array is of incorrect shape = {scan_file.shape}')
    with pytest.raises(ValueError, match=match):
        BaseMFractalMRI_instance.load_scan(scan_file)
    
def test__slice_scan_basic(BaseMFractalMRI_instance):
    scan_file = 'test-data/scanfile.nii.gz'
    BaseMFractalMRI_instance.load_scan(scan_file)
    scan_shape = BaseMFractalMRI_instance.scan.shape
    slice_axes = ['x','y','z']
    for i,slice_axis in enumerate(slice_axes):
        BaseMFractalMRI_instance.slice_scan(slice_axis=slice_axis)
        assert scan_shape[i] == BaseMFractalMRI_instance.slices.shape[0]
        
    for slice_axis in ['none','None',None]:
        BaseMFractalMRI_instance.slice_scan(slice_axis=slice_axis)
        assert BaseMFractalMRI_instance.slices.shape[0] == 1
        assert BaseMFractalMRI_instance.slices.shape[1:] == BaseMFractalMRI_instance.scan.shape

def test__slice_to_sfc_basic(BaseMFractalMRI_instance):
    BaseMFractalMRI_instance.scan = np.random.randn(10,10,10)
    BaseMFractalMRI_instance.slice_scan(slice_axis='z')
    BaseMFractalMRI_instance.slice_to_sfc()
    assert BaseMFractalMRI_instance.sfcs.shape == (10,256)
    
    BaseMFractalMRI_instance.scan = np.random.randn(10,10,10)
    BaseMFractalMRI_instance.slice_scan(slice_axis='none')
    BaseMFractalMRI_instance.slice_to_sfc(sfc_type = 'hilbert3d')
    assert BaseMFractalMRI_instance.sfcs.shape == (1,16**3)
    
def test_slice_to_sfc_apply_incompatible_sfc_method(BaseMFractalMRI_instance):
    BaseMFractalMRI_instance.scan = np.random.randn(10,10,10)
    
    # 2d slice -> 3d sfc
    BaseMFractalMRI_instance.slice_scan(slice_axis='z')
    with pytest.raises(ValueError, match='error: hilbert3d needs a 3d slice'):
        BaseMFractalMRI_instance.slice_to_sfc(sfc_type = 'hilbert3d')

    # 3d slice -> 2d sfc
    BaseMFractalMRI_instance.scan = np.random.randn(10,10,10)
    BaseMFractalMRI_instance.slice_scan(slice_axis='none')
    with pytest.raises(ValueError, match="error: hilbert needs a 2d slice"):
        BaseMFractalMRI_instance.slice_to_sfc()
        
def test_slice_to_sfc_unknown_sfc_type(BaseMFractalMRI_instance):
    BaseMFractalMRI_instance.scan = np.random.randn(10,10,10)
    BaseMFractalMRI_instance.slice_scan(slice_axis='z')
    with pytest.raises(ValueError, match="error: wrong sfc_type = grogoth123"):
        BaseMFractalMRI_instance.slice_to_sfc(sfc_type='grogoth123')
        
def test__calc_mfdfa_basic(BaseMFractalMRI_instance):
    BaseMFractalMRI_instance.sfcs = np.random.randn(10,256)
    BaseMFractalMRI_instance.calc_mfdfa()
    
    assert BaseMFractalMRI_instance.scales.shape == (10,31)
    assert BaseMFractalMRI_instance.fqs.shape == (10,31,40)

def test__calc_mfdfa_methods(BaseMFractalMRI_instance):
    np.random.seed(124)
    BaseMFractalMRI_instance.sfcs = np.random.randn(10,256)
    BaseMFractalMRI_instance.calc_mfdfa(mfdfa_lib = 'py')
    fqs1 = BaseMFractalMRI_instance.fqs
    BaseMFractalMRI_instance.calc_mfdfa(mfdfa_lib = 'matlab')
    fqs2 = BaseMFractalMRI_instance.fqs
    
    assert (fqs1-fqs2).sum() < 1e-10
    
def test__calc_ghurst_basic(BaseMFractalMRI_instance):
    BaseMFractalMRI_instance.sfcs = np.random.randn(10,256)
    BaseMFractalMRI_instance.calc_mfdfa()
    BaseMFractalMRI_instance.calc_ghurst()
    
    assert BaseMFractalMRI_instance.ghs.shape == (10,40)

def test__calc_ghurst_scales(BaseMFractalMRI_instance):
    np.random.seed(123)
    BaseMFractalMRI_instance.sfcs = np.random.randn(100,1024)
    BaseMFractalMRI_instance.calc_mfdfa()

    x = BaseMFractalMRI_instance.scales[0].astype(float)
    y = BaseMFractalMRI_instance.fqs[0,:,29].astype(float)

    sl=slice(None,5)
    scales = x[sl]
    Fq = y[sl]

    eps = 1e-10
    params, res, _, _, _ = np.polyfit(x=np.log(scales)+eps,
                                      y=np.log(Fq+eps),
                                      deg=1,
                                      full=True)
    
    BaseMFractalMRI_instance.calc_ghurst(scale_preset='small_scales')
    

    assert BaseMFractalMRI_instance.ghs[0,29] == pytest.approx(params[0])
    assert BaseMFractalMRI_instance.ghs_res[0,29] == pytest.approx(res)

    sl=slice(-10,None)
    scales = x[sl]
    Fq = y[sl]

    eps = 1e-10
    params, res, _, _, _ = np.polyfit(x=np.log(scales)+eps,
                                      y=np.log(Fq+eps),
                                      deg=1,
                                      full=True)
    
    BaseMFractalMRI_instance.calc_ghurst(scale_preset='large_scales')
    

    assert BaseMFractalMRI_instance.ghs[0,29] == pytest.approx(params[0])
    assert BaseMFractalMRI_instance.ghs_res[0,29] == pytest.approx(res)


