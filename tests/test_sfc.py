import pytest
import numpy as np

from mfmri.sfc import ddsfc2d, hilbert3d_sfc

@pytest.fixture
def matlab_engine_instance():
    import matlab.engine
    eng = matlab.engine.start_matlab()

    yield eng
    eng.quit()

def test__ddsfc2d_no_matlab_engine():
    arr = np.random.randn(8,8)
    
    arr_sfc = ddsfc2d(arr)
    
    assert isinstance(arr_sfc, np.ndarray)
    assert arr_sfc.shape == (8*8,)
    
def test__ddsfc2d_use_existing_matlab_engine(matlab_engine_instance):
    arr = np.random.randn(8,8)
    
    arr_sfc = ddsfc2d(arr, matlab_engine = matlab_engine_instance)
    
    assert isinstance(arr_sfc, np.ndarray)
    assert arr_sfc.shape == (8*8,)
    
def test__ddsfc2d_return_dictionary():
    
    arr = np.random.randn(8,8)
    arr_dict = ddsfc2d(arr, return_dict = True)
    
    assert isinstance(arr_dict, dict)
    assert arr_dict['LT'].shape == (8*8,)
    assert arr_dict['VO'].shape == (8*8,2)
    
def test__hilbert3d_basic():
    arr = np.random.randn(16,16,16)
    arr_sfc = hilbert3d_sfc(arr)
    
    assert isinstance(arr_sfc,np.ndarray)
    assert arr_sfc.shape == (16**3,)
    
