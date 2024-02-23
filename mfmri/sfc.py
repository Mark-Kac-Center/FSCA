import numpy as np
from typing import Union
import os

def padding(arr: np.ndarray) -> np.ndarray:
    '''
    arr is embedded inside a cube of size 2**n (any dimensionality works)
    '''
    maxsize = max(arr.shape)
    nextpow2 = int(2**np.ceil(np.log2(maxsize)))

    befores = []
    afters = []
    for s in arr.shape:
        if s % 2 == 0:
#             print('symmetric pad')
            befores += [int((nextpow2 - s)/2)]
            afters += [int((nextpow2 - s)/2)]
        else:
#             print('non-symmetric pad')
            before = int(nextpow2/2 - np.floor(s/2))
            befores += [before]
            afters += [nextpow2 - s - before]

    pad_width = np.array([befores,afters]).T
    arrpad = np.pad(arr, pad_width = pad_width, constant_values = 0)
#     print(pad_width)
    return arrpad

def _check_arr_dims(arr: np.ndarray) -> None:
    if len(arr.shape) != 2:
        print('error: incorrect arr dimensionality; len(arr.shape) != 2')

    if arr.shape[0] != arr.shape[1]:
        print('error: arr is not square; arr.shape[0] != arr.shape[1]')

def hilbert2d_sfc(arr: np.ndarray,
                  return_dict: bool = False) -> Union[dict,np.ndarray]:
    '''
    arr - 2d array to be hilbertized
    '''
    
    _check_arr_dims(arr)

    rowNumel = arr.shape[0]
    order = int(np.log2(rowNumel))

    if 2**order != rowNumel:
        print('error: arr dimensions != 2**n')

    a = 1+1j
    b = 1-1j
    z = 0

    for k in range(order):
        w = 1j*np.conj(z)
        z = np.array([w-a,z-b,z+a,b-w])/2
        z = z.reshape(-1)

    newCol = np.real(z)
    newRow = np.imag(z)

    newCol = rowNumel*newCol/2 + rowNumel/2 + 0.5
    newRow = rowNumel*newRow/2 + rowNumel/2 + 0.5

    hilbertInd = ((newCol-1)*rowNumel+newRow-1).astype(int)

    ixs = np.indices(arr.shape)
    ixs = np.transpose(ixs,(1,2,0))
    ixs = ixs.reshape(-1,2)

    ixsH = ixs[hilbertInd]
    arrH = arr.reshape(-1)[hilbertInd]

    #LT, VO
    if return_dict:
        return {'LT':arrH,'VO':ixsH}
    else:
        return arrH

def ddsfc2d(arr: np.ndarray, 
            matlab_engine = None, 
            return_dict: bool = False) -> Union[dict,np.ndarray]:
    '''
    Data-Driven SFC 
    wrapper of code https://github.com/zhou-l/DataDrivenSpaceFillCurve
    
    EXPERIMENTAL - can cause problems if changedir operations fail
    '''
    
    import io
    from pathlib import Path
    import os
    from scipy.io import savemat
    
    _check_arr_dims(arr)
    
    sfc_module_dir = os.path.dirname(os.path.realpath(__file__))
    
    TEMP_FILE = 'tempfile.mat'

    if not matlab_engine:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        
    else:
        eng = matlab_engine

    # ensure we are in the right directory
    eng.cd(f'{sfc_module_dir}/ddsfc_matlab')
        
    curr_path = Path(eng.cd())
    
    temp_file_path = curr_path / TEMP_FILE
    
    savemat(temp_file_path,{'V':arr})
    clLT, clVisitOrder, fullLT = eng.SFCQuadTreeMultiScaleMain(TEMP_FILE,
                                                               nargout = 3, 
                                                               stdout = io.StringIO()
                                                              )
    os.remove(temp_file_path)
    
    if not matlab_engine:
        eng.quit()
    
    clVisitOrder = np.asarray(clVisitOrder)
    clLT = np.asarray(clLT)
    clLT = clLT[:,0]

    if return_dict:
        return {'LT': clLT,'VO': clVisitOrder}
    else:
        return clLT
