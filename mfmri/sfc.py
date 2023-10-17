import numpy as np

def padding(arr):
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

def hilbert2d_sfc(arr,return_dict = False):
    '''
    arr - 2d array to be hilbertized
    '''
    if len(arr.shape) != 2:
        print('error: incorrect arr dimensionality; len(arr.shape) != 2')

    if arr.shape[0] != arr.shape[1]:
        print('error: arr is rectangular; arr.shape[0] != arr.shape[1]')


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
