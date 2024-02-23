import numpy as np

DEFAULT_BRAIN_TH = 0.99
DEFAULT_SFC_TYPE = 'hilbert'
DEFAULT_SLICE_AXIS = 'z'
DEFAULT_NORM_LEVEL = 'slice'
DEFAULT_WINDOW_SIZE = 10
DEFAULT_SCALES = (10,None,20)
DEFAULT_QORDERS = np.concatenate((np.arange(-4,0,.2),np.arange(0.2,4.2,.2)))
DEFAULT_MFDFA_LIB = 'py'
DEFAULT_FIT_ORDER = 2
DEFAULT_SCALE_PRESET = 'small_scales'
