import numpy as np
from MFDFA import MFDFA as mfdfa_py

mfdfa_matlab = None

def ghurst(Fq,s,nmin=0,nmax=0, eps=1e-100, exponent_cutoff = 10):
    '''
    calc generalized Hurst exponent
    Fq.shape = (N_SAMPLES,N_EXP)
    s.shape = (N_SAMPLES)
    eps is a regularizing parameter
    exponent_cutoff - casts large Hurst exponents as nans; typically an artefact of using Fq with q<0
    '''

    assert len(Fq.shape)==2, 'cast to shape (N_SAMPLES,1) for one dimensional Fq'
    assert s.shape[0] == Fq.shape[0], 'incompatible shapes of Fq and s'
    
    if nmax != 0:
        s = s[nmin:nmax]
        Fq = Fq[nmin:nmax]
        
    params, res, _, _, _ = np.polyfit(x=np.log(s),
                                      y=np.log(Fq+eps),
                                      deg=1,
                                      full=True)
    
    params[0,params[0]>exponent_cutoff] = np.nan
    
    return params[0],res