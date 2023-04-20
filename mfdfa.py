import numpy as np
from MFDFA import MFDFA as mfdfa_py

def aut_multpolyfit_MFDFA(x,y,n):
    '''
    refactored aut_multpolyfit_MFDFA.m
    '''
    m = x.shape[1]
    c = np.zeros((n+1,m))
    r = np.zeros_like(y)
    for k in range(m):
        M = np.stack([x[:,k]**i for i in range(n+1)],axis=0).T
        c[:,k] = np.linalg.pinv(M) @ y[:,k]
        r[:,k] = M @ c[:,k] - y[:,k]
    sserr = sum(r**2)
    sstot = sum((y-y.mean(axis=0))**2)
    
    R2 = 1 - sserr/sstot
    return c,R2

def aut_MFDFA(seria,s,q,m):
    '''
    refactored aut_MFDFA.m
    s - scales
    q - multifractal q exponents
    m - polynomial order
    '''
    meanx = np.mean(seria)
    
    N = len(seria)
    seria1 = seria - meanx
    Y = np.cumsum(seria1)
    liczba_q = len(q)
    
    F = np.zeros((len(s),np.floor(N/s[0]).astype(int)*2))
    Fq = np.zeros((len(s),liczba_q))
    powerF = np.zeros((len(s),np.floor(N/s[0]).astype(int)*2))
    
    for i in range(len(s)):
        Ns = np.floor(N/s[i]).astype(int)
        
        wartosci_start = np.zeros((Ns,s[i]))
        wartosci_end = np.zeros((Ns,s[i]))
        y_start = np.zeros((Ns,s[i]))
        y_end = np.zeros((Ns,s[i]))

        xx = np.arange(s[i])+1
        x = np.tile(xx,(Ns,1)).T
        
        R = N % s[i]
        y_start = Y[:len(Y)-R].reshape(Ns,s[i])
        # y_end = Y[R:][::-1].reshape(Ns,s[i])
        y_end = Y[R:].reshape(Ns,s[i]) # IS THIS RIGHT?

        c_start,_ = aut_multpolyfit_MFDFA(x,y_start.T,m)
        c_start = c_start[::-1]
        
        c_end,_ = aut_multpolyfit_MFDFA(x,y_end.T,m)        
        c_end = c_end[::-1]
        
        for j in range(Ns):
            wartosci_start[j,:] = np.poly1d(c_start[:,j])(xx)
            wartosci_end[j,:] = np.poly1d(c_end[:,j])(xx)
        
        # j = slice(0,Ns)
        j = np.arange(Ns)
        F[i,:len(j)] = ((y_start[j,:] - wartosci_start[j,:])**2).T.sum(axis=0)/s[i]
        F[i,Ns:2*len(j)] = ((y_end[j,:] - wartosci_end[j,:])**2).T.sum(axis=0)/s[i]
        
    for i in range(liczba_q):
        for j in range(len(s)):
            powerF = F[j,:np.floor(N/s[j]).astype(int)*2]**(q[i]/2)
            Fq[j,i] = (powerF.sum(axis=0)/(2*np.floor(N/s[j])))**(1/q[i])
    
    return Fq

def mfdfa_matlab(timeseries, lag, q, order):
    return aut_MFDFA(seria = timeseries,s = lag ,q = q,m = order)


def aut_artefact8(series: np.ndarray, window_size : int = 10):
    '''
    clean series by removing parts where the signal is flat
    series - np.ndarray
    window_size - flat signal length
    '''
    
    n = window_size
    
    imax = np.floor(len(series)/n).astype(int)
    swv = np.lib.stride_tricks.sliding_window_view
    out = swv(series,window_shape=(n,))

    subout = out[::n]
    subout2 = subout[list(map(lambda x: len(set(x))>1,subout))]
    series_clean = subout2.reshape(-1)
    ix_post = (len(series) % n)

    if ix_post > 0:
        series_clean = np.append(series_clean,series[-ix_post:])

    return series_clean

def ghurst(Fq : np.ndarray,
           scales : np.ndarray,
           min_scale_ix : int = None,
           max_scale_ix : int = None,
           eps : float = 1e-10,
           exponent_cutoff : int = 10):
    '''
    calc generalized Hurst exponent
    Fq.shape = (N_SAMPLES,N_EXP)
    s.shape = (N_SAMPLES)
    eps is a regularizing parameter
    exponent_cutoff - casts large Hurst exponents as nans; typically an artefact of using Fq with q<0
    '''
    
    assert len(Fq.shape)==2, 'cast to shape (N_SAMPLES,1) for one dimensional Fq'
    assert scales.shape[0] == Fq.shape[0], 'incompatible shapes of Fq and s'
    
    scales = scales.astype(float)
    Fq = Fq.astype(float)

    sl = slice(min_scale_ix,max_scale_ix)
    scales = scales[sl]
    Fq = Fq[sl]
    
    params, res, _, _, _ = np.polyfit(x=np.log(scales)+eps,
                                      y=np.log(Fq+eps),
                                      deg=1,
                                      full=True)
    
    params[0,params[0]>exponent_cutoff] = np.nan
    
    return params[0],res