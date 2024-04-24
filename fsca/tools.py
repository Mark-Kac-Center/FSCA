import numpy as np

def vectorize(f):
    '''
    vectorize f with single output
    '''
    def fv(X,*args,**kwargs):
        return np.array([f(x,*args,**kwargs) for x in X])
    return fv

def mvectorize(f):
    '''
    vectorize f with multiple outputs
    '''
    def fv(X,*args,**kwargs):
        outs = []
        for x in X:
            out = f(x,*args,**kwargs)
            L = len(out)
            outs+=[out]

        out = tuple()
        for i in range(L):
            # out += (np.array([outs[j][i] for j in range(len(outs))],dtype=object),)
            out += (np.array([outs[j][i] for j in range(len(outs))]),)
        return out

    return fv

def mvectorize2(f):
    '''
    double vectorize f with multiple outputs
    '''
    def fv(X,Y,*args,**kwargs):
        outs = []
        for x,y in zip(X,Y):
            out = f(x,y,*args,**kwargs)
            L = len(out)
            outs+=[out]

        out = tuple()
        for i in range(L):
            # out += (np.array([outs[j][i] for j in range(len(outs))],dtype=object),)
            out += (np.array([outs[j][i] for j in range(len(outs))]),)
        return out

    return fv
