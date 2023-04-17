import numpy as np

def vectorize(f):
    def fv(X,*args,**kwargs):
        return np.array([f(x,*args,**kwargs) for x in X])
    return fv