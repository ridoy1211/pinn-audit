"""Specified localized initial field and quadrature-based heat solution."""

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_legendre

CENTER = 0.35
WIDTH = 0.04


def unnormalized(x):
    x = np.asarray(x, dtype=float)
    return x * (1-x) * np.exp(-((x-CENTER)/WIDTH)**2/2)


PEAK_LOCATION = brentq(
    lambda x: 1/x - 1/(1-x) - (x-CENTER)/WIDTH**2,
    1e-12, 1-1e-12, xtol=1e-14,
)
NORMALIZATION = float(unnormalized(PEAK_LOCATION))


def initial(x):
    return unnormalized(x) / NORMALIZATION


@lru_cache(maxsize=None)
def coefficients(modes=256, quadrature_points=1024):
    z, weights = roots_legendre(quadrature_points)
    x = (z+1)/2
    # a_j = 2 integral_0^1 f(x) sin(j*pi*x) dx.
    k = np.arange(1, modes+1)*np.pi
    a = np.sin(k[:,None]*x) @ (weights*initial(x))
    a.setflags(write=False)
    return a


def paired(x, t, alpha=0.05, modes=256, quadrature_points=1024):
    x, t = np.asarray(x,dtype=float), np.asarray(t,dtype=float)
    if x.ndim != 1 or x.shape != t.shape or np.any(t<0):
        raise ValueError('Expected paired vectors and nonnegative times.')
    a = coefficients(modes,quadrature_points)
    k = np.arange(1,modes+1)*np.pi
    result = np.empty_like(x)
    for start in range(0,len(x),2048):
        stop=start+2048
        result[start:stop] = np.sum(
            np.sin(x[start:stop,None]*k)*np.exp(-alpha*t[start:stop,None]*k**2)*a,
            axis=1,
        )
    # Initial-field scoring uses the defining function, not its truncation.
    zero = t==0
    result[zero] = initial(x[zero])
    return result


def grid(x,t,alpha=0.05,modes=256,quadrature_points=1024):
    x,t=np.asarray(x),np.asarray(t)
    a=coefficients(modes,quadrature_points)
    k=np.arange(1,modes+1)*np.pi
    field=(np.exp(-alpha*t[:,None]*k**2)*a) @ np.sin(k[:,None]*x)
    field[t==0]=initial(x)
    return field
