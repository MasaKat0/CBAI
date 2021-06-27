import numpy as np

# Bernoulli distributions

eps = 2.220446049250313e-16


def dBernoulli(p, q):

    res = 0
    if (p != q):
        if (p <= 0):
            p = eps
        if (p >= 1):
            p = 1-eps
        res = (p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q)))

    return res


def dupBernoulli(p, level):
    # KL upper confidence bound:
    # return qM>p such that d(p,qM)=level
    lM = p
    uM = min(min(1, p+np.sqrt(level/2)), 1)
    for j in range(16):
        qM = (uM+lM)/2
        if dBernoulli(p, qM) > level:
            uM = qM
        else:
            lM = qM
    return uM


def dlowBernoulli(p, level):
    # KL lower confidence bound:
    # return lM<p such that d(p,lM)=level
    lM = max(min(1, p-np.sqrt(level/2)), 0)
    uM = p
    for j in range(16):
        qM = (uM+lM)/2
        if dBernoulli(p, qM) > level:
            lM = qM
        else:
            uM = qM
    return lM


# Poisson distributions

def dPoisson(p, q):
    if (p == 0):
        res = q
    else:
        res = q-p + p*np.log(p/q)
    return res


def dupPoisson(p, level):
    # KL upper confidence bound: generic way
    # return qM>p such that d(p,qM)=level
    lM = p
    # finding an upper bound
    uM = max(2*p, 1)
    while (dPoisson(p, uM) < level):
        uM = 2*uM
    for j in range(16):
        qM = (uM+lM)/2
        if dPoisson(p, qM) > level:
            uM = qM
        else:
            lM = qM

    return uM


def dlowPoisson(p, level):
    # KL lower confidence bound: generic way
    # return lM<p such that d(p,lM)=level
    # finding a lower bound
    lM = p/2
    if p != 0:
        while (dPoisson(p, lM) < level):
            lM = lM/2
    uM = p
    for j in range(16):
        qM = (uM+lM)/2
        if dPoisson(p, qM) > level:
            lM = qM
        else:
            uM = qM
    return lM

# Exponential distribution


def dExpo(p, q):
    res = 0
    if (p != q):
        if (p <= 0) | (q <= 0):
            res = np.inf
        else:
            res = p/q - 1 - np.log(p/q)

    return res


def dupExpo(p, level):
    # KL upper confidence bound: generic way
    # return qM>p such that d(p,qM)=level
    lM = p
    # finding an upper bound
    uM = max(2*p, 1)
    while (dExpo(p, uM) < level):
        uM = 2*uM
    for j in range(16):
        qM = (uM+lM)/2
        if dExpo(p, qM) > level:
            uM = qM
        else:
            lM = qM
    return uM


def dlowExpo(p, level):
    # KL lower confidence bound: generic way
    # return lM<p such that d(p,lM)=level
    # finding a lower bound
    lM = p/2
    if p != 0:
        while (dExpo(p, lM) < level):
            lM = lM/2
    uM = p
    for j in range(16):
        qM = (uM+lM)/2
        if dExpo(p, qM) > level:
            lM = qM
        else:
            uM = qM

    return lM
