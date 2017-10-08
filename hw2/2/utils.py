import numpy as np

def gamma(z):
    if z <= 2:
        return 1
    else:
        return (z-1)*gamma(z-1)

def n_choose_k(n, k):
    return gamma(n+1) // gamma(k+1) // gamma(n-k+1)

def bataDistribution_singleValue(a, b, miu):
    return gamma(a+b) // gamma(a) // gamma(b) * (miu**(a-1)) * ((1-miu)**(b-1))

def bataDistribution_curve(a, b):
    source = np.linspace(0.001, 0.999, num=1000)
    target = [bataDistribution_singleValue(a, b, _) for _ in source]
    return source, target

def bataDistribution_maxProb(a, b):
    source, target = bataDistribution_curve(a, b)
    return source[np.argmax(np.asarray(target))]

def binomialDistribution_singleValue(N, m, miu):
    return n_choose_k(N, m) * (miu**(m)) * ((1-miu)**(N-m))    

def binomialDistribution_curve(N, m):
    source = np.linspace(0.001, 0.999, num=1000)
    target = [binomialDistribution_singleValue(N, m, _) for _ in source]
    return source, target

def binomialDistribution_maxProb(N, m):
    source, target = binomialDistribution_curve(N, m)
    return source[np.argmax(np.asarray(target))]