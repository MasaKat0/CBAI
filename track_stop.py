import numpy as np
from klfunctions import *

typeDistribution = "Bernoulli"


# if (typeDistribution == "Bernoulli"):
d = dBernoulli
dup = dupBernoulli
dlow = dlowBernoulli


def sample(mu):
    if np.random.uniform() < mu:
        return 1
    else:
        return 0


def bdot(theta):
    return exp(theta)/(1+exp(theta))


def bdotinv(mu):
    return np.log(mu/(1-mu))

# COMPUTING THE OPTIMAL WEIGHTS


def dicoSolve(f, xMin, xMax, delta=1e-11):
    # find m such that f(m)=0 using dichotomix search
    l = xMin
    u = xMax
    sgn = f(xMin)
    while u-l > delta:
        m = (u+l)/2
        if f(m)*sgn > 0:
            l = m
        else:
            u = m

    m = (u+l)/2
    return m


def I(alpha, mu1, mu2):
    if (alpha == 0) | (alpha == 1):
        return 0
    else:
        mid = alpha*mu1 + (1-alpha)*mu2

    return alpha*d(mu1, mid)+(1-alpha)*d(mu2, mid)


def muddle(mu1, mu2, nu1, nu2): return (nu1*mu1 + nu2*mu2)/(nu1+nu2)


def cost(mu1, mu2, nu1, nu2):
    if (nu1 == 0) & (nu2 == 0):
        return 0
    else:
        alpha = nu1/(nu1+nu2)
        return (nu1 + nu2)*I(alpha, mu1, mu2)


def xkofy(y, k, mu, delta=1e-11):
    # return x_k(y), i.e. finds x such that g_k(x)=y
    def g(x): return (1+x)*cost(mu[0], mu[k], 1/(1+x), x/(1+x))-y
    xMax = 1
    while g(xMax) < 0:
        xMax = 2*xMax

    return dicoSolve(g, 0, xMax, 1e-11)


def aux(y, mu):
    # returns F_mu(y) - 1
    K = len(mu)
    x = [xkofy(y, k, mu) for k in range(1, K)]
    m = [muddle(mu[0], mu[k], 1, x[k-1]) for k in range(1, K)]

    return sum([zero_dev(d(mu[0], m[k-1]), (d(mu[k], m[k-1]))) for k in range(1, K)]) - 1


def zero_dev(a, b):
    if b == 0:
        return 0
    else:
        return a / b


def oneStepOpt(mu, delta=1e-11):
    yMax = 0.5
    if d(mu[0], mu[1]) == np.inf:
        # find yMax such that aux(yMax,mu)>0
        while aux(yMax, mu) < 0:
            yMax = yMax*2
    else:
        yMax = d(mu[0], mu[1])

    def aux_m(y):
        return aux(y, mu)

    y = dicoSolve(aux_m, 0, yMax, delta)
    x = [xkofy(y, k, mu, delta) for k in range(1, len(mu))]
    x = np.insert(x, 0, 1)
    nuOpt = np.copy(x/np.sum(x))

    # print(x/np.sum(x))
    return nuOpt[0]*y, nuOpt


def OptimalWeights(mu, delta=1e-11):
    # returns T*(mu) and w*(mu)
    K = len(mu)
    IndMax = np.where(mu == np.max(mu))
    L = len(IndMax)
    if (L > 1):
        # multiple optimal arms
        vOpt = np.zeros(K)
        vOpt[IndMax] = 1/L
        return 0, vOpt
    else:
        index = np.argsort(-mu)
        mu = mu[index]
        unsorted = np.array([i for i in range(K)])
        invindex = np.zeros(K, np.int)
        invindex[index] = unsorted
        # one-step optim
        vOpt, NuOpt = oneStepOpt(mu, delta)
        # back to good ordering
        nuOpt = NuOpt[invindex]
        return vOpt, nuOpt

# OPTIMAL ALGORITHMS


def track_stop(mu, delta):
    # Uses a Tracking of the cummulated sum
    def rate(t):
        return np.log(2*t*(len(mu) - 1)/delta)
        # return np.log((np.log(t)+1)/delta)

    condition = True
    K = len(mu)
    N = np.zeros(K)
    S = np.zeros(K)
    score_list = []
    recom_arm_list = []
    # initialization
    for a in range(K):
        N[a] = 1
        S[a] = sample(mu[a])
        score_list.append(0)
        recom_arm_list.append(-1)

    t = K

    SumWeights = np.ones((1, K))/K
    while (condition):
        mu_est = S / N
        # Empirical best arm
        IndMax = np.where(mu_est == np.max(mu_est))[0]
        Best = np.random.choice(IndMax)
        # Compute the stopping statistic
        NB = N[Best]
        SB = S[Best]
        muB = SB/NB
        mu_mid = (NB/(NB+N))*muB + (N/(NB+N))*mu_est
        index = np.array([i for i in range(K)])
        index_without_best = np.delete(index, Best)
        Score = np.min([NB*d(muB, mu_mid[a])+N[a]*d(mu_est[a], mu_mid[a])
                        for a in index_without_best])

        print(t)
        print(Score)
        print(rate(t))
        if (t >= 2000 - 1):
            # stop
            condition = False
        elif (t > 1000000):
            # stop and output (0,0)
            condition = False
            Best = -1
            N = np.zeros((1, K))
        else:
            # continue and sample an arm
            _, est_weights = OptimalWeights(mu_est, 1e-11)
            #SumWeights = SumWeights+Dist
            #SumWeights = SumWeights/np.sum(SumWeights)
            # choice of the arm
            if (np.min(N) <= np.max(np.sqrt(t) - K/2, 0)):
                # forced exploration
                A = np.argmin(N)
            else:
                A = np.argmax((t*est_weights-N))
            # draw the arm
            t += 1
            S[A] += sample(mu[A])
            N[A] += 1

        score_list.append(Score)
        recom_arm_list.append(Best)

    recommendation = Best
    return (recommendation, N, score_list, recom_arm_list)
