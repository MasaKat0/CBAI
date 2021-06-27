import numpy as np
from scipy.optimize import minimize
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


def semipara_optimal_weights(mu_context, zeta, delta=1e-11, initial_weight=None):
    # returns T*(mu) and w*(mu)
    K, D = mu_context.shape

    mu_context_x = mu_context*zeta.T
    mu = np.sum(mu_context_x, axis=1)

    idxmax = np.where(mu == np.max(mu))[0]
    L = len(idxmax)
    if (L > 1):
        # multiple optimal arms
        weight = np.zeros(shape=(K, D))
        weight[idxmax, :] = 1/L
        return 0, weight
    else:
        def semipara_kl_divergence_x(weight_x, alt_mu, x):
            KL_temp = 0

            for a in range(K):
                KL_temp += weight_x[a] * \
                    d(mu_context[a, x],
                        alt_mu[a])

            return (zeta[x]*KL_temp)[0]

        def semipara_kl_divergence(weight, alt_mu):
            KL = 0
            weight = weight.reshape((K, D))
            alt_mu = alt_mu.reshape((K, D))
            for x in range(D):
                KL += semipara_kl_divergence_x(weight[:, x],
                                               alt_mu[:, x], x)

            return KL

        def const_alt_mu_func(alt_mu, idxmax):
            alt_mu = alt_mu.reshape(K, D)
            alt_mu_x = alt_mu*zeta.T
            alt_mu = np.sum(alt_mu_x, axis=1)
            # print(alt_mu[idx] - alt_mu[idxmax])

            diff = []
            for idx in range(K):
                if idx != idxmax[0]:
                    diff.append((alt_mu[idx] - alt_mu[idxmax])[0])

            return np.max(diff)

        const_alt_mu = [
            {'type': 'ineq', 'fun': lambda alt_mu: const_alt_mu_func(alt_mu, idxmax)}]

        const_weight = []
        for x in range(D):
            const_weight.append(
                {'type': 'eq', 'fun': lambda weight: weight[x*K: (x+1)*K].sum() - 1})

        bnds = []
        for x in range(D):
            for a in range(K):
                bnds.append((0, 1))

        alt_mu = np.ones(shape=mu_context.shape)*0.5
        alt_mu = alt_mu.reshape(K*D)

        if initial_weight is not None:
            weight = initial_weight
        else:
            weight = np.ones(shape=(K, D)).T
            weight /= np.sum(weight, axis=0)
            weight = weight.T.reshape(K*D)

        def min_semipara_kl_divergence(weight, alt_mu):
            KL = np.zeros(K)
            for x in range(D):
                weight_x = weight[x*K: (x+1)*K]
                # weight_x[1] /= 100000

                for a in range(K):
                    KL_temp = weight_x[a] * \
                        d(mu_context[a, x],
                            alt_mu[x*D+a: x*D+a+1])

                    KL[a] += (zeta[x]*KL_temp)[0]
            return np.min(KL[idxmax] + KL[~idxmax[0]])

        class ObjectiveFunc():
            def __init__(self):
                self.alt_mu = mu_context.copy()
                self.alt_mu = alt_mu.reshape(K*D)

            def calc(self, weight):
                weight_mat = weight.reshape((K, D))
                # print(np.sum(weight_mat, axis=0))
                weight_mat /= np.sum(weight_mat, axis=0)
                weight = weight_mat.reshape(K*D)

                def inner_obj(alt_mu):
                    return semipara_kl_divergence(weight, alt_mu)

                #self.alt_mu = np.ones((K, D))/K
                # self.alt_mu = alt_mu.reshape(K*D)

                # print(tuple(const_alt_mu))
                # if alt_mu is None:

                # res_alt_mu = minimize(inner_obj, alt_mu, method='SLSQP', bounds=tuple(bnds),
                # constraints=tuple(const_alt_mu), options={'maxiter': 1000, 'ftol': 1e-100, 'eps': 1.4901161193847656e-10})

                res_alt_mu = minimize(inner_obj, self.alt_mu, method='SLSQP', bounds=tuple(bnds),
                                      constraints=tuple(const_alt_mu))

                self.alt_mu = res_alt_mu.x

                min_semipara_kl_divergence(weight, self.alt_mu)

                return - semipara_kl_divergence(weight, self.alt_mu)
                # return - min_semipara_kl_divergence(weight, self.alt_mu)

        bnds_x = []
        for a in range(K):
            for x in range(D):
                bnds_x.append((0.01, 0.99))

        outer_obj = ObjectiveFunc()

        res_weight = minimize(
            outer_obj.calc, weight, method='SLSQP', bounds=tuple(bnds_x), options={'disp': True})

        # res_weight = minimize(outer_obj, weight, method='BFGS', bounds=tuple(bnds_x))

        weight = res_weight.x
        alt_mu = outer_obj.alt_mu

        vOPT = semipara_kl_divergence(weight, alt_mu)

        return vOPT, weight

# OPTIMAL ALGORITHMS


def contextual_track_stop(mu_context, zeta, delta):
    # Uses a Tracking of the cummulated sum

    condition = True
    K = len(mu_context)
    D = len(zeta)

    def rate(t):
        return np.log((np.log(t) + 1)/delta)

    contexts = np.array([i for i in range(D)])
    num_sampling = np.zeros((K, D))
    sum_rewards = np.zeros((K, D))
    sum_context = np.zeros((D, 1))
    # initialization
    # for a in range(K):
    # N[a] = 1
    # S[a] = sample(mu[a])

    # t = K

    # SumWeights = np.ones((1, K))/K
    bnds = []
    for x in range(D):
        for a in range(K):
            bnds.append((0, 1))

    time = 0
    zeta_p = zeta.T[0]
    initialization_array = np.zeros(D, dtype=np.int64)
    initialize_weight = True
    count = 0
    est_time = 1
    est_weights = np.ones((K, D))/K
    est_weights_sum = est_weights.copy()
    score_list = []
    recom_arm_list = []
    while (condition):
        pulled_ratio = (num_sampling/sum_context[:, 0])
        print(pulled_ratio.sum(axis=0))
        pulled_ratio = pulled_ratio.reshape(K*D)

        observed_context = np.random.choice(contexts, p=zeta_p)
        sum_context[observed_context] += 1

        if (num_sampling > 0).all():
            mu_context_est = sum_rewards / num_sampling
            zeta_est = sum_context / time
            one_matrix = np.ones(sum_context.shape)
            mu_context_x_est = mu_context_est*zeta_est.T
            mu_est = mu_context_x_est.sum(axis=1)

            IndMax = np.where(mu_est == np.max(mu_est))[0]
            best_est = np.random.choice(IndMax)
            # Compute the stopping statistic

            if initialize_weight is True:
                try:
                    _, est_weights_temp = semipara_optimal_weights(
                        mu_context_est, zeta_est, initial_weight=est_weights)
                except:
                    est_weights_temp = np.ones((K, D))/K

                print(est_weights_temp.reshape((K, D)))
                initialize_weight = False
                est_weights_sum += est_weights_temp.reshape(K, D)
                est_time += 1
                est_weights = (est_weights_sum / est_time)
                est_weights = est_weights / est_weights.sum(axis=0)

            def semipara_kl_divergence_x(weight_x, alt_mu, x):
                KL_temp = 0

                for a in range(K):
                    KL_temp += weight_x[a] * \
                        d(mu_context_est[a, x],
                            alt_mu[a])

                return (one_matrix[x]*KL_temp)[0]

            def semipara_kl_divergence(weight, alt_mu):
                KL = 0
                weight = weight.reshape((K, D))
                alt_mu = alt_mu.reshape((K, D))
                for x in range(D):
                    KL += semipara_kl_divergence_x(weight[:, x],
                                                   alt_mu[:, x], x)
                return KL

            def inner_obj(alt_mu):
                return semipara_kl_divergence(num_sampling, alt_mu)

            def const_alt_mu_func(alt_mu, best_est):
                alt_mu = alt_mu.reshape(K, D)
                alt_mu_x = alt_mu*zeta_est.T
                alt_mu = np.sum(alt_mu_x, axis=1)
                # print(alt_mu[idx] - alt_mu[idxmax])

                diff = []
                for idx in range(K):
                    if idx != best_est:
                        diff.append((alt_mu[idx] - alt_mu[best_est]))

                return np.max(diff)

            init_alt_mu = np.ones((K, D))/K

            const_alt_mu = [
                {'type': 'ineq', 'fun': lambda alt_mu: const_alt_mu_func(alt_mu, best_est)}]

            res_alt_mu = minimize(inner_obj, init_alt_mu, method='SLSQP', bounds=tuple(bnds),
                                  constraints=tuple(const_alt_mu))

            score = res_alt_mu.fun

            # score =

            #score = np.max(score_list)
            print(time)
            print(np.sum(sum_context))
            print(score)
            print(rate(time))
            # print(np.argmax(score_list))
            print(np.mean(pulled_ratio - est_weights.reshape(K*D))**2)

            if (time >= 2000 - 1):
                # if (score > 1000000000):
                # stop
                condition = False
            elif (time > 10000000):
                # stop and output (0,0)
                condition = False
                best_est = -1
                num_sampling = np.zeros((K, D))
            else:
                # SumWeights = SumWeights+Dist
                # SumWeights = SumWeights/np.sum(SumWeights)
                # choice of the arm
                if (np.min(num_sampling[:, observed_context]) <= np.max(np.sqrt(sum_context[observed_context]) - K/2, 0)):
                    # forced exploration
                    A = np.argmin(num_sampling[:, observed_context])
                else:
                    A = np.argmax(
                        (sum_context[observed_context]*est_weights[:, observed_context]-num_sampling[:, observed_context]))
                # draw the arm
                num_sampling[A, observed_context] += 1
                sum_rewards[A,
                            observed_context] += sample(mu_context[A, observed_context])

        else:
            if initialization_array[observed_context] >= K:
                initialization_array[observed_context] -= K

            num_sampling[initialization_array[observed_context],
                         observed_context] += 1
            sum_rewards[initialization_array[observed_context],
                        observed_context] += sample(mu_context[initialization_array[observed_context], observed_context])
            initialization_array[observed_context] += 1
            score = 0
            best_est = -1

            # Empirical best arm
        if initialize_weight is False:
            count += 1
            if count == 10:
                initialize_weight = True
                count = 0

        time += 1

        score_list.append(score)
        recom_arm_list.append(best_est)

    recommendation = best_est
    return (recommendation, num_sampling.sum(axis=1), score_list, recom_arm_list)
