# Run Experiments, display results (and possibly save data) on a Bandit Problem to be specified

import numpy as np
import time
from klfunctions import *
from track_stop import *
from semipara_track_stop import *

np.random.seed(0)

# DO YOU WANT TO SAVE RESULTS?
typeExp = "Save"
typeExp = "NoSave"

# TYPE OF DISTRIBUTION


typeDistribution = "Bernoulli"
# CHANGE NAME (save mode)
fname = "results/Experiment4arms"

# BANDIT PROBLEM


zeta = np.array([[0.5, 0.5]]).T

mu = np.array([[0.9, 0.2]]).T
mu_temp = np.array([[0.5, 0.2]]).T

K = len(mu)
D = len(zeta)

mu_temp2 = 2*mu - mu_temp
mu_context = np.array([mu_temp.T[0], mu_temp2.T[0]]).T
mu_context_x = mu_context*zeta.T
mu_context *= mu/np.sum(mu_context_x, axis=1).reshape(K, 1)

best = np.argmax(mu)

#print(np.sum(mu_context_x, axis=0))
# RISK LEVEL
delta = 0.001

# NUMBER OF SIMULATIONS
num_similator = 20


# OPTIMAL SOLUTION


v, optWeights = OptimalWeights(mu.T[0])

gammaOpt = optWeights[best]
print("mu=", mu)
print("Theoretical number of samples:", (1/v)*np.log(1/delta))
print("Optimal weights:", optWeights)

v0, semiparaoptWeights = semipara_optimal_weights(mu_context, zeta)

gammaOpt = optWeights[best]
print("mu=", mu_context)
print("Theoretical number of samples:", (1/v0)*np.log(1/delta))
print("Optimal weights:", optWeights)
print("Optimal weights:", semiparaoptWeights)

# POLICIES


# policies = [TrackAndStop2, ChernoffBC2,ChernoffPTSHalf, ChernoffPTSOpt, KLLUCB, UGapE]

# names = ["TrackAndStop", "ChernoffBC", "ChernoffPTS","ChernoffPTSOpt", "KLLUCB", "UGapE"]

policies = [track_stop]

names = ["TrackAndStop"]

# EXPLORATION RATES


lP = len(policies)

# RUN EXPERIMENTS


def SaveData(mu, delta, num_similator):
    K = len(mu)

    for imeth in range(lP):
        policy = policies[imeth]
        # namePol = names[imeth]
        # startTime = time()

        a = []
        recommends_save = np.zeros((2000, num_similator))
        scores_save = np.zeros((2000, num_similator))
        for j in range(num_similator):
            recommendation, num_sampling, score_list, recom_arm_list = policy(
                mu, delta)
            a.append((recommendation, num_sampling))
            recommends_save[:, j] = recom_arm_list
            scores_save[:, j] = score_list
        res = [a[j] for j in range(num_similator)]

        draws = np.zeros((num_similator, K))
        proportion = np.zeros((num_similator, K))

        for k in range(num_similator):
            r = res[k][1]
            draws[k, :] = r
            proportion[k, :] = r/sum(r)

        recommend = [res[j][0] for j in range(num_similator)]
        Error = np.array([0 if (r == best) else 1 for r in recommend])

        # fraction of not terminate
        FracNT = np.sum([r == -1 for r in recommend])/num_similator

        FracReco = np.zeros(K)
        for k in range(K):
            FracReco[k] = sum(
                [1 if (r == k) else 0 for r in recommend])/(num_similator*(1-FracNT))

        print("Results for %s, average on %d runs\n" % (policy, num_similator))
        np.savetxt("results/draws_track_stop20.csv", draws, delimiter=",")
        np.savetxt("results/recommend_track_stop20.csv",
                   recommend, delimiter=",")
        np.savetxt("results/recommens_array20.csv",
                   recommends_save, delimiter=",")
        np.savetxt("results/scores_array20.csv",
                   scores_save, delimiter=",")

        # print("proportion of runs that did not terminate: $(FracNT)\n")
        # print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
        # print("average proportions of draws: $(mean(proportion,1))\n")
        # print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
        # print("proportion of recommendation made when termination: $(FracReco)\n")
        # print("elapsed time: $(time()-startTime)\n\n")
        # name = "$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
        # h5write(name, "mu", mu)
        # h5write(name, "delta", delta)
        # h5write(name, "FracNT", collect(FracNT))
        # h5write(name, "FracReco", FracReco)
        # h5write(name, "Draws", Draws)
        # h5write(name, "Error", Error)

        print(FracReco)


if __name__ == '__main__':
    SaveData(mu, delta, num_similator)
