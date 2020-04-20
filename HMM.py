'''
Problem 1 (Likelihood): 
Given an HMM λ = (A, B) and an observation sequence O, 
determine the likelihood P(O|λ ).

Problem 2 (Decoding): 
Given an observation sequence O and an HMM λ = (A, B), 
discover the best hidden state sequence Q.

Problem 3 (Learning): 
Given an observation sequence O and the set of states 
in the HMM, learn the HMM parameters A and B.
'''
import numpy as np


def forward(obs, em, pi, P_s, likelihood=False):# likelihood
	'''
	obs: list of observations
	em:  matrix of emission probs for each state and observation
	pi:  vector of initialisation probabilities
	P_s: markov chain matrix for the states
	'''
	T = len(obs)
	N = len(em)
	frwd = np.zeros((N, T))
	for s in range(N):
		frwd[s, 0] = pi[s] * em[s][obs[0]]
	for t in range(1, T):
		for s in range(N):
			frwd[s, t] = sum([frwd[z, t-1]*P_s[z, s]*em[s][obs[t]] for z in range(N)])
	if likelihood:
		# print(frwd)
		return sum(frwd[:, -1])
	return frwd

def viterbi(obs, em, pi, P_s):# decoding
	'''
	obs: list of observations
	em:  matrix of emission probs for each state and observation
	pi:  vector of initialisation probabilities
	P_s: markov chain matrix for the states
	'''
	T = len(obs)
	N = len(em)
	vtrb = np.zeros((N, T))
	bkpt = np.zeros((N, T), dtype=int)
	for s in range(N):
		vtrb[s, 0] = pi[s] * em[s][obs[0]]
	for t in range(1, T):
		for s in range(N):
			ar = np.array([vtrb[z, t-1]*P_s[z, s]*em[s][obs[t]] for z in range(N)])
			# print(ar)
			bkpt[s, t] = np.argmax( ar )
			vtrb[s, t] = ar[bkpt[s, t]]
	best_end = np.argmax(vtrb[:, -1])
	prob_best_path = vtrb[best_end, -1]
	path = [best_end]
	for t in range(T-1, 0, -1):
		# print(t)
		path.append(bkpt[path[-1], t])
	# print(vtrb)
	# print(bkpt)
	# print(best_end)
	return path[::-1], prob_best_path

'''
The standard algorithm for HMM training is the forward-backward/Baum-Welch 
algorithm, (special case of the Expectation-Maximisation/EM algorithm. The 
algorithm will let us train both the transition and emission probabilities.
EM is an iterative algorithm, computing an initial estimate for the 
probabilities, then using those estimates to computing a better estimate, 
and so on, iteratively improving the probabilities that it learns.
'''
def backward(obs, em, pi, P_s):
	'''
	obs: list of observations
	em:  matrix of emission probs for each state and observation
	pi:  vector of initialisation probabilities
	P_s: markov chain matrix for the states
	'''
	T = len(obs)
	N = len(em)
	bkwd = np.ones((N, T))
	for t in range(T-2, -1, -1):
		for s in range(N):
			bkwd[s, t] = sum([bkwd[z, t+1]*P_s[s, z]*em[z][obs[t]] for z in range(N)])
	# for s in range(N):
	# 	bkwd[s, 0] = pi[s] * em[s][obs[0]]
	p_o = sum([pi[s]*em[s][obs[0]]*bkwd[s, 0] for s in range(N)])
	return bkwd, p_o

def baumWelch(obs, V, N, pi, acc=0.001):
	T = len(obs)
	a, b = np.ones((N, N)), np.ones((N, V))
	A = B = 0
	while np.sum(np.abs(B-b)) > acc:
		print((a, b))
		print((A, B))
		A, B = a, b
		# E-step
		alpha = forward(obs, b, pi, a)
		beta, p_o = backward(obs, b, pi, a)
		gamma = np.zeros(beta.shape)
		for s in range(beta.shape[0]):
			for t in range(beta.shape[1]):
				gamma[s, t] = alpha[s, t] * beta[s, t] / p_o
		zeta = np.zeros((N, N, T-1))
		for s in range(N):
			for ns in range(N):
				for t in range(T-1):
					zeta[s, ns, t] = alpha[s, t] * A[s, ns] * B[ns][obs[t+1]] * beta[s, t+1] / p_o
		# M-step
		zeta = np.sum(zeta, axis=2)
		zet = np.sum(zeta, axis=1)
		a = (zeta.T / zet).T
		ga = np.sum(gamma, axis=1)
		b = np.zeros((N, V))
		for t in range(T):
			b[:, obs[t]] += gamma[:, t]
		b = (b.T / ga).T
	print((a, b))
	print((A, B))
	return a, b

pi = np.array([.8, .2])
P_s = np.array([[.6, .4], [.5, .5]])
em = np.array([[.2, .4, .4], [.5, .4, .1]])
obs = [2,0,2]
print('\n\n')
print(forward(obs, em, pi, P_s, likelihood=True))
print('\n\n')
print(viterbi(obs, em, pi, P_s))
N=2
V=3
print('\n\n')
baumWelch(obs, V, N, pi)
