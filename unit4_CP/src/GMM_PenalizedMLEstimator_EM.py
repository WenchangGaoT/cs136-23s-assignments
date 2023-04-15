'''
Summary
=======
Defines a penalized ML estimator for Gaussian Mixture Models, using EM

EM = Expectation-Maximization.

Provides a CONCRETE implementation of an sk-learn-like estimator API

* fit
* score
* get_params
* set_params
'''

import numpy as np
from collections import defaultdict
import scipy.stats as stats
from scipy.special import logsumexp
import scipy.optimize
import time

from GMM_PenalizedMLEstimator import GMM_PenalizedMLEstimator
from GMM_PenalizedMLEstimator import calc_neg_log_lik

class GMM_PenalizedMLEstimator_EM(GMM_PenalizedMLEstimator):
    """ Maximum Likelihood Estimator for Gaussian Mixtures, trained with EM.

    Attributes
    ----------
    K : int
        Number of components
    D : int
        Number of data dimensions
    seed : int
        Seed for random number generator used for initialization
    variance_penalty_mode : float
        Must be positive.
        Defines mode of penalty on variance.
        See calc_penalty_stddev module.
    variance_penalty_spread : float,
        Must be positive.
        Defines spread of penalty on variance.
        See calc_penalty_stddev module.
    max_iter : int
        Maximum allowed number of iterations for training algorithm
    ftol : float
        Threshold that determines if training algorithm has converged
        Same definition as `ftol` setting used by scipy.optimize.minimize

    Additional Attributes (after calling fit)
    -----------------------------------------
    log_pi_K : 1D array, shape (K,)
        GMM parameter: Log of mixture weights
        Must satisfy logsumexp(log_pi_K) == 0.0 (which means sum(exp(log_pi_K)) == 1.0)
    mu_KD : 2D array, shape (K, D)
        GMM parameter: Means of all components
        The k-th row is the mean vector for the k-th component
    stddev_KD : 2D array, shape (K, D)
        GMM parameter: Standard Deviations of all components
        The k-th row is the stddev vector for the k-th component
    history : dict of lists
        Access performance metrics computed throughout iterative training.
        history['iter'] contains integer iteration count at each checkpoint
        history['train_loss'] contains training loss value at each checkpoint
        history['valid_score_per_pixel'] contains validation score at each checkpoint
            Normalized "per pixel" means divided by total number of observed feature dimensions (pixels)
            So that values for different size datasets can be fairly compared.

    Inherits
    --------
    * Constructor __init__() from GMM_PenalizedMLEstimator_LBFGS parent class
    * Initialization method generate_initial_parameters() from parent as well
    """

    
    def calc_EM_loss(self, r_NK, x_ND):
        ''' Compute the overall loss function minimized by the EM algorithm

        Includes three additive terms:
        * Negative of the expected complete likelihoods E_q[ log p(x,z)]
        * Negative of the entropy of the assignment distribution q(z|r)
        * Penalty on the standard deviation parameters

        Args
        ----
        r_NK : 2D array, shape (N, K)
            Parameters that define the approximate assignment distribution q(z)
            The n-th row r_NK[n] defines the K-length vector r_n that is non-negative & sums to one.
            Can interpret r_NK[n,k] as the probability of assigning cluster k to n-th example
            Formally, the n-th example's assignment distribution is given by:
                q(z_n | r_n) = CategoricalPMF(z_n | r_n[0], r_n[1], ... r_n[K-1])
        x_ND : 2D array, shape (N, D)
            Dataset of observed feature vectors
            The n-th row x_ND[n] defines a length-D feature vector

        Returns
        -------
        loss_em : float
            scalar value of the loss of provided x and r arrays
            Uses this object's internal GMM params (self.log_pi_K, self.mu_KD, self.stddev_KD)
        '''
        log_prior = np.sum(np.dot(r_NK, self.log_pi_K))
        log_lik = 0.0
        for k in range(self.K):
            log_lik_k_N = np.sum(stats.norm.logpdf(x_ND, self.mu_KD[k], self.stddev_KD[k]), axis=1)
            log_lik += np.inner(r_NK[:,k], log_lik_k_N)
        entropy = -1.0 * np.sum(r_NK * np.log(r_NK + 1e-100))
        penalty_stddev = self.calc_penalty_stddev()
        return -1.0 * (log_prior + log_lik + entropy) + penalty_stddev

    def estep__calc_r_NK(self, x_ND):
        N = x_ND.shape[0]
        r_NK = np.zeros((N, self.K), dtype=np.float64)

        log_r_NK = np.zeros_like(r_NK, dtype=np.float64)
        for k in range(self.K):
            log_r_NK[:, k] = self.log_pi_K[k]
            # for d in range(self.D):
            #     log_r_NK[:, k] += stats.norm.logpdf(x_ND[:, d], self.mu_KD[k, d], self.stddev_KD[k, d])
            log_r_NK[:, k] += stats.multivariate_normal.logpdf(x_ND, mean=self.mu_KD[k, :], cov=self.stddev_KD[k, :]**2)

        log_deno = logsumexp(log_r_NK, axis=1)
        # for k in range(self.K):
        #     log_r_NK[:, k] -= log_deno
        log_r_NK -= log_deno.reshape(-1, 1)
        r_NK = np.exp(log_r_NK)
        # r_NK = np.exp(log_r_NK)/np.exp(log_deno).reshape(-1,1)
        # r_NK = np.exp(log_r_NK)

        if not np.allclose(np.sum(r_NK, axis=1), 1.0):
            print(log_r_NK)
        assert np.allclose(np.sum(r_NK, axis=1), 1.0)
        return r_NK

    def mstep__update_log_pi_K(self, r_NK):
        # log_pi_K = np.log(np.mean(r_NK, axis=0))
        N, K = r_NK.shape
        log_pi_K = np.log(np.sum(r_NK, axis=0)/N)
        if not np.allclose(logsumexp(log_pi_K), 0.0):
            print(r_NK)
        assert np.allclose(logsumexp(log_pi_K), 0.0)
        return log_pi_K

    def mstep__update_mu_KD(self, r_NK, x_ND):
        mu_KD = np.zeros((self.K, self.D)) # FIXME

        for k in range(self.K):
            # for d in range(self.D):
            mu_KD[k, :] = np.sum(x_ND*r_NK[:, k].reshape(-1,1), axis=0)/np.sum(r_NK[:, k])

        return mu_KD

    def mstep__update_stddev_KD(self, r_NK, x_ND):
        ## TODO compute optimal update of stddev_KD
        m = self.variance_penalty_mode
        s = self.variance_penalty_spread

        stddev_KD = np.zeros((self.K, self.D), dtype=np.float64)

        for k in range(self.K):
            temp_ND = x_ND-self.mu_KD[k, :]
            temp_ND = temp_ND ** 2
            
            nomi = np.sum(r_NK[:, k].reshape(-1,1)*temp_ND, axis=0)+1./m/s
            deno = np.sum(r_NK[:, k])+1./(s*m*m)
            stddev_KD[k, :] = nomi/deno
            # assert (stddev_KD[k, :] >= m).all()
            stddev_KD[k, :] = np.sqrt(stddev_KD[k, :])
            # print(stddev_KD)
        # stddev_KD = np.sqrt(stddev_KD)
        return stddev_KD

    def fit(self, x_ND, x_valid_ND=None, verbose=True):

        N = np.maximum(x_ND.shape[0], 1.0)

        ## Define initial parameters
        if not hasattr(self, 'log_pi_K'):
            self.log_pi_K, self.mu_KD, self.stddev_KD = self.generate_initial_parameters(x_ND)

        self.history = defaultdict(list)
        start_time_sec = time.time()
        for iter_id in range(self.max_iter + 1):

            ## Loss step
            ND = N * x_ND.shape[1]
            tr_score = self.score(x_ND)
            loss_with_penalty = -1.0 * tr_score + self.calc_penalty_stddev()
            self.history['iter'].append(iter_id)
            self.history['train_loss'].append(loss_with_penalty)
            self.history['train_loss_per_pixel'].append(loss_with_penalty / ND)
            self.history['train_score_per_pixel'].append(tr_score / ND)

            if x_valid_ND is None:
                va_score_message = ""
            else:
                # TODO compute the per-pixel negative log likelihood on validation set
                N_val, D = x_valid_ND.shape
                va_score_per_pixel = -1.0*calc_neg_log_lik(x_valid_ND, self.log_pi_K, self.mu_KD, self.stddev_KD) / (N_val*D)
                self.history['valid_score_per_pixel'].append(va_score_per_pixel)
                va_score_message = "| valid score %9.6f" % (self.history['valid_score_per_pixel'][-1])

            ## E step
            r_NK = self.estep__calc_r_NK(x_ND)
            if self.do_double_check_correctness:
                # Verify the loss after E step is equal to the incomplete loss
                loss_e = self.calc_EM_loss(r_NK, x_ND)
                self.history['train_loss_em'].append(loss_e)
                ## TODO this should pass: assert np.allclose(loss_with_penalty, loss_e)
                assert np.allclose(loss_with_penalty, loss_e)

            ## M step
            if r_NK.shape[0] > 1:
                self.log_pi_K = self.mstep__update_log_pi_K(r_NK)
                self.mu_KD = self.mstep__update_mu_KD(r_NK, x_ND)
            self.stddev_KD = self.mstep__update_stddev_KD(r_NK, x_ND)

            if self.do_double_check_correctness:
                # Verify the loss goes down after the M step
                loss_m = self.calc_EM_loss(r_NK, x_ND)
                self.history['train_loss_em'].append(loss_m)
                ## TODO this should pass: assert loss_m <= loss_e + 1e-9
                assert loss_m <= loss_e + 1e-9

            if verbose:
                print("iter %4d / %4d after %9.1f sec | train loss % 9.6f %s" % (
                    iter_id, self.max_iter, time.time() - start_time_sec,
                    self.history['train_loss_per_pixel'][-1],
                    va_score_message,
                    ))
            # The iteration stops when
            # (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= ftol.
            if iter_id >= 2:
                fnew = self.history['train_loss'][-1]
                fold = self.history['train_loss'][-2]
                numer = (fold - fnew)
                denom = np.max(np.abs([fnew, fold, 1]))
                if numer / denom <= self.ftol:
                    break


if __name__ == '__main__':
    np.set_printoptions(suppress=False, precision=3, linewidth=80)
    D = 2

## Verify that variance penalty works as expected
# Empty components (with no assigned data) should have variance equal to the intended "mode" of the penalty
# We'll use a mode of 2.0 (so stddev = sqrt(2.0) = 1.414...)
    gmm_em = GMM_PenalizedMLEstimator_EM(K=3, D=2, seed=42, variance_penalty_mode=2.0)
    empty_ND = np.zeros((0,D))
    log_pi_K, mu_KD, stddev_KD = gmm_em.generate_initial_parameters(empty_ND)
    print(calc_neg_log_lik(empty_ND, log_pi_K, mu_KD, stddev_KD))
    # -0.0
    gmm_em.fit(empty_ND, verbose=False)
    print(gmm_em.stddev_KD)

    N = 25; K = 3
    prng = np.random.RandomState(8675309)
    x1_ND = 0.1 * prng.randn(N, D) + np.asarray([[0, 0]])
    x2_ND = 0.1 * prng.randn(N, D) + np.asarray([[-1, 0]])
    x3_ND = np.asarray([[0.2, 0.05]]) * prng.randn(N, D) + np.asarray([[0, +1]])
    x_ND = np.vstack([x1_ND, x2_ND, x3_ND])
    gmm_em = GMM_PenalizedMLEstimator_EM(K=3, D=2, seed=42, variance_penalty_mode=2.0, max_iter=1)

    gmm_em.stddev_KD = 0.1 * np.ones((K,D))
    gmm_em.stddev_KD[-1] = [0.2, 0.05]
    gmm_em.mu_KD = np.asarray([[0, 0], [-1., 0], [0, 1.]])
    gmm_em.log_pi_K = np.log(1./3 * np.ones(K))
    print(gmm_em.estep__calc_r_NK(x_ND[:3]))
    #array([[1.000e+00, 5.336e-25, 3.829e-75],
    #    [1.000e+00, 2.151e-17, 3.063e-97],
    #    [1.000e+00, 4.367e-19, 1.984e-90]])
    print(gmm_em.estep__calc_r_NK(x_ND[-3:]))
    # array([[4.752e-25, 1.362e-38, 1.000e+00],
    #     [2.278e-17, 7.579e-46, 1.000e+00],
    #     [4.189e-22, 4.117e-34, 1.000e+00]])
    gmm_em.fit(x_ND, verbose=False)
    print(np.exp(gmm_em.log_pi_K))
    # array([0.333, 0.333, 0.333])
    print(gmm_em.mu_KD)
    # array([[-0.007,  0.01 ],
    #     [-1.008,  0.009],
    #     [-0.005,  1.005]])

    print(gmm_em.stddev_KD)
    # array([[0.076, 0.091],
    #     [0.098, 0.103],
    #     [0.24 , 0.042]])

    gmm_em = GMM_PenalizedMLEstimator_EM( K=3, D=2, seed=42, variance_penalty_mode=2.0, 
                                         max_iter=1000, do_double_check_correctness=True)
    gmm_em.fit(x_ND, verbose=False)
    print(np.exp(gmm_em.log_pi_K))
    # array([0.333, 0.333, 0.333])
    print(gmm_em.mu_KD)
    # array([[-1.008,  0.009],
    #     [-0.005,  1.005],
    #     [-0.007,  0.01 ]])
    print(gmm_em.stddev_KD)
    # array([[0.098, 0.103],
    #     [0.24 , 0.042],
    #     [0.076, 0.091]])