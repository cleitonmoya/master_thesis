"""
BOCD Models
Adapted from Matías Altamirano (https://github.com/maltamiranomontero/DSM-bocd)
Copyright (c) 2023 Matías Altamirano

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
IN THE SOFTWARE.
"""

import numpy as np
from scipy import stats


class GaussianUnknownMean:
    def __init__(self, mean0, var0, varx):
        """Initialize model, for standard Bayes.
        Prior: Normal
        Likelihood: Normal known variance
        Predictive posterior: GaussNormalian
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def log_pred_prob(self, t, x, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t,
        update all run length hypotheses.
        """
        new_prec_params = self.prec_params + (1/self.varx)
        new_mean_params = (self.mean_params * self.prec_params +
                           (x / self.varx)) / new_prec_params

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx


class Gaussian:
    def __init__(self, mu0, kappa0, alpha0, omega0):
        """Initialize model, for standard Bayes.
        Prior: Normal-inverse gamma
        Likelihood: Normal
        Predictive posterior: t-student
        """
        self.alpha = np.array([alpha0])
        self.alpha0 = np.array([alpha0])

        self.omega = np.array([omega0])
        self.omega0 = np.array([omega0])

        self.kappa = np.array([kappa0])
        self.kappa0 = np.array([kappa0])

        self.mu = np.array([mu0])
        self.mu0 = np.array([mu0])
        
        
    def log_pred_prob(self, x, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        df = 2 * self.alpha[indices]
        loc = self.mu[indices]
        scale = np.sqrt(self.omega[indices] * (self.kappa[indices] + 1) /
                        (self.alpha[indices] * self.kappa[indices]))

        return stats.t.logpdf(x=x, df=df, loc=loc, scale=scale)

    def update_params(self, x):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """

        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + x) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        omegaT0 = np.concatenate(
            (
             self.omega0,
             self.omega
             + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )    

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.omega = omegaT0


class Hazard:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ConstantHazard(Hazard):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def __call__(self, r):
        if isinstance(r, np.ndarray):
            shape = r.shape
        else:
            shape = 1

        return np.ones(shape) / self._lambda


# Verify if the change-point was alreay identified
# @author: Cleiton Moya
def check_previous_cp(cp, CP, min_seg):
    previous = False
    for j in range(min_seg+1):
        if ((cp-j) in CP) or ((cp+j) in CP):
            previous = True
    return previous