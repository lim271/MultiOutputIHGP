import numpy as np
try:
    from scipy.optimize.lbfgsb import _minimize_lbfgsb, MemoizeJac
except:
    from scipy.optimize._lbfgsb_py import _minimize_lbfgsb, MemoizeJac
from .pywrapper import MOIHGP



class MOIHGPOnlineLearning:

    def __init__(self, dt, num_output, num_latent, gamma, x_init=None, windowsize=None, kernel="Matern32", threading=False):
        self.moihgp = MOIHGP(dt, num_output, num_latent, kernel=kernel, threading=threading)
        self.num_output = num_output
        self.num_latent = num_latent
        self.ihgp_dim = self.moihgp.igp_dim
        self.ihgp_nparam = self.moihgp.num_igp_param
        self.parameter_bounds = [
            (-np.inf, np.inf) # bounds of U
        ] * (num_output * num_latent) + [
            (1e-4, np.inf) # bounds of S
        ] * num_latent + [
            (1e-4, 1e+2) # bounds of mixing noise
        ] + [
            (1e-2, 1e+2), # bounds of IHGP magnitude
            (1e-2, 1e+2), # bounds of IHGP lengthscale
            (1e-4, 1e+2)  # bounds of IHGP noise
        ] * num_latent
        self.gamma = gamma
        self.x = np.zeros(
            (num_latent, self.ihgp_dim),
            dtype=np.float64
        ) if x_init is None else x_init
        self.dx = np.zeros(
            (num_latent, self.ihgp_nparam, self.ihgp_dim),
            dtype=np.float64
        )
        self.xinit = np.zeros(
            (num_latent, self.ihgp_dim),
            dtype=np.float64
        ) if x_init is None else x_init
        self.dxinit = np.zeros(
            (num_latent, self.ihgp_nparam, self.ihgp_dim),
            dtype=np.float64
        )
        self.hess_inv = np.eye(len(self.moihgp.params))
        self.buffer = []
        self.windowsize = 1 if windowsize is None else windowsize
        self.ma = np.zeros((num_output,), dtype=np.float64)


    def step(self, y=None):
        self.buffer.append(y)
        self.ma = np.mean(self.buffer, axis=0)
        while len(self.buffer) > self.windowsize:
            self.buffer.pop(0)
            self.xinit, _, self.dxinit = self.moihgp.step(self.xinit, y = self.buffer[0] - self.ma, dx = self.dxinit)
        xnew, yhat, dxnew = self.moihgp.step(self.x, y = y - self.ma, dx = self.dx)
        yhat += self.ma
        self.x = xnew
        self.dx = dxnew
        oldparams = self.moihgp.params.copy()
        def objective(params, eval_gradient=True):
            dparams = params - oldparams
            self.moihgp.update(params)
            p = np.linalg.solve(self.hess_inv, dparams)
            xt = self.xinit
            dxt = self.dxinit
            if eval_gradient:
                loss = self.gamma * 0.5 * dparams.dot(p)
                grad = self.gamma * p
                for yt in self.buffer:
                    xtnew, _, dxtnew = self.moihgp.step(xt, y = yt - self.ma, dx = dxt)
                    l, g = self.moihgp.negLogLikelihood(xt, yt - self.ma, dxt)
                    loss += l
                    grad += g
                    xt = xtnew
                    dxt = dxtnew
                return loss, grad
            else:
                loss = self.gamma * 0.5 * dparams.dot(p)
                for yt in self.buffer:
                    xtnew, _, dxtnew = self.moihgp.step(xt, y = yt - self.ma, dx = dxt)
                    loss += self.moihgp.negLogLikelihood(xt, yt - self.ma)
                    xt = xtnew
                    dxt = dxtnew
                return loss
        fun = MemoizeJac(objective)
        jac = fun.derivative
        res = _minimize_lbfgsb(fun, oldparams, bounds=self.parameter_bounds, jac=jac, maxiter=5, maxls=3)
        newparams = res['x']
        self.moihgp.update(newparams)
        self.hess_inv = res['hess_inv'].todense()
        return yhat


    @property
    def covariance(self):
        return self.moihgp.covariance


    @property
    def params(self):
        return self.moihgp.params
