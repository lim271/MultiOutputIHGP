import os.path
from ctypes import POINTER, c_bool, c_double, c_size_t, c_void_p, cdll
import numpy as np

c_double_p = POINTER(c_double)



class MOIHGP(object):

    def __init__(self, dt, num_output, num_latent, kernel="Matern32", l1_reg=0.0, threading=False):
        self.dt = dt
        self.__num_output = num_output
        self.__num_latent = num_latent
        lib = os.path.join(os.path.dirname(__file__), "build/libmoihgp.so")
        self.__lib = cdll.LoadLibrary(lib)
        if kernel=="Matern32":
            self.__lib.gp32_new.restype = c_void_p
            self.__obj = self.__lib.gp32_new(
                c_double(dt),
                c_size_t(num_output),
                c_size_t(num_latent),
                c_double(np.float64(l1_reg)),
                c_bool(threading)
            )
            self.__del = self.__lib.gp32_del
            self.__lib.gp32_num_param.restype = c_size_t
            self.__num_param = int(self.__lib.gp32_num_param(self.__obj))
            self.__lib.gp32_num_igp_param.restype = c_size_t
            self.__num_igp_param = int(self.__lib.gp32_num_igp_param(self.__obj))
            self.__lib.gp32_igp_dim.restype = c_size_t
            self.__igp_dim = int(self.__lib.gp32_igp_dim(self.__obj))
            self.__del = self.__lib.gp32_del
            self.__step1 = self.__lib.gp32_step1
            self.__step2 = self.__lib.gp32_step2
            self.__step3 = self.__lib.gp32_step3
            self.__step4 = self.__lib.gp32_step4
            self.__update = self.__lib.gp32_update
            self.__lik1 = self.__lib.gp32_lik1
            self.__lik2 = self.__lib.gp32_lik2
            self.__get_params = self.__lib.gp32_get_params
        elif kernel=="Matern52":
            self.__lib.gp52_new.restype = c_void_p
            self.__obj = self.lib.gp52_new(
                c_double(dt),
                c_size_t(num_output),
                c_size_t(num_latent),
                c_double(np.float64(l1_reg)),
                c_bool(threading)
            )
            self.__del = self.__lib.gp52_del
            self.__lib.gp52_num_param.restype = c_size_t
            self.__num_param = int(self.__lib.gp52_num_param(self.__obj))
            self.__lib.gp52_num_igp_param.restype = c_size_t
            self.__num_igp_param = int(self.__lib.gp52_num_igp_param(self.__obj))
            self.__lib.gp52_igp_dim.restype = c_size_t
            self.__igp_dim = int(self.__lib.gp52_igp_dim(self.__obj))
            self.__del = self.__lib.gp52_del
            self.__step1 = self.__lib.gp52_step1
            self.__step2 = self.__lib.gp52_step2
            self.__step3 = self.__lib.gp52_step3
            self.__step4 = self.__lib.gp52_step4
            self.__update = self.__lib.gp52_update
            self.__lik1 = self.__lib.gp52_lik1
            self.__lik2 = self.__lib.gp52_lik2
            self.__get_params = self.__lib.gp52_get_params
        else:
            raise NotImplementedError("Unsupported kernel type.")
        self.__del.restype = c_void_p
        self.__step1.restype = c_void_p
        self.__step2.restype = c_void_p
        self.__step3.restype = c_void_p
        self.__step4.restype = c_void_p
        self.__update.restype = c_void_p
        self.__lik1.restype = c_double
        self.__lik2.restype = c_double
        self.__get_params.restype = c_void_p
        self.__step1.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        self.__step2.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        self.__step3.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        self.__step4.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        self.__update.argtypes = [
            c_void_p,
            c_double_p,
        ]
        self.__lik1.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
            c_double_p,
            c_double_p,
        ]
        self.__lik2.argtypes = [
            c_void_p,
            c_double_p,
            c_double_p,
        ]
        self.__get_params.argtypes = [
            c_void_p,
            c_double_p
        ]
        self.__params = np.zeros((self.num_param,), dtype=np.float64)
        self.__params_p = self.__params.ctypes.data_as(c_double_p)
        self.__grad = np.zeros((self.num_param,), dtype=np.float64)
        self.__grad_p = self.__grad.ctypes.data_as(c_double_p)
        self.__x = np.zeros((self.num_latent, self.igp_dim), dtype=np.float64)
        self.__x_p = self.__x.ctypes.data_as(c_double_p)
        self.__y = np.zeros((self.num_output,), dtype=np.float64)
        self.__y_p = self.__y.ctypes.data_as(c_double_p)
        self.__dx = np.zeros(
            (self.num_latent, self.num_igp_param, self.igp_dim),
            dtype=np.float64
        )
        self.__dx_p = self.__dx.ctypes.data_as(c_double_p)
        self.__xnew = np.zeros((self.num_latent, self.igp_dim), dtype=np.float64)
        #self.__xnew_p = np.ctypeslib.as_ctypes(self.__xnew)
        self.__xnew_p = self.__xnew.ctypes.data_as(c_double_p)
        self.__yhat = np.zeros((self.num_output,), dtype=np.float64)
        #self.__yhat_p = np.ctypeslib.as_ctypes(self.__yhat)
        self.__yhat_p = self.__yhat.ctypes.data_as(c_double_p)
        self.__dxnew = np.zeros(
            (self.num_latent, self.num_igp_param, self.igp_dim),
            dtype=np.float64
        )
        #self.__dxnew_p = np.ctypeslib.as_ctypes(self.__dxnew)
        self.__dxnew_p = self.__dxnew.ctypes.data_as(c_double_p)


    def __del__(self):
        self.__del(self.__obj)
        del self.__lib


    def step(self, x, y=None, dx=None):
        self.__x.setfield(x, dtype=np.float64)
        if y is None:
            self.__step4(self.__obj, self.__x_p, self.__xnew_p, self.__yhat_p)
            return self.__xnew.astype(np.float64), self.__yhat.astype(np.float64)
        else:
            self.__y.setfield(y, dtype=np.float64)
            if dx is None:
                self.__step3(
                    self.__obj,
                    self.__x_p, self.__y_p,
                    self.__xnew_p, self.__yhat_p
                )
                return self.__xnew.astype(np.float64), self.__yhat.astype(np.float64)
            else:
                self.__dx.setfield(dx, dtype=np.float64)
                self.__step1(
                    self.__obj,
                    self.__x_p, self.__y_p, self.__dx_p,
                    self.__xnew_p, self.__yhat_p, self.__dxnew_p
                )
                return self.__xnew.astype(np.float64), self.__yhat.astype(np.float64), self.__dxnew.astype(np.float64)


    def update(self, params):
        self.__params.setfield(params, dtype=np.float64)
        self.__update(self.__obj, self.__params_p)


    def negLogLikelihood(self, x, y, dx=None):
        self.__x.setfield(x, dtype=np.float64)
        self.__y.setfield(y, dtype=np.float64)
        if dx is None:
            res = self.__lik2(
                self.__obj,
                self.__x_p, self.__y_p,
            )
            loss = np.float64(res)
            return loss
        else:
            self.__dx.setfield(dx, dtype=np.float64)
            res = self.__lik1(
                self.__obj,
                self.__x_p, self.__y_p, self.__dx_p,
                self.__grad_p
            )
            loss = np.float64(res)
            return loss, self.__grad.astype(np.float64)


    @property
    def num_output(self):
        return self.__num_output


    @property
    def num_latent(self):
        return self.__num_latent


    @property
    def igp_dim(self):
        return self.__igp_dim


    @property
    def num_param(self):
        return self.__num_param


    @property
    def num_igp_param(self):
        return self.__num_igp_param


    @property
    def params(self):
        self.__get_params(self.__obj, self.__params_p)
        return self.__params


    @property
    def covariance(self):
        params = self.params.copy()
        U = np.reshape(params[:self.num_output * self.num_latent], (self.num_output, self.num_latent))
        sqrtS = np.diag(
            np.sqrt(
                params[self.num_output * self.num_latent:(self.num_output+1) * self.num_latent]
            )
        )
        B = []
        igp_params = np.reshape(params[-self.num_latent * 3:], (self.num_latent, 3))
        for magnitude, lengthscale, _ in igp_params:
            B.append(magnitude**0.5 * (3**0.5/lengthscale**0.5)**1.5)
        B = np.diag(B)
        return U @ sqrtS @ B @ sqrtS @ U.T
