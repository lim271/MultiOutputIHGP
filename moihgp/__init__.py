import os.path
import ctypes
import numpy as np



class MOIHGPOnlineLearning(object):

    def __init__(self, dt, num_output, num_latent, gamma=0.9, l1_reg=1.0, windowsize=1, threading=False, kernel="Matern32", libpath=None):
        self.dt = dt
        self.num_output = num_output
        self.num_latent = num_latent
        self.gamma = gamma
        self.l1_reg = l1_reg
        self.windowsize = windowsize
        self.threading = threading
        if libpath is None:
            libpath = os.path.dirname(__file__)
        lib = os.path.join(libpath, "build/libmoihgp_ctypes.so")
        if os.path.exists(lib):
            self.lib = ctypes.cdll.LoadLibrary(lib)
        else:
            raise FileNotFoundError("libmoihgp_ctypes.so not found.")
        if kernel=="Matern32":
            self.lib.online32_new.restype = ctypes.c_void_p
            self.__obj = self.lib.online32_new(
                ctypes.c_double(dt),
                ctypes.c_size_t(num_output),
                ctypes.c_size_t(num_latent),
                ctypes.c_double(gamma),
                ctypes.c_double(l1_reg),
                ctypes.c_size_t(windowsize),
                ctypes.c_bool(threading)
            )
            self.lib.online32_getNumParam.restype = ctypes.c_size_t
            num_param = self.lib.online32_getNumParam(self.__obj)
            self.num_param = int(num_param)
            self.__del = self.lib.online32_del
            self.__step = self.lib.online32_step
            self.__params = self.lib.online32_getParams
        elif kernel=="Matern52":
            self.lib.online52_new.restype = ctypes.c_void_p
            self.__obj = self.lib.online52_new(
                ctypes.c_double(dt),
                ctypes.c_size_t(num_output),
                ctypes.c_size_t(num_latent),
                ctypes.c_double(gamma),
                ctypes.c_double(l1_reg),
                ctypes.c_size_t(windowsize),
                ctypes.c_bool(threading)
            )
            self.lib.online52_getNumParam.restype = ctypes.c_size_t
            num_param = self.lib.online52_getNumParam(self.__obj)
            self.num_param = int(num_param)
            self.__del = self.lib.online52_del
            self.__step = self.lib.online52_step
            self.__params = self.lib.online52_getParams
        else:
            raise NotImplementedError("Unsupported kernel type.")
        self.__del.restype = ctypes.c_void_p
        self.__step.restype = ctypes.c_void_p
        self.__params.restype = ctypes.c_void_p
        self.__step.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        self.__params.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double)
        ]


    def __del__(self):
        self.__del(self.__obj)
        del self.lib


    def step(self, y):
        y = y.astype(np.float64)
        y_p = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        yhat = np.zeros_like(y)
        yhat_p = np.ctypeslib.as_ctypes(yhat)
        self.__step(self.__obj, y_p, yhat_p, self.num_output)
        return yhat.astype(np.float64)


    @property
    def params(self):
        params = np.zeros((self.num_param))
        params_p = np.ctypeslib.as_ctypes(params)
        self.__params(self.__obj, params_p)
        return params.astype(np.float64)


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
