#include <cstdlib>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <moihgp/moihgp.h>
#include <moihgp/matern32ss.h>
#include <moihgp/matern52ss.h>



typedef moihgp::MOIHGP<moihgp::Matern32StateSpace> GP32;
typedef moihgp::MOIHGP<moihgp::Matern32StateSpace> GP52;



extern "C"
{



GP32* gp32_new(double dt, size_t num_output, size_t num_latent, double lambda, bool threading)
{
    return new GP32(dt, num_output, num_latent, lambda, threading);
} // GP32* gp32_new(double dt, size_t num_output, size_t num_latent, bool threading)


void gp32_del(GP32* gp)
{
    gp->~MOIHGP();
} // void gp32_del(GP32* gp)


void gp32_step1(GP32* gp, double* x, double* y, double* dx, double* xnew, double* yhat, double* dxnew)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dxnew(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    gp->step(_x, _y, _dx, _xnew, _yhat, _dxnew);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                dxnew[idx1 * igp_num_param * dim + idx2 * dim + idx3] = _dxnew[idx1][idx2](idx3);
            }
        }
    }
} // void gp32_step1(GP32* gp, double* x, double* y, double* dx, double* xnew, double* yhat, double* dxnew)


void gp32_step2(GP32* gp, double* x, double* y, double* dx, double* xnew, double* dxnew)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    std::vector<std::vector<Eigen::VectorXd>> _dxnew(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    gp->step(_x, _y, _dx, _xnew, _dxnew);

    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                dxnew[idx1 * igp_num_param * dim + idx2 * dim + idx3] = _dxnew[idx1][idx2](idx3);
            }
        }
    }
} // void gp32_step2(GP32* gp, double* x, double* y, double* dx, double* xnew, double* dxnew)


void gp32_step3(GP32* gp, double* x, double* y, double* xnew, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    gp->step(_x, _y, _xnew, _yhat);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
    }
} // void gp32_step3(GP32* gp, double* x, double* y, double* xnew, double* yhat)


void gp32_step4(GP32* gp, double* x, double* xnew, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);

    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    gp->step(_x, _xnew, _yhat);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
    }
} // void gp32_step4(GP32* gp, double* x, double* xnew, double* yhat)


void gp32_update(GP32* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = Eigen::Map<Eigen::VectorXd>(params, num_param, 1);
    gp->update(_params);
} // void gp32_update(GP32* gp, double* params)


double gp32_lik1(GP32* gp, double* x, double* y, double* dx, double* grad)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t num_param = gp->getNumParam();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    Eigen::VectorXd _grad(num_param);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    double loss = gp->negLogLikelihood(_x, _y, _dx, _grad);

    for (size_t idx=0; idx < num_param; idx++)
    {
        grad[idx] = _grad(idx);
    }

    return loss;
} // double gp32_lik1(GP32* gp, double* x, double* y, double* dx, Eigen::VectorXd &grad)


double gp32_lik2(GP32* gp, double* x, double* y)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t num_param = gp->getNumParam();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    double loss = gp->negLogLikelihood(_x, _y);

    return loss;
} // double gp32_lik2(GP32* gp, double* x, double* y)


void gp32_get_params(GP32* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
} // void gp32_get_params(GP32* gp, double* params)


size_t gp32_igp_dim(GP32* gp)
{
    return gp->getIGPDim();
}


size_t gp32_num_param(GP32* gp)
{
    return gp->getNumParam();
}


size_t gp32_num_igp_param(GP32* gp)
{
    return gp->getNumIGPParam();
}


GP52* gp52_new(double dt, size_t num_output, size_t num_latent, double lambda, bool threading)
{
    return new GP52(dt, num_output, num_latent, lambda, threading);
} // GP52* gp52_new(double dt, size_t num_output, size_t num_latent, bool threading)


void gp52_del(GP52* gp)
{
    gp->~MOIHGP();
} // void gp52_del(GP52* gp)


void gp52_step1(GP52* gp, double* x, double* y, double* dx, double* xnew, double* yhat, double* dxnew)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dxnew(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    gp->step(_x, _y, _dx, _xnew, _yhat, _dxnew);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                dxnew[idx1 * igp_num_param * dim + idx2 * dim + idx3] = _dxnew[idx1][idx2](idx3);
            }
        }
    }
} // void gp52_step1(GP52* gp, double* x, double* y, double* dx, double* xnew, double* yhat, double* dxnew)


void gp52_step2(GP52* gp, double* x, double* y, double* dx, double* xnew, double* dxnew)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    std::vector<std::vector<Eigen::VectorXd>> _dxnew(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    gp->step(_x, _y, _dx, _xnew, _dxnew);

    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                dxnew[idx1 * igp_num_param * dim + idx2 * dim + idx3] = _dxnew[idx1][idx2](idx3);
            }
        }
    }
} // void gp52_step2(GP52* gp, double* x, double* y, double* dx, double* xnew, double* dxnew)


void gp52_step3(GP52* gp, double* x, double* y, double* xnew, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    gp->step(_x, _y, _xnew, _yhat);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
    }
} // void gp52_step3(GP52* gp, double* x, double* y, double* xnew, double* yhat)


void gp52_step4(GP52* gp, double* x, double* xnew, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    std::vector<Eigen::VectorXd> _xnew(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _yhat(num_output);

    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    gp->step(_x, _xnew, _yhat);

    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            xnew[idx1 * dim + idx2] = _xnew[idx1](idx2);
        }
    }
} // void gp52_step4(GP52* gp, double* x, double* xnew, double* yhat)


void gp52_update(GP52* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = Eigen::Map<Eigen::VectorXd>(params, num_param, 1);
    gp->update(_params);
} // void gp52_update(GP52* gp, double* params)


double gp52_lik1(GP52* gp, double* x, double* y, double* dx, double* grad)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t num_param = gp->getNumParam();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);
    std::vector<std::vector<Eigen::VectorXd>> _dx(num_latent, std::vector<Eigen::VectorXd>(igp_num_param, Eigen::VectorXd(dim).setZero()));
    Eigen::VectorXd _grad(num_param);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
        for (size_t idx2=0; idx2 < igp_num_param; idx2++)
        {
            for (size_t idx3=0; idx3 < dim; idx3++)
            {
                _dx[idx1][idx2](idx3) = dx[idx1 * igp_num_param * dim + idx2 * dim + idx3];
            }
        }
    }

    double loss = gp->negLogLikelihood(_x, _y, _dx, _grad);

    for (size_t idx=0; idx < num_param; idx++)
    {
        grad[idx] = _grad(idx);
    }

    return loss;
} // double gp52_lik1(GP52* gp, double* x, double* y, double* dx, Eigen::VectorXd &grad)


double gp52_lik2(GP52* gp, double* x, double* y)
{
    size_t num_output = gp->getNumOutput();
    size_t num_latent = gp->getNumLatent();
    size_t num_param = gp->getNumParam();
    size_t igp_num_param = gp->getNumIGPParam();
    size_t dim = gp->getIGPDim();

    std::vector<Eigen::VectorXd> _x(num_latent, Eigen::VectorXd(dim).setZero());
    Eigen::VectorXd _y(num_output);

    _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    for (size_t idx1=0; idx1 < num_latent; idx1++)
    {
        for (size_t idx2=0; idx2 < dim; idx2++)
        {
            _x[idx1](idx2) = x[idx1 * dim + idx2];
        }
    }

    double loss = gp->negLogLikelihood(_x, _y);

    return loss;
} // double gp52_lik2(GP52* gp, double* x, double* y)


void gp52_get_params(GP52* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
} // void gp52_get_params(GP52* gp, double* params)


size_t gp52_igp_dim(GP52* gp)
{
    return gp->getIGPDim();
}


size_t gp52_num_param(GP52* gp)
{
    return gp->getNumParam();
}


size_t gp52_num_igp_param(GP52* gp)
{
    return gp->getNumIGPParam();
}



} // extern "C"
