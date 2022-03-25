#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <moihgp/moihgp_online.h>
#include <moihgp/moihgp_regression.h>
#include <moihgp/matern32ss.h>
#include <moihgp/matern52ss.h>



namespace moihgp {

extern "C" {

MOIHGPOnlineLearning<Matern32StateSpace>* online32_new(double dt, size_t num_output, size_t num_latent, double gamma, size_t windowsize, bool threading)
{
    return new MOIHGPOnlineLearning<Matern32StateSpace>(dt, num_output, num_latent, gamma, windowsize, threading);
}

void online32_del(MOIHGPOnlineLearning<Matern32StateSpace>* gp)
{
    gp->~MOIHGPOnlineLearning();
}

void online32_step(MOIHGPOnlineLearning<Matern32StateSpace>* gp, double* y, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    Eigen::VectorXd _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    Eigen::VectorXd _yhat = gp->step(_y);
    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
}

void online32_getParams(MOIHGPOnlineLearning<Matern32StateSpace>* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
}

size_t online32_getNumParam(MOIHGPOnlineLearning<Matern32StateSpace>* gp)
{
    size_t num_param = gp->getNumParam();
    return num_param;
}

MOIHGPOnlineLearning<Matern52StateSpace>* online52_new(double dt, size_t num_output, size_t num_latent, double gamma, size_t windowsize, bool threading)
{
    return new MOIHGPOnlineLearning<Matern52StateSpace>(dt, num_output, num_latent, gamma, windowsize, threading);
}

void online52_del(MOIHGPOnlineLearning<Matern52StateSpace>* gp)
{
    gp->~MOIHGPOnlineLearning();
}

void online52_step(MOIHGPOnlineLearning<Matern52StateSpace>* gp, double* y, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    Eigen::VectorXd _y = Eigen::Map<Eigen::VectorXd>(y, num_output, 1);
    Eigen::VectorXd _yhat = gp->step(_y);
    for (size_t idx=0; idx < num_output; idx++)
    {
        yhat[idx] = _yhat(idx);
    }
}

void online52_getParams(MOIHGPOnlineLearning<Matern52StateSpace>* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
}

size_t online52_getNumParam(MOIHGPOnlineLearning<Matern52StateSpace>* gp)
{
    size_t num_param = gp->getNumParam();
    return num_param;
}

MOIHGPRegression<Matern32StateSpace>* regression32_new(double dt, size_t num_output, size_t num_latent, size_t num_data, bool threading)
{
    return new MOIHGPRegression<Matern32StateSpace>(dt, num_output, num_latent, num_data, threading);
}

void regression32_del(MOIHGPRegression<Matern32StateSpace>* gp)
{
    gp->~MOIHGPRegression();
}

void regression32_fit(MOIHGPRegression<Matern32StateSpace>* gp, double* y, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_data = gp->getNumData();
    std::vector<Eigen::VectorXd> _y;
    _y.reserve(num_data);
    for (size_t idx=0; idx < num_data; idx++)
    {
        _y.push_back(Eigen::Map<Eigen::VectorXd>(&y[idx * num_output], num_output, 1));
    }
    gp->fit(_y);
}

void regression32_getParams(MOIHGPRegression<Matern32StateSpace>* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
}

size_t regression32_getNumParam(MOIHGPRegression<Matern32StateSpace>* gp)
{
    size_t num_param = gp->getNumParam();
    return num_param;
}

MOIHGPRegression<Matern52StateSpace>* regression52_new(double dt, size_t num_output, size_t num_latent, size_t num_data, bool threading)
{
    return new MOIHGPRegression<Matern52StateSpace>(dt, num_output, num_latent, num_data, threading);
}

void regression52_del(MOIHGPRegression<Matern52StateSpace>* gp)
{
    gp->~MOIHGPRegression();
}

void regression52_fit(MOIHGPRegression<Matern52StateSpace>* gp, double* y, double* yhat)
{
    size_t num_output = gp->getNumOutput();
    size_t num_data = gp->getNumData();
    std::vector<Eigen::VectorXd> _y;
    _y.reserve(num_data);
    for (size_t idx=0; idx < num_data; idx++)
    {
        _y.push_back(Eigen::Map<Eigen::VectorXd>(&y[idx * num_output], num_output, 1));
    }
    gp->fit(_y);
}

void regression52_getParams(MOIHGPRegression<Matern52StateSpace>* gp, double* params)
{
    size_t num_param = gp->getNumParam();
    Eigen::VectorXd _params = gp->getParams();
    for (size_t idx=0; idx < num_param; idx++)
    {
        params[idx] = _params(idx);
    }
}

size_t regression52_getNumParam(MOIHGPRegression<Matern52StateSpace>* gp)
{
    size_t num_param = gp->getNumParam();
    return num_param;
}

} // extern "C"

} // namespace moihgp
