#ifndef _MOIHGP_H_
#define _MOIHGP_H_

#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <assert.h>
#include <pthread.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <moihgp/ihgp.h>



namespace moihgp {



template <typename GP>
class Args
{

public:

    Args() {}

    int option;
    int tid;
    GP* gp;
    Eigen::VectorXd x;
    double y;
    std::vector<Eigen::VectorXd> dx;
    Eigen::VectorXd xnew;
    double yhat;
    std::vector<Eigen::VectorXd> dxnew;
    double loss;
    Eigen::VectorXd grad;

};



template<typename GP>
void* worker(void* arg)
{
    Args<GP>& _arg = *(Args<GP>*)arg;
    if (_arg.option==0)
    {
        _arg.gp->step(_arg.x, _arg.y, _arg.dx, _arg.xnew, _arg.yhat, _arg.dxnew);
    }
    else if (_arg.option==1)
    {
        _arg.gp->step(_arg.x, _arg.y, _arg.dx, _arg.xnew, _arg.dxnew);
    }
    else if (_arg.option==2)
    {
        _arg.gp->step(_arg.x, _arg.y, _arg.xnew, _arg.yhat);
    }
    else if (_arg.option==3)
    {
        _arg.gp->step(_arg.x, _arg.xnew, _arg.yhat);
    }
    else if (_arg.option==4)
    {
        _arg.loss = _arg.gp->negLogLikelihood(_arg.x, _arg.y, _arg.dx, _arg.grad);
    }
    else if (_arg.option==5)
    {
        _arg.loss = _arg.gp->negLogLikelihood(_arg.x, _arg.y);
    }
}



template <typename StateSpace>
class MOIHGP{

public:

    MOIHGP(const double& dt, const size_t& num_output, const size_t& num_latent, const double& lambda, const bool& threading)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _lambda.setZero(1, num_latent);
        double reg_weight = lambda / double((num_latent - 1) * (num_latent - 1));
        for (size_t idx = 0; idx < _num_latent; idx++)
        {
            _lambda(0, idx) = double(idx * idx) * reg_weight;
        }
        _IGPs = std::vector<IHGP<StateSpace>*>(num_latent, nullptr);
        for (size_t idx=0; idx < num_latent; idx++)
        {
            _IGPs[idx] = new IHGP<StateSpace>(dt);
        }
        _dim = _IGPs[0]->getDim();
        _igp_num_param = _IGPs[0]->getNumParam();
        _num_param = num_output * num_latent + num_latent + 1 + num_latent * _igp_num_param;
        dA.reserve(num_output * num_latent);
        for (size_t row=0; row < num_output; row++)
        {
            for (size_t col=0; col < num_latent; col++)
            {
                dA.push_back(Eigen::MatrixXd(num_output, num_latent).setZero());
                dA.back()(row, col) = 1.0;
            }
        }
        Eigen::MatrixXd I(num_output, num_latent);
        I.setIdentity();
        std::random_device rd;
        std::mt19937 mersenne(rd());
        std::normal_distribution<> distr(0.0, 1e-3);
        Eigen::MatrixXd rand(num_output, num_latent);
        for (size_t row=0; row < num_output; row++)
        {
            for (size_t col=0; col < num_latent; col++)
            {
                rand(row, col) = distr(mersenne);
            }
        }
        if (num_output * num_latent > 16)
        {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(I + rand, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        else
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(I + rand, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        S.setOnes(num_latent);
        sigma = 1e-2;
        if (_num_latent < 2)
        {
            _threading = false;
        }
        else
        {
            _threading = threading;
        }
    } // constructor MOIHGP


    ~MOIHGP()
    {
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            delete _IGPs[idx];
        }
    }


    void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd>>& dx, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat, std::vector<std::vector<Eigen::VectorXd>>& dxnew)
    {
        std::vector<int> idx_observed;
        idx_observed.reserve(_num_output);
        for (size_t idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (size_t idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd U0T = U0.transpose();
            Ty = sqrtSinv * ((U0T * U0).ldlt().solve(U0T * y_observed));
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::VectorXd Tyhat(_num_latent);
        if (_threading)
        {
            pthread_t* threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace>>* args = new Args<IHGP<StateSpace>>[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 0;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].y = Ty(idx);
                args[idx].dx = dx[idx];
                args[idx].xnew = Eigen::VectorXd(_dim);
                args[idx].yhat = double(0.0);
                args[idx].dxnew = std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim));
                pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void * ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                xnew[idx] = args[idx].xnew;
                Tyhat(idx) = args[idx].yhat;
                dxnew[idx] = args[idx].dxnew;
            }
            delete [] threads;
            delete [] args;
        }
        else
        {
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                _IGPs[idx]->step(x[idx], Ty(idx), dx[idx], xnew[idx], Tyhat(idx), dxnew[idx]);
            }
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (size_t idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    } // void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd>>& dx, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat, std::vector<std::vector<Eigen::VectorXd>>& dxnew)


    void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd>>& dx, std::vector<Eigen::VectorXd>& xnew, std::vector<std::vector<Eigen::VectorXd>>& dxnew)
    {
        std::vector<int> idx_observed;
        idx_observed.reserve(_num_output);
        for (size_t idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (size_t idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd U0T = U0.transpose();
            Ty = sqrtSinv * ((U0T * U0).ldlt().solve(U0T * y_observed));
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        if (_threading)
        {
            pthread_t* threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace>>* args = new Args<IHGP<StateSpace>>[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 1;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].y = Ty(idx);
                args[idx].dx = dx[idx];
                args[idx].xnew = Eigen::VectorXd(_dim);
                args[idx].yhat = double(0.0);
                args[idx].dxnew = std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim));
                pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void * ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                xnew[idx] = args[idx].xnew;
                dxnew[idx] = args[idx].dxnew;
            }
            delete [] threads;
            delete [] args;
        }
        else
        {
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                _IGPs[idx]->step(x[idx], Ty(idx), dx[idx], xnew[idx], dxnew[idx]);
            }
        }
    } // void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd>>& dx, std::vector<Eigen::VectorXd>& xnew, std::vector<std::vector<Eigen::VectorXd>>& dxnew)


    void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat)
    {
        std::vector<int> idx_observed;
        for (size_t idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (size_t idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd U0T = U0.transpose();
            Ty = sqrtSinv * ((U0T * U0).ldlt().solve(U0T * y_observed));
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::VectorXd Tyhat(_num_latent);
        if (_threading)
        {
            pthread_t *threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 2;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].y = Ty(idx);
                args[idx].xnew = Eigen::VectorXd(_dim);
                args[idx].yhat = double(0.0);
                pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void* ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                xnew[idx] = args[idx].xnew;
                Tyhat(idx) = args[idx].yhat;
            }
            delete [] threads;
            delete [] args;
        }
        else
        {
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                _IGPs[idx]->step(x[idx], Ty(idx), xnew[idx], Tyhat(idx));
            }
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (size_t idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    } // void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat)


    void step(std::vector<Eigen::VectorXd>& x, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat)
    {
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Tyhat(_num_latent);
        if (_threading)
        {
            pthread_t* threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 3;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].xnew = Eigen::VectorXd(_dim);
                args[idx].yhat = double(0.0);
                assert(!pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]));
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void * ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                xnew[idx] = args[idx].xnew;
                Tyhat(idx) = args[idx].yhat;
            }
            delete [] threads;
            delete [] args;
        }
        else
        {
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                _IGPs[idx]->step(x[idx], xnew[idx], Tyhat(idx));
            }
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (size_t idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    } // void step(std::vector<Eigen::VectorXd>& x, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat)


    void update(const Eigen::VectorXd& params)
    {
        size_t sizeU = U.size();
        if (sizeU > 16)
        {
            Eigen::MatrixXd Uparam = params.head(_num_output * _num_latent);
            Uparam.resize(_num_latent, _num_output);
            Eigen::BDCSVD<Eigen::MatrixXd> svd(Uparam.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        else
        {
            Eigen::MatrixXd Uparam = params.head(_num_output * _num_latent);
            Uparam.resize(_num_latent, _num_output);
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Uparam.transpose(), Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        S = params.segment(sizeU, _num_latent);
        sigma = params(sizeU + _num_latent);
        Eigen::MatrixXd igp_params = params.tail(_igp_num_param * _num_latent);
        igp_params.resize(_igp_num_param, _num_latent);
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            Eigen::VectorXd tmp = igp_params.col(idx);
            _IGPs[idx]->update(tmp);
        }
    } // void update(const Eigen::VectorXd& params)


    double negLogLikelihood(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd> >& dx, Eigen::VectorXd& grad)
    {
        std::vector<int> idx_observed;
        idx_observed.reserve(_num_output);
        for (size_t idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        Eigen::MatrixXd sqrtSinv3(_num_latent, _num_latent);
        sqrtSinv3.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            double sqrtSi = sqrt(S(idx));
            sqrtSinv(idx, idx) = 1 / sqrtSi;
            sqrtSinv3(idx, idx) = 1 / sqrtSi / sqrtSi / sqrtSi;
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (size_t idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd U0T = U0.transpose();
            Ty = sqrtSinv * ((U0T * U0).ldlt().solve(U0T * y_observed));
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::MatrixXd I(_num_output, _num_output);
        I.setIdentity();
        double y_UUTy = ((I - U * U.transpose()) * y).norm();
        double m_n = std::max(double(_num_output - _num_latent), 0.0);
        double loss = 0.5 * log(S.sum()) + 0.5 * m_n * log(sigma) + 0.5 * y_UUTy / sigma + (_lambda * S)(0, 0);
        Eigen::VectorXd pv(_num_latent);
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            Eigen::MatrixXd HA = _IGPs[idx]->HA;
            Eigen::MatrixXd xi = x[idx];
            Eigen::MatrixXd HAx = HA * xi;
            double vi = y(idx) - HAx(0, 0);
            pv(idx) = vi * (1 - (_IGPs[idx]->HA * _IGPs[idx]->K)(0, 0)) / _IGPs[idx]->S(0, 0);
        }
        Eigen::MatrixXd svdU;
        Eigen::MatrixXd svdV;
        Eigen::VectorXd svdS;
        size_t sizeU = U.size();
        if (sizeU > 16)
        {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(
                U,
                Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            svdU = svd.matrixU();
            svdV = svd.matrixV();
            svdS = svd.singularValues();
        }
        else
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                U,
                Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            svdU = svd.matrixU();
            svdV = svd.matrixV();
            svdS = svd.singularValues();
        }
        grad.setZero();
        for (size_t idx1=0; idx1 < _num_output * _num_latent; idx1++)
        {
            Eigen::MatrixXd invS = (1.0 / svdS.array()).matrix().asDiagonal();
            Eigen::MatrixXd Io(_num_output, _num_output);
            Io.setIdentity();
            Eigen::MatrixXd Il(_num_latent, _num_latent);
            Il.setIdentity();
            Eigen::MatrixXd dU = (Io + svdU * (invS - Il) * svdU.transpose()) * dA[idx1] * (Il + svdV * (invS - Il) * svdV.transpose());
            grad(idx1) = (-y.transpose() * U * dU.transpose() * y)(0, 0) / sigma;
            Eigen::MatrixXd dAdT = sqrtSinv * dU.transpose();
            for (size_t idx2=0; idx2 < _num_latent; idx2++)
            {
                grad(idx1) += (pv(idx2) * dAdT.row(idx2) * y)(0, 0);
            }
        }
        Eigen::MatrixXd dS(_num_latent, _num_latent);
        for (size_t idx1=0; idx1 < _num_latent; idx1++) {
            grad(sizeU + idx1) = 0.5 / S(idx1) + _lambda(0, idx1);
            dS.setZero();
            dS(idx1, idx1) = 1.0;
            Eigen::MatrixXd dAdT = -0.5 * sqrtSinv3 * dS * U.transpose();
            for (size_t idx2=0; idx2 < _num_latent; idx2++) {
                grad(sizeU + idx1) += (pv(idx2) * dAdT.row(idx2) * y)(0, 0);
            }
        }
        grad(sizeU + _num_latent) = 0.5 * (m_n - y_UUTy / sigma) / sigma;
        Eigen::MatrixXd igp_grad(_igp_num_param, _num_latent);
        if (_threading)
        {
            pthread_t* threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 4;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].y = Ty(idx);
                args[idx].dx = dx[idx];
                args[idx].loss = double(0.0);
                args[idx].grad = Eigen::VectorXd(_igp_num_param);
                pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void * ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                loss += args[idx].loss;
                igp_grad.col(idx) = args[idx].grad;
                double dn = args[idx].grad(_igp_num_param - 1);
                grad(sizeU + idx) -= dn * sigma / S(idx) / S(idx);
                grad(sizeU + _num_latent) += dn / S(idx);
            }
            delete [] threads;
            delete [] args;
        }
        else{
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                Eigen::VectorXd g(_igp_num_param);
                _IGPs[idx]->negLogLikelihood(x[idx], Ty(idx), dx[idx], g);
                igp_grad.col(idx) = g;
                double dn = g(_igp_num_param - 1);
                grad(sizeU + idx) -= dn * sigma / S(idx) / S(idx);
                grad(sizeU + _num_latent) += dn / S(idx);
            }
        }
        igp_grad.resize(igp_grad.size(), 1);
        grad.tail(igp_grad.size()) = igp_grad;
        return loss;
    } // double negLogLikelihood(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd> >& dx, Eigen::VectorXd& grad)


    double negLogLikelihood(std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y)
    {
        std::vector<int> idx_observed;
        idx_observed.reserve(_num_output);
        for (size_t idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (size_t idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd U0T = U0.transpose();
            Ty = sqrtSinv * ((U0T * U0).ldlt().solve(U0T * y_observed));
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::MatrixXd I(_num_output, _num_output);
        I.setIdentity();
        double y_UUTy = ((I - U * U.transpose()) * y).norm();
        double m_n = std::max(double(_num_output -_num_latent), 0.0);
        double loss = 0.5 * log(S.sum()) + 0.5 * m_n * log(sigma) + 0.5 * y_UUTy / sigma + (_lambda * S)(0, 0);
        if (_threading)
        {
            pthread_t* threads = new pthread_t[_num_latent];
            Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                args[idx].option = 5;
                args[idx].tid = idx;
                args[idx].gp = _IGPs[idx];
                args[idx].x = x[idx];
                args[idx].y = Ty(idx);
                args[idx].loss = double(0.0);
                pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                void * ret;
                pthread_join(threads[idx], &ret);
            }
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                loss += args[idx].loss;
            }
            delete [] threads;
            delete [] args;
        }
        else
        {
            for (size_t idx=0; idx < _num_latent; idx++)
            {
                loss += _IGPs[idx]->negLogLikelihood(x[idx], Ty(idx));
            }
        }
        return loss;
    } // double negLogLikelihood(Eigen::VectorXd *x, const Eigen::VectorXd &y)


    size_t getIGPDim()
    {
        return _dim;
    }


    size_t getNumOutput()
    {
        return _num_output;
    }


    size_t getNumLatent()
    {
        return _num_latent;
    }


    size_t getNumParam()
    {
        return _num_param;
    }


    size_t getNumIGPParam()
    {
        return _igp_num_param;
    }


    Eigen::VectorXd getParams()
    {
        Eigen::VectorXd params(_num_param);
        Eigen::MatrixXd igp_params(_igp_num_param, _num_latent);
        for (size_t idx=0; idx < _num_latent; idx++)
        {
            igp_params.col(idx) = _IGPs[idx]->getParams();
        }
        igp_params.resize(_num_latent * _igp_num_param, 1);
        Eigen::MatrixXd Uparam = U.transpose();
        size_t sizeU = U.size();
        Uparam.resize(sizeU, 1);
        params.head(sizeU) = Uparam;
        params.segment(sizeU, _num_latent) = S;
        params(sizeU + _num_latent) = sigma;
        params.tail(_num_latent * _igp_num_param) = igp_params;
        return params;
    }


    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    std::vector<Eigen::MatrixXd> dA;
    double sigma;

private:

    std::vector<IHGP<StateSpace>*> _IGPs;
    bool _threading;
    double _dt;
    Eigen::MatrixXd _lambda;
    size_t _dim;
    size_t _num_output;
    size_t _num_latent;
    size_t _num_param;
    size_t _igp_num_param;

}; // class MOIHGP



} // namespace moihgp



#endif