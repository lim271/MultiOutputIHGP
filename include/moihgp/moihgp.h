#ifndef _MOIHGP_H_
#define _MOIHGP_H_
#include <cstdlib>
#include <vector>
#include <cmath>
#include <assert.h>
#include <pthread.h>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <pinv.h>
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
        _arg.gp->step(_arg.x, _arg.y, _arg.xnew, _arg.yhat);
    }
    else if (_arg.option==2)
    {
        _arg.gp->step(_arg.x, _arg.xnew, _arg.yhat);
    }
    else if (_arg.option==3)
    {
        _arg.loss = _arg.gp->negLogLikelihood(_arg.x, _arg.y, _arg.dx, _arg.grad);
    }
    else if (_arg.option==4)
    {
        _arg.loss = _arg.gp->negLogLikelihood(_arg.x, _arg.y);
    }
}

template <typename StateSpace>
class MOIHGP{

public:

    MOIHGP(const double& dt, const size_t& num_output, const size_t& num_latent)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _IGPs = std::vector<IHGP<StateSpace>*>(num_latent, nullptr);
        for (int idx=0; idx < num_latent; idx++)
        {
            _IGPs[idx] = new IHGP<StateSpace>(dt);
        }
        _dim = _IGPs[0]->getDim();
        _igp_nparam = _IGPs[0]->getNumParam();
        _nparam = (num_output + 1 + _igp_nparam) * num_latent + 1;
        dA.reserve(num_output * num_latent);
        int idx=0;
        for (int row=0; row < num_output; row++)
        {
            for (int col=0; col < num_latent; col++)
            {
                dA.push_back(Eigen::MatrixXd(num_output, num_latent).setZero());
                dA[idx](row, col) = 1.0;
                ++idx;
            }
        }
        Eigen::MatrixXd rand(num_output, num_latent);
        rand.setRandom();
        Eigen::MatrixXd I(num_output, num_latent);
        I.setIdentity();
        if (num_output * num_latent > 16)
        {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(I + 0.1 * rand, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        else
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(I + 0.1 * rand, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        S.setOnes(num_latent);
        sigma = 1e-1;
    }

    void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd& y, const std::vector<std::vector<Eigen::VectorXd>>& dx, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat, std::vector<std::vector<Eigen::VectorXd>>& dxnew)
    {
        std::vector<int> idx_observed;
        idx_observed.reserve(_num_output);
        for (int idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (int idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
            Ty = sqrtSinv * PseudoInverse(U0) * y_observed;
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        pthread_t *threads = new pthread_t[_num_latent];
        Args<IHGP<StateSpace>>* args = new Args<IHGP<StateSpace>>[_num_latent];
        for (int idx=0; idx < _num_latent; idx++)
        {
            args[idx].option = 0;
            args[idx].tid = idx;
            args[idx].gp = _IGPs[idx];
            args[idx].x = x[idx];
            args[idx].y = Ty(idx);
            args[idx].dx = dx[idx];
            args[idx].xnew = Eigen::VectorXd(_dim);
            args[idx].yhat = double(0.0);
            args[idx].dxnew = std::vector<Eigen::VectorXd>(_igp_nparam, Eigen::VectorXd(_dim));
            pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            void * ret;
            pthread_join(threads[idx], &ret);
        }
        Eigen::VectorXd Tyhat(_num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            xnew[idx] = args[idx].xnew;
            Tyhat(idx) = args[idx].yhat;
            dxnew[idx] = args[idx].dxnew;
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (int idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    }

    void step(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd &y, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd &yhat)
    {
        std::vector<int> idx_observed;
        for (int idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (int idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
            Ty = sqrtSinv * PseudoInverse(U0) * y_observed;
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        pthread_t *threads = new pthread_t[_num_latent];
        Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
        for (int idx=0; idx < _num_latent; idx++)
        {
            args[idx].option = 1;
            args[idx].tid = idx;
            args[idx].gp = _IGPs[idx];
            args[idx].x = x[idx];
            args[idx].y = Ty(idx);
            args[idx].xnew = Eigen::VectorXd(_dim);
            args[idx].yhat = double(0.0);
            pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            void* ret;
            pthread_join(threads[idx], &ret);
        }
        Eigen::VectorXd Tyhat(_num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            xnew[idx] = args[idx].xnew;
            Tyhat(idx) = args[idx].yhat;
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (int idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    }

    void step(std::vector<Eigen::VectorXd>& x, std::vector<Eigen::VectorXd>& xnew, Eigen::VectorXd& yhat)
    {
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        pthread_t *threads = new pthread_t[_num_latent];
        Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
        for (int idx=0; idx < _num_latent; idx++)
        {
            args[idx].option = 2;
            args[idx].tid = idx;
            args[idx].gp = _IGPs[idx];
            args[idx].x = x[idx];
            args[idx].xnew = std::vector<Eigen::VectorXd>(_igp_nparam, Eigen::VectorXd(_dim));
            args[idx].yhat = double(0.0);
            assert(!pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]));
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            void * ret;
            pthread_join(threads[idx], &ret);
        }
        Eigen::VectorXd Tyhat(_num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            xnew[idx] = args[idx].xnew;
            Tyhat(idx) = args[idx].yhat;
        }
        Eigen::MatrixXd sqrtS(_num_latent, _num_latent);
        sqrtS.setZero();
        for (int idx=0; idx < _num_latent; idx++) sqrtS(idx, idx) = sqrt(S(idx));
        yhat = U * sqrtS * Tyhat;
    }

    void update(const Eigen::VectorXd &params)
    {
        if (U.size() > 16)
        {
            Eigen::MatrixXd Uparam = params.head(_num_output * _num_latent);
            Uparam.resize(_num_output, _num_latent);
            Eigen::BDCSVD<Eigen::MatrixXd> svd(Uparam, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        else
        {
            Eigen::MatrixXd Uparam = params.head(_num_output * _num_latent);
            Uparam.resize(_num_output, _num_latent);
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Uparam, Eigen::ComputeThinU | Eigen::ComputeThinV);
            U = svd.matrixU() * svd.matrixV().transpose();
        }
        S = params.block(_num_output * _num_latent, 0, _num_latent, 1);
        sigma = params(_num_output * (_num_latent + 1));
        Eigen::MatrixXd igp_params = params.tail(_igp_nparam * _num_latent);
        igp_params.resize(_igp_nparam, _num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            Eigen::VectorXd tmp = igp_params.col(idx);
            _IGPs[idx]->update(tmp);
        }
    }

    double negLogLikelihood(const std::vector<Eigen::VectorXd>& x, const Eigen::VectorXd &y, const std::vector<std::vector<Eigen::VectorXd> >& dx, Eigen::VectorXd &grad)
    {
        std::vector<int> idx_observed;
        for (int idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (int idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
            Ty = sqrtSinv * PseudoInverse(U0) * y_observed;
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::MatrixXd I(_num_output, _num_output);
        I.setIdentity();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        double loss = 0.5 * log(S.sum()) + 0.5 * (_num_output - _num_latent) * log(sigma) + 0.5 / sigma * ((I - U * U.transpose()) * y).norm();
        Eigen::VectorXd pv(_num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            Eigen::MatrixXd HA = _IGPs[idx]->HA;
            Eigen::MatrixXd xi = x[idx];
            Eigen::MatrixXd HAx = HA * xi;
            double vi = y(idx) - HAx(0, 0);
            pv(idx) = vi * (1 - (_IGPs[idx]->HA * _IGPs[idx]->K)(0, 0)) / _IGPs[idx]->S;
        }
        Eigen::MatrixXd svdU;
        Eigen::MatrixXd svdV;
        Eigen::MatrixXd dAdT;
        if (U.size() > 16)
        {
            Eigen::BDCSVD<Eigen::MatrixXd> svd(
                U,
                Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            svdU = svd.matrixU();
            svdV = svd.matrixV();
        }
        else
        {
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                U,
                Eigen::ComputeThinU | Eigen::ComputeThinV
            );
            svdU = svd.matrixU();
            svdV = svd.matrixV();
        }
        U = svdU * svdV.transpose();
        for (int idx1=0; idx1 < _num_output * _num_latent; idx1++)
        {
            Eigen::MatrixXd dU = dA[idx1] - svdU * (svdU.transpose() * dA[idx1] * svdV).diagonal() * svdV.transpose();
            grad(idx1) = (-y.transpose() * U * dU.transpose() * y)(0, 0) / sigma;
            Eigen::MatrixXd dAdT = sqrtSinv * dU.transpose();
            for (int idx2=0; idx2 < _num_latent; idx2++)
            {
                grad(idx1) += (pv(idx2) * dAdT.row(idx2) * y)(0, 0);
            }
        }
        int sizeU = _num_output * _num_latent;
        for (int idx1=0; idx1 < _num_latent; idx1++) {
            grad(sizeU + idx1) = 0.5 / S(idx1);
            Eigen::MatrixXd dS(_num_latent, _num_latent);
            dS.setZero();
            dS(idx1, idx1) = 1.0;
            Eigen::MatrixXd dAdT = -sqrtSinv * sqrtSinv * sqrtSinv * dS * U.transpose() / 2.0;
            for (int idx2=0; idx2 < _num_latent; idx2++) {
                grad(sizeU + idx1) += (pv(idx2) * dAdT.row(idx2) * y)(0, 0);
            }
        }
        grad(sizeU + _num_latent) = 0.5 / sigma;
        pthread_t *threads = new pthread_t[_num_latent];
        Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
        for (int idx=0; idx < _num_latent; idx++)
        {
            args[idx].option = 3;
            args[idx].tid = idx;
            args[idx].gp = _IGPs[idx];
            args[idx].x = x[idx];
            args[idx].y = Ty(idx);
            args[idx].dx = dx[idx];
            args[idx].loss = double(0.0);
            args[idx].grad = Eigen::VectorXd(_igp_nparam);
            pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            void * ret;
            pthread_join(threads[idx], &ret);
        }
        Eigen::MatrixXd igp_grad(_igp_nparam, _num_latent);
        for (int idx=0; idx < _num_latent; idx++)
        {
            loss += args[idx].loss;
            igp_grad.col(idx) = args[idx].grad;
        }
        grad.tail(igp_grad.size()) = igp_grad;
    }

    double negLogLikelihood(Eigen::VectorXd *x, const Eigen::VectorXd &y)
    {
        std::vector<int> idx_observed;
        for (int idx=0; idx < _num_output; idx++)
        {
            if (!std::isnan(y(idx)))
            {
                idx_observed.push_back(idx);
            }
        }
        Eigen::MatrixXd sqrtSinv(_num_latent, _num_latent);
        sqrtSinv.setZero();
        for (int idx=0; idx < _num_latent; idx++)
        {
            sqrtSinv(idx, idx) = 1 / sqrt(S(idx));
        }
        Eigen::VectorXd Ty;
        size_t num_observed = idx_observed.size();
        if (num_observed != _num_output)
        {
            Eigen::VectorXd y_observed(num_observed);
            Eigen::MatrixXd U0(num_observed, _num_latent);
            for (int idx = 0; idx < num_observed; idx++)
            {
                U0.row(idx) = U.row(idx_observed[idx]);
                y_observed(idx) = y(idx_observed[idx]);
            }
            Ty = sqrtSinv * PseudoInverse(U0) * y_observed;
        }
        else
        {
            Ty = sqrtSinv * U.transpose() * y;
        }
        Eigen::MatrixXd I(_num_output, _num_output);
        I.setIdentity();
        double loss = 0.5 * log(S.sum()) + 0.5 * (_num_output - _num_latent) * log(sigma) + 0.5 / sigma * ((I - U * U.transpose()) * y).norm();
        pthread_t *threads = new pthread_t[_num_latent];
        Args<IHGP<StateSpace> > *args = new Args<IHGP<StateSpace> >[_num_latent];
        for (int idx=0; idx < _num_latent; idx++)
        {
            args[idx].option = 4;
            args[idx].tid = idx;
            args[idx].gp = _IGPs[idx];
            args[idx].x = x[idx];
            args[idx].y = Ty(idx);
            args[idx].loss = double(0.0);
            pthread_create(&threads[idx], 0, &worker<IHGP<StateSpace>>, (void*)&args[idx]);
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            void * ret;
            pthread_join(threads[idx], &ret);
        }
        for (int idx=0; idx < _num_latent; idx++)
        {
            loss += args[idx].loss;
        }
    }

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
        return _nparam;
    }

    size_t getNumIGPParam()
    {
        return _igp_nparam;
    }

    Eigen::VectorXd getParams()
    {
        Eigen::VectorXd params(_nparam);
        Eigen::MatrixXd igp_params(_dim, _igp_nparam);
        for (int idx=0; idx < _igp_nparam; idx++)
        {
            igp_params.col(idx) = _IGPs[idx]->getParams();
        }
        params << U, S, sigma, igp_params.resize(_igp_nparam, 1);
        return params;
    }

    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    std::vector<Eigen::MatrixXd> dA;
    double sigma;

private:

    std::vector<IHGP<StateSpace>*> _IGPs;
    double _dt;
    size_t _dim;
    size_t _num_output;
    size_t _num_latent;
    size_t _nparam;
    size_t _igp_nparam;

}; // class MOIHGP

} // namespace moihgp

#endif