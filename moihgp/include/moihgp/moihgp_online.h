#ifndef _MOIHGP_ONLINE_H_
#define _MOIHGP_ONLINE_H_

#include <cstdlib>
#include <vector>
#include <list>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <LBFGSpp/LBFGSB.h>
#include <moihgp/moihgp.h>



namespace moihgp {



template <typename StateSpace>
class Objective {

public:

    Objective(MOIHGP<StateSpace>* gp, const double& gamma, const size_t& windowsize)
    {
        _gp = gp;
        _dim = _gp->getIGPDim();
        _num_param = _gp->getNumParam();
        _igp_num_param = _gp->getNumIGPParam();
        _num_output = _gp->getNumOutput();
        _num_latent = _gp->getNumLatent();
        _gamma = gamma;
        _windowsize = windowsize;
        _x = std::vector<Eigen::VectorXd>(_num_latent, Eigen::VectorXd(_dim).setZero());
        _dx = std::vector<std::vector<Eigen::VectorXd>>(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
    }


    double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad)
    {
        Eigen::VectorXd dparams = params - oldparams;
        _gp->update(params);
        Eigen::VectorXd Bp(_num_param);
        if (bfgs_mat.get_m() > 0)
        {
            bfgs_mat.apply_Hv(dparams, _gamma, Bp);    // p = gamma*inv(B)*grad
        }
        else
        {
            Bp = dparams;
        }
        double loss = 0.5 * (dparams.transpose() * Bp)(0, 0);
        grad = Bp;
        std::vector<Eigen::VectorXd> x(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd>> dx(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        x = _x;
        dx = _dx;
        for (std::list<Eigen::VectorXd>::iterator it = Y.begin(); it != Y.end(); it++)
        {
            Eigen::VectorXd& y = *it;
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
            _gp->step(x, y, dx, xnew, dxnew);
            Eigen::VectorXd g(_num_param);
            loss += _gp->negLogLikelihood(x, y, dx, g);
            grad += g;
            x = xnew;
            dx = dxnew;
        }
        return loss;
    }


    void push_back(Eigen::MatrixXd y)
    {
        Y.push_back(y);
        while (Y.size() > _windowsize)
        {
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
            _gp->step(_x, *Y.begin(), _dx, xnew, dxnew);
            Y.pop_front();
            _x = xnew;
            _dx = dxnew;
        }
    }


    Eigen::VectorXd oldparams;
    LBFGSpp::BFGSMat<double, true> bfgs_mat;
    std::list<Eigen::VectorXd> Y;

private:

    size_t _num_output;
    size_t _num_latent;
    size_t _igp_num_param;
    size_t _num_param;
    size_t _dim;
    size_t _windowsize;
    double _gamma;
    MOIHGP<StateSpace>* _gp;
    std::vector<Eigen::VectorXd> _x;
    std::vector<std::vector<Eigen::VectorXd>> _dx;

};



template <typename StateSpace>
class MOIHGPOnlineLearning {

public:

    MOIHGPOnlineLearning(const double& dt, const size_t& num_output, const size_t& num_latent, const double& gamma, const size_t& windowsize, const bool& threading)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _threading = threading;
        _moihgp = new MOIHGP<StateSpace>(dt, num_output, num_latent, threading);
        _dim = _moihgp->getIGPDim();
        _igp_num_param = _moihgp->getNumIGPParam();
        _num_param = _moihgp->getNumParam();
        _lb = Eigen::VectorXd(_num_param).setConstant(-10.0);
        _ub = Eigen::VectorXd(_num_param).setConstant( 10.0);
        _lb.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(1e-4);
        _ub.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(10.0);
        x.assign(_num_latent, Eigen::VectorXd(_dim).setZero());
        dx.assign(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _gamma = gamma;
        _windowsize = windowsize;
        _params = _moihgp->getParams();
        _LBFGSB_param.m = 5;
        _LBFGSB_param.max_iterations = 5;
        _LBFGSB_param.max_linesearch = 5;
        _LBFGSB_param.max_step = 1.0;
        _solver = new LBFGSpp::LBFGSBSolver<double>(_LBFGSB_param);
        _obj = new Objective<StateSpace>(_moihgp, _gamma, _windowsize);
    }


    Eigen::MatrixXd step(const Eigen::VectorXd& y) {
        Eigen::VectorXd yhat;
        std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _moihgp->step(x, y, xnew, yhat);
        _obj->push_back(y);
        x = xnew;
        dx = dxnew;
        _obj->bfgs_mat = _solver->getBFGSMat();
        _obj->oldparams = _params;
        double fx;
        int niter = _solver->minimize(*_obj, _params, fx, _lb, _ub);
        return yhat;
    }


    std::vector<Eigen::VectorXd> x;
    std::vector<std::vector<Eigen::VectorXd>> dx;

private:

    MOIHGP<StateSpace>* _moihgp;
    bool _threading;
    double _dt;
    size_t _dim;
    size_t _num_output;
    size_t _num_latent;
    size_t _num_param;
    size_t _igp_num_param;
    size_t _windowsize;
    double _gamma;
    Eigen::VectorXd _params;
    LBFGSpp::LBFGSBParam<double> _LBFGSB_param;
    LBFGSpp::LBFGSBSolver<double>* _solver;
    Eigen::VectorXd _lb;
    Eigen::VectorXd _ub;
    Objective<StateSpace>* _obj;

};

}



#endif