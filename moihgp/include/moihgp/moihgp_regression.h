#ifndef _MOIHGP_ONLINE_H_
#define _MOIHGP_ONLINE_H_

#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <LBFGSpp/LBFGSB.h>
#include <moihgp/moihgp.h>



namespace moihgp {



template <typename StateSpace>
class Objective {

public:

    Objective(const size_t& num_data, MOIHGP<StateSpace>* gp)
    {
        _gp = gp;
        _dim = _gp->getIGPDim();
        _num_param = _gp->getNumParam();
        _igp_num_param = _gp->getNumIGPParam();
        _num_latent = _gp->getNumLatent();
        _num_data = num_data;
        Y.reserve(_num_data);
    }


    double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad)
    {
        double loss = 0.0;
        grad.setZero();
        std::vector<Eigen::VectorXd> x(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd> > dx(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        for (std::vector<Eigen::VectorXd>::iterator it = Y.begin(); it != Y.end(); it++)
        {
            Eigen::VectorXd& y = *it;
            Eigen::VectorXd g(_num_param);
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
            _gp->step(x, y, dx, xnew, dxnew);
            loss += _gp->negLogLikelihood(x, y, dx, g);
            grad += g;
            x = xnew;
            dx = dxnew;
        }
        return loss;
    }


    std::vector<Eigen::VectorXd> Y;

private:

    size_t _dim;
    size_t _num_param;
    size_t _igp_num_param;
    size_t _num_latent;
    size_t _num_data;
    double _gamma;
    MOIHGP<StateSpace>* _gp;

};



template <typename StateSpace>
class MOIHGPRegression {

public:

    MOIHGPRegression(const double& dt, const size_t& num_output, const size_t& num_latent, const size_t& num_data, const bool& threading)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _num_data = num_data;
        _threading = threading;
        _moihgp = new MOIHGP<StateSpace>(dt, num_output, num_latent, threading);
        _dim = _moihgp->getIGPDim();
        _num_param = _moihgp->getNumParam();
        _igp_num_param = _moihgp->getNumIGPParam();
        _lb = Eigen::VectorXd(_num_param).setConstant(-10.0);
        _ub = Eigen::VectorXd(_num_param).setConstant( 10.0);
        _lb.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(1e-4);
        _ub.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(10.0);
        _params = _moihgp->getParams();
        _solver = new LBFGSpp::LBFGSBSolver<double>(_LBFGSB_param);
        _obj = new Objective<StateSpace>(_num_data, _moihgp);
    }


    int fit(const std::vector<Eigen::VectorXd>& Y) {
        _obj->Y = Y;
        double fx;
        int niter = _solver->minimize(*_obj, _params, fx, _lb, _ub);
        _params = _moihgp->getParams();
        return niter;
    }


    std::vector<Eigen::VectorXd> predict(const std::vector<Eigen::VectorXd>& Y) {
        std::vector<Eigen::VectorXd> Yhat;
        Yhat.reserve(Y.size());
        std::vector<Eigen::VectorXd> x(_num_latent, Eigen::VectorXd(_dim).setZero());
        for (std::vector<Eigen::VectorXd>::iterator it=Y.begin(); it != Y.end(); it++) {
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            Eigen::VectorXd yhat(_num_output);
            _moihgp->step(x, *it, xnew, yhat);
            Yhat.push_back(yhat);
            x = xnew;
        }
        return Yhat;
    }


    Eigen::VectorXd getParams()
    {
        Eigen::VectorXd params = _moihgp->getParams();
        return params;
    }

private:

    MOIHGP<StateSpace>* _moihgp;
    double _threading;
    double _dt;
    size_t _num_output;
    size_t _num_latent;
    size_t _num_data;
    size_t _num_param;
    size_t _igp_num_param;
    size_t _dim;
    Eigen::VectorXd _params;
    LBFGSpp::LBFGSBParam<double> _LBFGSB_param;
    LBFGSpp::LBFGSBSolver<double>* _solver;
    Eigen::VectorXd _lb;
    Eigen::VectorXd _ub;
    Objective<StateSpace>* _obj;

};



}



#endif