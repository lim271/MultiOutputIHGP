#ifndef _MOIHGP_ONLINE_H_
#define _MOIHGP_ONLINE_H_
#include <cstdlib>
#include <vector>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <LBFGSpp/LBFGSB.h>
#include <moihgp/moihgp.h>



namespace moihgp {

class Stage {
public:
    Stage() {}
    std::vector<Eigen::VectorXd> x;
    Eigen::VectorXd y;
    std::vector<std::vector<Eigen::VectorXd>> dx;
};

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
        _buffer.reserve(_num_data);
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
            Eigen::VectorXd yhat;
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
            _gp->step(x, y, dx, xnew, yhat, dxnew);
            x = xnew;
            dx = dxnew;
            loss += _gp->negLogLikelihood(x, y, dx, g);
            grad += g;
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
    std::vector<Stage> _buffer;

};

template <typename StateSpace>
class MOIHGPRegression {

public:

    MOIHGPRegression(const double& dt, const size_t& num_output, const size_t& num_latent, const size_t& num_data)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _num_data = num_data;
        _moihgp = new MOIHGP<StateSpace>(dt, num_output, num_latent);
        _num_param = _moihgp->getNumParam();
        _igp_num_param = _moihgp->getNumIGPParam();
        _lb = Eigen::VectorXd(_num_param).setConstant(-10.0);
        _ub = Eigen::VectorXd(_num_param).setConstant( 10.0);
        _lb.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(1e-4);
        _ub.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(10.0);
        _params = _moihgp->getParams();
        _LBFGSB_param.m = 5;
        _LBFGSB_param.max_iterations = 5;
        _LBFGSB_param.max_linesearch = 5;
        _LBFGSB_param.max_step = 1.0;
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

    Eigen::VectorXd getParams()
    {
        Eigen::VectorXd params = _moihgp->getParams();
        return params;
    }

private:
    MOIHGP<StateSpace>* _moihgp;
    double _dt;
    size_t _num_output;
    size_t _num_latent;
    size_t _num_data;
    size_t _num_param;
    size_t _igp_num_param;
    Eigen::VectorXd _params;
    LBFGSpp::LBFGSBParam<double> _LBFGSB_param;
    LBFGSpp::LBFGSBSolver<double>* _solver;
    Eigen::VectorXd _lb;
    Eigen::VectorXd _ub;
    Objective<StateSpace>* _obj;

};

}

#endif