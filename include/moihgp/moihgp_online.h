#ifndef _MOIHGP_ONLINE_H_
#define _MOIHGP_ONLINE_H_
#include <cstdlib>
#include <vector>
#include <list>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <LBFGSB.h>
#include <moihgp/moihgp.h>
#include <iostream>



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

    Objective(const size_t& nparam, const size_t& windowsize, const double& gamma, MOIHGP<StateSpace>* gp)
    {
        _gp = gp;
        _nparam = nparam;
        _gamma = gamma;
        _windowsize = windowsize;
    }

    double operator()(const Eigen::VectorXd& params, Eigen::VectorXd& grad)
    {
        Eigen::VectorXd dparams = params - oldparams;
        _gp->update(params);
        Eigen::VectorXd Bp(_nparam);
        if (bfgs_mat.m_m > 0)
        {
            bfgs_mat.apply_Hv(dparams, _gamma, Bp);    // p = gamma*inv(B)*grad
        }
        else
        {
            Bp = dparams;
        }
        double loss = 0.5 * (dparams.transpose() * Bp)(0, 0);
        grad = Bp;
        for (std::list<Stage>::iterator it = buffer.begin(); it != buffer.end(); it++)
        {
            Eigen::VectorXd g(_nparam);
            loss += _gp->negLogLikelihood(it->x, it->y, it->dx, g);
            grad += g;
        }
        return loss;
    }

    void push_back(Stage stage)
    {
        buffer.push_back(stage);
        if (buffer.size() > _windowsize)
        {
            buffer.pop_front();
        }
    }

    Eigen::VectorXd oldparams;
    LBFGSpp::BFGSMat<double, true> bfgs_mat;
    std::list<Stage> buffer;

private:

    size_t _nparam;
    size_t _windowsize;
    double _gamma;
    MOIHGP<StateSpace>* _gp;

};

template <typename StateSpace>
class MOIHGPOnlineLearning {

public:

    MOIHGPOnlineLearning(const double& dt, const size_t& num_output, const size_t& num_latent, const double& gamma, const size_t& windowsize)
    {
        init(dt, num_output, num_latent, gamma, windowsize);
    }

    void init(const double& dt, const size_t& num_output, const size_t& num_latent, const double& gamma, const size_t& windowsize)
    {
        _dt = dt;
        _num_output = num_output;
        _num_latent = num_latent;
        _moihgp = new MOIHGP<StateSpace>(dt, num_output, num_latent);
        _dim = _moihgp->getIGPDim();
        _igp_num_param = _moihgp->getNumIGPParam();
        _nparam = _moihgp->getNumParam();
        _lb = Eigen::VectorXd(_nparam).setConstant(-10.0);
        _ub = Eigen::VectorXd(_nparam).setConstant( 10.0);
        _lb.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(1e-4);
        _ub.tail(_igp_num_param * _num_latent + _num_latent + 1).setConstant(10.0);
        x.assign(_num_latent, Eigen::VectorXd(_dim).setZero());
        dx.assign(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _gamma = gamma;
        _windowsize = windowsize;
        _params = _moihgp->getParams();
        _solver = new LBFGSpp::LBFGSBSolver<double>(_LBFGSB_param);
        _obj = new Objective<StateSpace>(_nparam, _windowsize, _gamma, _moihgp);
    }

    Eigen::MatrixXd step(const Eigen::VectorXd &y) {
        Eigen::VectorXd yhat;
        std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _moihgp->step(x, y, dx, xnew, yhat, dxnew);
        Stage stage;
        stage.x = xnew;
        stage.y = y;
        stage.dx = dxnew;
        _obj->push_back(stage);
        x = xnew;
        dx = dxnew;
        _obj->bfgs_mat = _solver->m_bfgs;
        _obj->oldparams = _params;
        double fx;
        int niter = _solver->minimize(*_obj, _params, fx, _lb, _ub);
        return yhat;
    }

    std::vector<Eigen::VectorXd> x;
    std::vector<std::vector<Eigen::VectorXd>> dx;

private:
    MOIHGP<StateSpace>* _moihgp;
    double _dt;
    size_t _dim;
    size_t _num_output;
    size_t _num_latent;
    size_t _nparam;
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