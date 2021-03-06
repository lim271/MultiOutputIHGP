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
class OnlineObjective {

public:

    OnlineObjective(MOIHGP<StateSpace>* gp, const double& gamma, const size_t& windowsize)
    {
        _gp = gp;
        _dim = _gp->getIGPDim();
        _num_param = _gp->getNumParam();
        _igp_num_param = _gp->getNumIGPParam();
        _num_output = _gp->getNumOutput();
        _num_latent = _gp->getNumLatent();
        oldparams = _gp->getParams();
        _gamma = gamma;
        _windowsize = windowsize;
        _x = std::vector<Eigen::VectorXd>(_num_latent, Eigen::VectorXd(_dim).setZero());
        _dx = std::vector<std::vector<Eigen::VectorXd>>(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        ma.setZero(_num_output);
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
        std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        for (std::list<Eigen::VectorXd>::iterator it = Y.begin(); it != Y.end(); it++)
        {
            Eigen::VectorXd y = *it - ma;
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
        ma.setZero(_num_output);
        for (std::list<Eigen::VectorXd>::iterator it = Y.begin(); it != Y.end(); it++)
        {
            ma += *it;
        }
        ma /= double(Y.size());
        while (Y.size() > _windowsize)
        {
            std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
            std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
            Y.pop_front();
            _gp->step(_x, Y.front() - ma, _dx, xnew, dxnew);
            _x = xnew;
            _dx = dxnew;
        }
    }


    Eigen::VectorXd oldparams;
    LBFGSpp::BFGSMat<double, true> bfgs_mat;
    std::list<Eigen::VectorXd> Y;
    Eigen::VectorXd ma;

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
        _lb = Eigen::VectorXd(_num_param);
        _ub = Eigen::VectorXd(_num_param);
        _lb.head(_num_output * _num_latent).setConstant(-1e+4);
        _ub.head(_num_output * _num_latent).setConstant( 1e+4);
        _lb.segment(_num_output * _num_latent, _num_latent).setConstant(1e-4);
        _ub.segment(_num_output * _num_latent, _num_latent).setConstant(1e+4);
        _lb.tail(_igp_num_param * _num_latent + 1).setConstant(1e-4);
        _ub.tail(_igp_num_param * _num_latent + 1).setConstant(1e+2);
        x.assign(_num_latent, Eigen::VectorXd(_dim).setZero());
        dx.assign(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _gamma = gamma;
        if (windowsize < 1)
        {
            _windowsize = 1;
        }
        else
        {
            _windowsize = windowsize;
        }
        _params = _moihgp->getParams();
        _LBFGSB_param.m = 10;
        _LBFGSB_param.max_iterations = 5;
        _LBFGSB_param.max_linesearch = 20;
        _LBFGSB_param.max_step = 1e-1;
        _LBFGSB_param.ftol = 1e-8;
        _LBFGSB_param.epsilon = 1e-8;
        _LBFGSB_param.epsilon_rel = 1e-8;
        _solver = new LBFGSpp::LBFGSBSolver<double>(_LBFGSB_param);
        _obj = new OnlineObjective<StateSpace>(_moihgp, _gamma, _windowsize);
    }


    ~MOIHGPOnlineLearning()
    {
        delete _moihgp;
        delete _solver;
        delete _obj;
    }


    Eigen::VectorXd step(const Eigen::VectorXd& y) {
        Eigen::VectorXd yhat;
        std::vector<Eigen::VectorXd> xnew(_num_latent, Eigen::VectorXd(_dim).setZero());
        std::vector<std::vector<Eigen::VectorXd> > dxnew(_num_latent, std::vector<Eigen::VectorXd>(_igp_num_param, Eigen::VectorXd(_dim).setZero()));
        _obj->push_back(y);
        _moihgp->step(x, y - _obj->ma, xnew, yhat);
        yhat += _obj->ma;
        x = xnew;
        dx = dxnew;
        _obj->bfgs_mat = _solver->getBFGSMat();
        _obj->oldparams = _params;
        double fx;
        _solver->minimize(*_obj, _params, fx, _lb, _ub);
        return yhat;
    }


    Eigen::VectorXd getParams()
    {
        Eigen::VectorXd params = _moihgp->getParams();
        return params;
    }


    size_t getNumParam()
    {
        return _num_param;
    }


    size_t getNumOutput()
    {
        return _num_output;
    }


    size_t getNumLatent()
    {
        return _num_latent;
    }


    size_t getNumIGPParam()
    {
        return _igp_num_param;
    }


    size_t getIGPDim()
    {
        return _dim;
    }


    size_t getWindowsize()
    {
        return _windowsize;
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
    OnlineObjective<StateSpace>* _obj;

};

}



#endif