#ifndef _MATERN32SS_H_
#define _MATERN32SS_H_

#include <cstdlib>
#include <vector>
#include <Eigen/Core>



namespace moihgp{


class Matern32StateSpace{

public:

    Matern32StateSpace()
    {
        F = Eigen::MatrixXd(_dim, _dim).setZero();
        F(0, 1) = 1.0;
        Pinf = Eigen::MatrixXd(_dim, _dim).setZero();
        H = Eigen::MatrixXd(1, _dim);
        H << 1.0, 0.0;
        R = Eigen::MatrixXd(1, 1).setZero();
        dF.assign(_num_param, Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.reserve(_num_param);
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setIdentity());
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setZero());
        dR.reserve(_num_param);
        dR.push_back(Eigen::MatrixXd(1, 1).setZero());
        dR.push_back(Eigen::MatrixXd(1, 1).setZero());
        dR.push_back(Eigen::MatrixXd(1, 1).setOnes());
        Eigen::VectorXd params(_num_param);
        params << 1.0, 1.0, 0.1;
        update(params);
    }


    void update(const Eigen::VectorXd& params)
    {
        double magnitude = params(0);
        double lengthscale = params(1);
        double lam = sqrt(3) / lengthscale;    // lambda = sqrt(3) / lengthscale
        double lam2 = lam * lam;    // lambda^2
        double len3 = 6.0 / (lengthscale * lengthscale * lengthscale);    // 6 / lengthscale^3
        F(1, 0) = -lam2;
        F(1, 1) = -2.0 * lam;
        Pinf(0, 0) = magnitude;
        Pinf(1, 1) = magnitude * lam2;
        R(0, 0) = params(2);

        // dF / d(lengthscale)
        dF[1](1, 0) = len3;
        dF[1](1, 1) = 2.0 * lam / lengthscale;

        // dPinf / d(magnitude)
        dPinf[0](1, 1) = lam2;

        // dPinf / d(lengthscale)
        dPinf[1](1, 1) = -magnitude * len3;

        _params = params;
    }


    size_t getDim()
    {
        return _dim;
    }


    size_t getNumParam()
    {
        return _num_param;
    }


    Eigen::VectorXd getParams()
    {
        return _params;
    }


    Eigen::MatrixXd F;
    Eigen::MatrixXd Pinf;
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    std::vector<Eigen::MatrixXd> dF;
    std::vector<Eigen::MatrixXd> dPinf;
    std::vector<Eigen::MatrixXd> dR;

private:

    size_t _dim = 2;
    size_t _num_param = 3;
    Eigen::VectorXd _params;

}; // class Matern32StateSpace



} // namespace moihgp



#endif