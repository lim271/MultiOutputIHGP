#ifndef _MATERN52SS_H_
#define _MATERN52SS_H_

#include <cstdlib>
#include <vector>
#include <Eigen/Core>



namespace moihgp{


class Matern52StateSpace{

public:

    Matern52StateSpace()
    {
        F = Eigen::MatrixXd(_dim, _dim).setZero();
        F(0, 1) = 1.0;
        F(1, 2) = 1.0;
        Pinf = Eigen::MatrixXd(_dim, _dim).setZero();
        H = Eigen::MatrixXd(1, _dim);
        H << 1.0, 0.0, 0.0;
        R = Eigen::MatrixXd(1, 1).setZero();
        dF.assign(_num_param, Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.assign(_num_param, Eigen::MatrixXd(_dim, _dim).setZero());
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
        double lam = sqrt(3.0) / lengthscale;    // lambda = sqrt(3) / lengthscale
        double lam2 = lam * lam;    // lambda^2
        double len2 = lengthscale * lengthscale;
        double len3 = len2 * lengthscale;
        double len4 = len2 * len2;
        double kappa = 5.0 / 3.0 * magnitude / len2;
        double kappa2 = -2.0 * kappa / lengthscale;
        double sq5 = sqrt(5.0);
        F(2, 0) = -lam2 * lam;
        F(2, 1) = -3.0 * lam2;
        F(2, 2) = -3.0 * lam;
        Pinf(0, 0) = magnitude;
        Pinf(2, 2) = 25.0 * magnitude / len4;
        Pinf(1, 1) = kappa;
        Pinf(2, 0) = -kappa;
        Pinf(0, 2) = -kappa;
        R(0, 0) = params(2);

        // dF / d(lengthscale)
        dF[1](2, 0) = 15.0 * sq5 / len4;
        dF[1](2, 1) = 30.0 / len3;
        dF[1](2, 2) = sq5 * lam2;

        // dPinf / d(magnitude)
        dPinf[0] = Pinf / magnitude;

        // dPinf / d(lengthscale)
        dPinf[1](1, 1) = kappa2;
        dPinf[1](2, 0) = -kappa2;
        dPinf[1](0, 2) = -kappa2;
        dPinf[1](2, 2) = -100.0 * magnitude / len2 / len3;

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

    size_t _dim = 3;
    size_t _num_param = 3;
    Eigen::VectorXd _params;

}; // class Matern32StateSpace



} // namespace moihgp



#endif