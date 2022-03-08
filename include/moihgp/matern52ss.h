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
        dF.assign(_nparam, Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.assign(_nparam, Eigen::MatrixXd(_dim, _dim).setZero());
        dR.reserve(_nparam);
        dR.push_back(0.0);
        dR.push_back(0.0);
        dR.push_back(1.0);
        Eigen::VectorXd params(_nparam);
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
        R = params(2);

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
        return _dim;
    }

    Eigen::VectorXd getParams()
    {
        return _params;
    }

    Eigen::MatrixXd F;
    Eigen::MatrixXd Pinf;
    Eigen::MatrixXd H;
    double R;
    std::vector<Eigen::MatrixXd> dF;
    std::vector<Eigen::MatrixXd> dPinf;
    std::vector<double> dR;

private:

    size_t _dim = 3;
    size_t _nparam = 3;
    Eigen::VectorXd _params;

}; // class Matern32StateSpace

} // namespace moihgp

#endif