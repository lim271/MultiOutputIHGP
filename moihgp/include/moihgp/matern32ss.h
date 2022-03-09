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
        dF.assign(_nparam, Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.reserve(_nparam);
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setIdentity());
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setZero());
        dPinf.push_back(Eigen::MatrixXd(_dim, _dim).setZero());
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
        double lam = sqrt(3) / lengthscale;    // lambda = sqrt(3) / lengthscale
        double lam2 = lam * lam;    // lambda^2
        double len3 = 6.0 / (lengthscale * lengthscale * lengthscale);    // 6 / lengthscale^3
        F(1, 0) = -lam2;
        F(1, 1) = -2.0 * lam;
        Pinf(0, 0) = magnitude;
        Pinf(1, 1) = magnitude * lam2;
        R = params(2);

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

    size_t _dim = 2;
    size_t _nparam = 3;
    Eigen::VectorXd _params;

}; // class Matern32StateSpace

} // namespace moihgp

#endif