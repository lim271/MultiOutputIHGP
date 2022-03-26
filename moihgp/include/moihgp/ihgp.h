#ifndef _IHGP_H_
#define _IHGP_H_

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <utils/dare.h>



namespace moihgp{



template <typename StateSpace>
class IHGP{

public:

    IHGP(const double& dt)
    {
        _dt = dt;
        _ss = StateSpace();
        _num_param = _ss.getNumParam();
        _dim = _ss.getDim();
        dS.assign(_num_param, Eigen::MatrixXd(1, 1).setZero());
        dA.assign(_num_param, Eigen::MatrixXd(_dim, _dim).setZero());
        dK.assign(_num_param, Eigen::MatrixXd(_dim, 1).setZero());
        dAKHA.assign(_num_param, Eigen::MatrixXd(_dim, _dim).setZero());
        HdA.assign(_num_param, Eigen::MatrixXd(_dim, 1).setZero());
        update(_ss.getParams());
    }


    void step(const Eigen::VectorXd& x, const double& y, const std::vector<Eigen::VectorXd>& dx, Eigen::VectorXd& xnew, double& yhat, std::vector<Eigen::VectorXd>& dxnew)
    {
        if (std::isnan(y))
        {
            xnew = A * x;
            yhat = xnew(0, 0);
            for (size_t idx = 0; idx < _num_param; idx++)
            {
                dxnew[idx] = dA[idx] * x + A * dx[idx];
            }
        }
        else
        {
            xnew = AKHA * x + K * y;
            yhat = xnew(0, 0);
            for (size_t idx = 0; idx < _num_param; idx++)
            {
                dxnew[idx] = dAKHA[idx] * x + AKHA * dx[idx] + dK[idx] * y;
            }
        }
    }


    void step(const Eigen::VectorXd& x, const double& y, const std::vector<Eigen::VectorXd>& dx, Eigen::VectorXd& xnew, std::vector<Eigen::VectorXd>& dxnew)
    {
        if (std::isnan(y))
        {
            xnew = A * x;
            for (size_t idx = 0; idx < _num_param; idx++)
            {
                dxnew[idx] = dA[idx] * x + A * dx[idx];
            }
        }
        else
        {
            xnew = AKHA * x + K * y;
            for (size_t idx = 0; idx < _num_param; idx++)
            {
                dxnew[idx] = dAKHA[idx] * x + AKHA * dx[idx] + dK[idx] * y;
            }
        }
    }


    void step(const Eigen::VectorXd& x, const double& y, Eigen::VectorXd& xnew, double& yhat)
    {
        if (std::isnan(y))
        {
            xnew = A * x;
            yhat = xnew(0, 0);
        }
        else
        {
            xnew = AKHA * x + K * y;
            yhat = xnew(0, 0);
        }
    }


    void step(const Eigen::VectorXd& x, Eigen::VectorXd& xnew, double& yhat)
    {
        xnew = A * x;
        yhat = xnew(0, 0);
    }


    void backwardSmoother(const std::vector<Eigen::VectorXd>& X, std::vector<Eigen::VectorXd>& Xprev, Eigen::MatrixXd& P, Eigen::MatrixXd& G)
    {
        Eigen::MatrixXd PP = A * PF * A + Q;
        G = PP.ldlt().solve(A * PF).transpose();    // G = PF*A*inv(PP)
        DLyap(G, PF - G * PP * G.transpose(), P);    // dlyap(G)
        Xprev.push_back(X.back());
        for (std::vector<Eigen::VectorXd>::iterator x = X.end(); x != X.begin(); x--)
        {
            Xprev.push_back(*x + G * Xprev.back() - A * (*x));
        }
        std::reverse(Xprev.begin(), Xprev.end());
    }


    void update(const Eigen::VectorXd& params)
    {
        _ss.update(params);
        A = (_dt * _ss.F).exp();    // c2d
        Q = _ss.Pinf - A * _ss.Pinf * A.transpose();
        Q = (Q + Q.transpose()) / 2.0;
        Eigen::MatrixXd PP(_dim, _dim);
        Eigen::MatrixXd HT = _ss.H.transpose();
        DARE(A, HT, Q, _ss.R, PP);
        S = _ss.H * PP * HT + _ss.R;    // H*PP*H.T + R
        K = PP * HT / S(0, 0);    // PP*H.T*inv(S)
        PF = PP - K * _ss.H * PP;
        HA = _ss.H * A;
        AKHA = A - K * HA;    // A-K*H*A
        Eigen::MatrixXd AT = A.transpose();
        Eigen::MatrixXd AK = A * K;
        Eigen::MatrixXd AAKH = A - AK * _ss.H;
        Eigen::MatrixXd zeros(_dim, _dim);
        zeros.setZero();
        for (size_t idx = 0; idx < _num_param; idx++)
        {
            Eigen::MatrixXd dAT(_dim, _dim);
            Eigen::MatrixXd dQ(_dim, _dim);
            Eigen::MatrixXd QLyap(_dim, _dim);
            if (_ss.dF[idx]==zeros)
            {
                dA[idx].setZero();
                if (_ss.dPinf[idx]==zeros)
                {
                    dQ.setZero();
                }
                else
                {
                    dQ = _ss.dPinf[idx] - A * _ss.dPinf[idx] * AT;
                }
                if (_ss.dR[idx](0, 0)==0.0)
                {
                    QLyap = dQ;
                }
                else
                {
                    QLyap = AK * AK.transpose() * _ss.dR[idx] + dQ;
                }
            }
            else
            {
                Eigen::MatrixXd FF(2 * _dim, 2 * _dim);
                FF.topLeftCorner(_dim, _dim) = _ss.F;
                FF.bottomRightCorner(_dim, _dim) = _ss.dF[idx];
                dA[idx] = ((_dt * FF).exp()).bottomLeftCorner(_dim, _dim);
                dAT = dA[idx].transpose();
                if (_ss.dPinf[idx]==zeros)
                {
                    dQ = - dA[idx] * _ss.Pinf * AT - A * _ss.Pinf * dAT;
                }
                else
                {
                    dQ = _ss.dPinf[idx] - dA[idx] * _ss.Pinf * AT - A * _ss.dPinf[idx] * AT - A * _ss.Pinf * dAT;
                }
                if (_ss.dR[idx](0, 0)==0.0)
                {
                    QLyap = dA[idx] * PP * AT + A * PP * dAT - dA[idx] * PP * HT * AK.transpose() - AK * _ss.H * PP * dAT + dQ;
                }
                else
                {
                    QLyap = dA[idx] * PP * AT + A * PP * dAT - dA[idx] * PP * HT * AK.transpose() - AK * _ss.H * PP * dAT + AK * _ss.dR[idx] * AK.transpose() + dQ;
                }
            }
            Eigen::MatrixXd dPP(_dim, _dim);
            DLyap(AAKH, QLyap, dPP);
            dS[idx] = _ss.H * dPP * HT + _ss.dR[idx];
            dK[idx] = (dPP - PP * dS[idx](0, 0) / S(0, 0)) * HT / S(0, 0);
            if (_ss.dF[idx]==zeros)
            {
                dAKHA[idx] = -dK[idx] * _ss.H * A;
                HdA[idx].setZero();
            }
            else
            {
                dAKHA[idx] = dA[idx] -dK[idx] * _ss.H * A - K * _ss.H * dA[idx];
                HdA[idx] = (_ss.H * dA[idx]).transpose();
            }
        } // for (size_t idx = 0; idx < _num_param; idx++)
    } // void update(const Eigen::VectorXd& params)


    double negLogLikelihood(const Eigen::VectorXd& x, const double& y)
    {
        double v = y - (HA * x)(0, 0);
        double loss = 0.5 * (v * v / S(0, 0) + log(S(0, 0)));
        return loss;
    }


    double negLogLikelihood(const Eigen::VectorXd& x, const double& y, const std::vector<Eigen::VectorXd>& dx, Eigen::VectorXd& grad)
    {
        double v = y - (HA * x)(0, 0);
        double loss = 0.5 * (v * v / S(0, 0) + log(S(0, 0)));
        for (size_t idx = 0; idx < _num_param; idx++)
        {
            double dv = (-HdA[idx] * x - HA * dx[idx])(0, 0);
            grad[idx] = (v * dv - 0.5 * (v * v / S(0, 0) - 1) * dS[idx](0, 0)) / S(0, 0);
        }
        return loss;
    }


    Eigen::VectorXd getParams()
    {
        return _ss.getParams();
    }


    size_t getNumParam()
    {
        return _num_param;
    }


    size_t getDim()
    {
        return _dim;
    }


    Eigen::MatrixXd A;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd K;
    Eigen::MatrixXd S;
    Eigen::MatrixXd PF;
    Eigen::MatrixXd HA;
    Eigen::MatrixXd AKHA;
    std::vector<Eigen::MatrixXd> dS;
    std::vector<Eigen::MatrixXd> dA;
    std::vector<Eigen::MatrixXd> dK;
    std::vector<Eigen::MatrixXd> dAKHA;
    std::vector<Eigen::MatrixXd> HdA;

private:

    double _dt;
    size_t _num_param;
    size_t _dim;
    StateSpace _ss;

};



} // namespace moihgp



#endif