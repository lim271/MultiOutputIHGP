#ifndef _DARE_H_
#define _DARE_H_
#include <Eigen/Core>



const double dare_tol = 1e-6;
const uint dare_maxiter = 100;

bool DARE(const Eigen::MatrixXd &Ad, const Eigen::MatrixXd &Bd, const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R, Eigen::MatrixXd &P)
{
    P = Q; // initialize

    Eigen::MatrixXd P_next;

    Eigen::MatrixXd AdT = Ad.transpose();
    Eigen::MatrixXd BdT = Bd.transpose();
    Eigen::MatrixXd Rinv = R.inverse();

    double diff;
    for (uint i = 0; i < dare_maxiter; ++i)
    {
        // -- discrete solver --
        P_next = AdT * P * Ad - AdT * P * Bd * (R + BdT * P * Bd).inverse() * BdT * P * Ad + Q;

        diff = fabs((P_next - P).maxCoeff());
        P = P_next;
        if (diff < dare_tol)
        {
          return true;
        }
    }
    return false; // over iteration limit
}


bool DLyap(const Eigen::MatrixXd &Ad, const Eigen::MatrixXd &Q, Eigen::MatrixXd &P)
{
    P = Q; // initialize
    
    Eigen::MatrixXd P_next;
    
    Eigen::MatrixXd AdT = Ad.transpose();
    
    double diff;
    for (uint i = 0; i < dare_maxiter; ++i)
    {
      // -- discrete solver --
      P_next = AdT * P * Ad - P + Q;
    
      diff = fabs((P_next - P).maxCoeff());
      P = P_next;
      if (diff < dare_tol)
      {
        return true;
      }
    }
    return false; // over iteration limit
}

#endif