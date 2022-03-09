#include <Eigen/Core>
#include <Eigen/SVD>

Eigen::MatrixXd PseudoInverse(Eigen::MatrixXd matrix)
{
    if (matrix.size() > 16)
    {
        Eigen::BDCSVD< Eigen::MatrixXd > svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
        double tolerance = 1.0e-12 * double(std::max(matrix.rows(), matrix.cols())) * svd.singularValues().array().abs()(0);
        return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    }
    else {
        Eigen::JacobiSVD< Eigen::MatrixXd > svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV );
        double tolerance = 1.0e-12 * double(std::max(matrix.rows(), matrix.cols())) * svd.singularValues().array().abs()(0);
        return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
    }
}