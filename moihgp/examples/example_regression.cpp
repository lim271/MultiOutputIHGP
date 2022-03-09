#include <cstdlib>
#include <list>
#include <Eigen/Core>
#include <iostream>
#include <moihgp/moihgp_regression.h>
#include <moihgp/matern32ss.h>
#include <time.h>


int main()
{
    double dt = 0.1;
    double gamma = 0.9;
    size_t num_output = 2;
    size_t num_latent = 2;
    Eigen::MatrixXd H(num_output, num_latent);
    H << 0.7, 0.3, -0.3, 0.7;
    std::vector<Eigen::VectorXd> data;
    double t = 0.0;
    while (t < 2 * M_PI)
    {
        Eigen::VectorXd x(num_latent);
        x << sin(t), sin(4*t);
        data.push_back(H * x + 0.1 * Eigen::VectorXd(num_output).setRandom());
        t += dt;
    }
    data.shrink_to_fit();
    size_t num_data = data.size();

    moihgp::MOIHGPRegression<moihgp::Matern32StateSpace> gp(dt, num_output, num_latent, num_data);
    clock_t tic = clock();
    int niter = gp.fit(data);
    clock_t toc = clock();

    std::cout << "Iteration count: " << niter << std::endl;
    std::cout << "Elapsed time: " << double(toc - tic) / 1000.0 << "ms" << std::endl;

    return 0;
}