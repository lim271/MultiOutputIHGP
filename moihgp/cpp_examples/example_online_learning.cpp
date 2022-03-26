#include <cstdlib>
#include <list>
#include <Eigen/Core>
#include <moihgp/moihgp_online.h>
#include <moihgp/matern32ss.h>
#include <iostream>
#include <time.h>



int main()
{

    double dt = 0.1;
    size_t num_output = 2;
    size_t num_latent = 1;
    size_t windowsize = 1;
    double gamma = 0.9;
    double lambda = 0.0;
    bool threading = false;
    Eigen::MatrixXd H(2, 2);
    H << 0.7, 0.3, -0.3, 0.7;
    std::list<Eigen::VectorXd> data;
    double t = 0.0;
    while (t < 2 * M_PI)
    {
        Eigen::VectorXd x(num_latent);
        x << sin(t), sin(4*t);
        data.push_back(H * x + 0.1 * Eigen::VectorXd(num_output).setRandom());
        t += dt;
    }

    moihgp::MOIHGPOnlineLearning<moihgp::Matern32StateSpace> gp(dt, num_output, num_latent, gamma, lambda, windowsize, threading);
    std::list<Eigen::VectorXd> yhat;
    for (std::list<Eigen::VectorXd>::iterator y=data.begin(); y!=data.end(); y++)
    {
        clock_t tic = clock();
        yhat.push_back(gp.step(*y));
        clock_t toc = clock();
        std::cout << "Elapsed time per step:" << double(toc - tic) / 1000.0 << "ms" << std::endl;
    }

    return 0;
}
