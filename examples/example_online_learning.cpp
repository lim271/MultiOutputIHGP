#include <cstdlib>
#include <list>
#include <Eigen/Core>
#include <iostream>
#include <moihgp/moihgp_online.h>
#include <moihgp/matern32ss.h>
#include <time.h>


int main()
{
    double dt = 0.1;
    double gamma = 0.9;
    size_t num_output = 2;
    size_t num_latent = 2;
    size_t windowsize = 3;
    Eigen::MatrixXd H(num_output, num_latent);
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
    moihgp::MOIHGPOnlineLearning<moihgp::Matern32StateSpace> gp(dt, num_output, num_latent, gamma, windowsize);
    std::list<Eigen::VectorXd> yhat;
    for (std::list<Eigen::VectorXd>::iterator y=data.begin(); y!=data.end(); y++)
    {
        clock_t tic = clock();
        yhat.push_back(gp.step(*y));
        clock_t toc = clock();
        std::cout << "Elapsed time:" << toc - tic << std::endl;
    }
    return 0;
}
