# MultiOutputIHGP
Implementation of the Multi-Output Infinite Horizon Gaussian Process (MOIHGP).

The code includes [LBFGS++](https://github.com/yixuan/LBFGSpp), which is a header-only c++ implementation of L-BFGS-B algorithm.

Compared to the original MOIHGP, the computation speed has improved by applying [Orthogonal Instantaneous Linear Mixing Model (OILMM)](https://github.com/wesselb/oilmm).

## Build
```
cd moihgp
mkdir build && cd build
cmake ..
make
```

## C++ Examples
- Regression example
```
cd moihgp/build
./example_regression
```
- Online learning example
```
cd moihgp/build
./example_online_learning
```

## Python3 Example
- Online learning example
```
python3 example.py
```

## Dependencies
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [SciPy](https://scipy.org/)

## Citation
J. Lim, J. Park, S. Nah, & J. Choi. (2021, May). Multi-output Infinite Horizon Gaussian Processes. In *2021 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 1542-1549). IEEE. (https://ieeexplore.ieee.org/document/9561031)
```
@inproceedings{lim2021multi,
  title={Multi-output Infinite Horizon Gaussian Processes},
  author={Lim, Jaehyun and Park, Jehyun and Nah, Sungjae and Choi, Jongeun},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1542--1549},
  year={2021},
  organization={IEEE}
}
```
