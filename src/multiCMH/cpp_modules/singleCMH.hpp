#pragma once
#include <Eigen/Dense> // For Eigen::MatrixXd and Eigen::VectorXd
#include <tuple>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

// VectorXd vec_eCDF(const VectorXd &vec);
// VectorXd vec_rank(const VectorXd &vec);
// MatrixXd mat_trans(const MatrixXd &Z);
void vec_eCDF(VectorXd &vec);
void vec_rank(VectorXd &vec);
void mat_trans(MatrixXd &Z);

/**
 * @brief Compute the CMH test statistics on a window A and returns the p-value
 *
 * @param x VectorXd of x \in (0, 1]
 * @param y VectorXd of y \in (0, 1]
 * @param Z MatrixXd of Z \in (0, 1]^d (nobs x ndim), perhaps R^d after standardizing?
 * @param x_cut double, determine 2x2xT table
 * @param y_cut double, determine 2x2xT table
 * @param T int, determine 2x2xT table
 * @param scr_all int, screening rule for the total count of a window
 * @param scr_mrg int, screening rule for each margin of a window
 * @param stt_method int, ward is 1, medtree is 2
 * @param verbose bool, if true, calculate an esitmate of the common log odds ratio and its variance (see Agresti 2013, p.229)
 *
 * @return double, p-value
 */
std::tuple<double, double, double> singleCMH(
    const VectorXd &x,
    const VectorXd &y,
    const MatrixXd &Z,
    double x_cut,
    double y_cut,
    int T,
    int scr_all,
    int scr_mrg,
    int stt_method,
    bool verbose);

py::dict singleCMH_detail(
    const VectorXd &x,
    const VectorXd &y,
    const MatrixXd &Z_const,
    double x_cut,
    double y_cut,
    int T,
    int scr_all,
    int scr_mrg,
    int stt_method);