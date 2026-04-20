#pragma once
#include <string>
#include <cmath>     // For std::isnan
#include <algorithm> // For std::min
#include <limits>    // For std::numeric_limits, specifically has_quiet_NaN
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h> // to output Eigen matrix
namespace py = pybind11;

// template functions must be defined in the header file
template <typename TT>
inline TT nanproof_min(TT a, TT b)
{
    static_assert(std::numeric_limits<TT>::has_quiet_NaN, "Type must support NaN."); // check during compile time
    return std::isnan(a) ? b : (std::isnan(b) ? a : std::min(a, b));                 // nested ifs
}

/**
 * @brief Perform multis-scale CMH tests on numerous 2x2xT tables to test conditional independnece of X \perp Y \mid Z
 *
 * @param x VectorXd of x \in (0, 1], CDF transformed
 * @param y VectorXd of y \in (0, 1], CDF transformed
 * @param Z MatrixXd of Z \in (0, 1]^d (nobs x ndim), perhaps R^d after standardizing?
 * @param x_dsc bool, is x discrete?
 * @param y_dsc bool, is y discrete?
 * @param k1 int, maximum partition level of X, rule of thumb: floor(log2(n/scr_mrg))
 * @param k2 int, maximum partition level of Y, rule of thumb: floor(log2(n/scr_mrg))
 * @param alp double, user-determined significance level
 * @param sttm_n int, specify strata size (hence T) as a number of samples, T = ceil(n / sttm_n)
 * @param stt_method int, ward is 1, medtree is 2
 * @param scr_all int, screening rule for the total count of a window, set to 25
 * @param scr_mrg int, screening rule for each margin of a window, set to 10
 * @param maxT int, maximum number of stratifications per window. default 200
 * @param verbose bool, return all windows information if true, only overall corrected p-value if false
 *
 * @return py::dict containing:
 * - p_value, the overall corrected p-value
 * - k1
 * - k2
 * - x_transformed
 * - y_transformed
 * - Z_transformed
 * - windows
 */
py::dict multiCMH_cpp(
    VectorXd &x,
    VectorXd &y,
    MatrixXd &Z,
    bool x_dsc,
    bool y_dsc,
    int k1,
    int k2,
    double alp,
    int sttm_n,
    int stt_method,
    int scr_all,
    int scr_mrg,
    int maxT,
    bool verbose);