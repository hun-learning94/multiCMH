#include "singleCMH.hpp"
#include "multiCMH.hpp"
// #include "invECDF.hpp"
#include <vector>
#include <algorithm> // For std::sort, std::min, std::transform, std::clamp
#include <unordered_set>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <limits>   // For std::numeric_limits
#include <string>   // For std::string, std::to_string
#include <iostream> // For std::cout, std::cerr, std::endl
#include <cmath>    // For std::ceil, std::pow
// #include <cassert>             // For assert
#include <Eigen/Dense>         // Explicitly include Eigen/Dense for Eigen types
#include <pybind11/pybind11.h> // For pybind11 functionalities
#include <pybind11/iostream.h> // For pybind11::scoped_ostream_redirect
#include <pybind11/eigen.h>    // to output Eigen matrix
namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;
using Eigen::VectorXd;
using Eigen::VectorXi;
using MaskArray = Eigen::Array<bool, Eigen::Dynamic, 1>;
constexpr double NaNd = std::numeric_limits<double>::quiet_NaN();

///////////////////////////////////////////////////////////////////////////////////////////////
// HELPERS ////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

/* check if x or y is binary */
inline bool is_binary(const VectorXd &vec)
{
    std::unordered_set<double> uniq_vec;
    for (size_t i{0}; i < vec.size(); ++i)
    {
        uniq_vec.insert(vec[i]);
        if (uniq_vec.size() > 2)
            return false;
    }
    return uniq_vec.size() == 2;
}

/* due to Eigen's quirkiness */
// to subset Eigen::VectorXd, MatrixXd using a boolean mask
inline VectorXd blsub_vec(const VectorXd &x, const MaskArray &mask)
{
    VectorXd x_yes(mask.count());
    int scanner = 0;
    for (int i = 0; i < x.size(); ++i)
    {
        if (mask(i))
            x_yes(scanner++) = x(i);
    }
    return x_yes;
}

inline MatrixXd blsub_mat(const MatrixXd &Z, const MaskArray &mask)
{
    MatrixXd Z_yes(mask.count(), Z.cols());
    int scanner = 0;
    for (int i = 0; i < Z.rows(); ++i)
    {
        if (mask(i))
            Z_yes.row(scanner++) = Z.row(i);
    }
    return Z_yes;
}

// get the minimum k s.t. each bin of level k+1 partition of (0,1] contains 0 or 1 unique values of discrete x
inline int dsc_getk(const VectorXd &x)
{
    double xmin = x.minCoeff();
    std::set<double> x_uniqs_set(x.data(), x.data() + x.size());
    VectorXd x_uniqs(x_uniqs_set.size());
    std::copy(x_uniqs_set.begin(), x_uniqs_set.end(), x_uniqs.data());
    int K = 10;
    for (int k = 1; k <= K; ++k)
    {
        int twok = 1 << k;
        VectorXd steps = Eigen::VectorXd::LinSpaced(twok + 1, 0, twok) / static_cast<double>(twok); // 0, 1, 2, 3, ..., 2^k
        int valid = true;
        for (int i = 0; i < (steps.size() - 1); ++i)
        {
            int cnt = (x_uniqs.array() > steps(i) && x_uniqs.array() <= steps(i + 1)).count();
            if (cnt > 1)
            {
                valid = false;
                break;
            }
        }
        if (valid)
        {
            return k;
        }
    }
    throw std::runtime_error("No suitable k found within the range [0, " + std::to_string(K) + "]");
}

/* related to number of stratum */
inline int getT(int nobs, int sttm_n)
{
    double nobsd = static_cast<double>(nobs);
    double sttm_nd = static_cast<double>(sttm_n);
    // if (sttm_how == "sttm_p")
    // {
    //     T = static_cast<int>(std::floor(std::pow(nobsd, 1.0 - sttm_p)));
    // }
    // else if (sttm_how == "sttm_n")
    // {
    //     T = static_cast<int>(std::floor(nobsd / sttm_nd));
    // }
    int T = static_cast<int>(std::floor(nobsd / sttm_nd));
    return T;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// MAIN ALGORITHM /////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
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
    bool verbose)
{
    pybind11::scoped_ostream_redirect stream(
        std::cout, pybind11::module_::import("sys").attr("stdout"));
    pybind11::scoped_ostream_redirect error_stream(
        std::cerr, pybind11::module_::import("sys").attr("stderr"));
    int nobs = Z.rows();
    int ndim = Z.cols();
    double sttm_p = 0.5;             // legacy code
    std::string sttm_how = "sttm_n"; // legacy code

    // PREPARATIONS ////////////////////////////////////////////////////////////////////////////////////////
    // 1. input validity checks
    // (a) do x, y, Z match in dimension?
    if (!(x.size() == nobs && y.size() == nobs))
        throw std::invalid_argument("dimensions of x, y, Z are incoherent");

    // (b) are the inputs sttm_how, sttm_p, sttm_n, stt_method valid?
    static const std::unordered_set<std::string> valid_opts = {"sttm_p", "sttm_n"};
    if (!valid_opts.count(sttm_how))
        throw std::invalid_argument("sttm_how must be sttm_p or sttm_n. sttm_how = " + sttm_how);
    if (sttm_how == "sttm_p" && (sttm_p <= 0 || sttm_p >= 1))
        throw std::invalid_argument("sttm_p must be in (0,1). sttm_p = " + std::to_string(sttm_p));
    if (sttm_how == "sttm_n" && (sttm_n <= 0 || sttm_n >= nobs))
        throw std::invalid_argument("sttm_n must be in (0, n). sttm_n = " + std::to_string(sttm_n));
    if (!(stt_method == 1 || stt_method == 2))
        throw std::invalid_argument("stt_method must be 1 (ward) or 2 (med tree)");

    // (c) are k1, k2, scr_all, scr_mrg positive?
    if (std::min(k1, k2) <= 0)
        throw std::invalid_argument("k1, k2 must be positive");
    if (std::min(scr_all, scr_mrg) <= 0)
        throw std::invalid_argument("scr_all, scr_mrg must be positive");

    // 2. data transforms
    // (a) x, y transform to be in (0,1]
    vec_eCDF(x);
    vec_eCDF(y);
    if (stt_method == 1) // transform only for ward. medtree only use ranks, so no need to transform
        mat_trans(Z);
    // std::cout << "DEBUG: x " << x.transpose() << "\n"
    //           << "DEBUG: y " << y.transpose() << std::endl;
    // std::cout.flush();

    // check: are transformed x, y in range (0,1]?
    bool allInRange = (x.minCoeff() > .0 && x.maxCoeff() <= 1.) && (y.minCoeff() > .0 && y.maxCoeff() <= 1.);
    if (!allInRange)
        throw std::invalid_argument("x, y must be in (0,1]");

    // std::cout << "DEBUG: x, y successfully transformed" << std::endl;
    // std::cout.flush();

    // 3. If X (and/or Y) is discrete, modify k1 (and/or k2) large enough so that each partition contain only one distinct categories
    // But is X (and/or Y) is binary, just make it 0.0 and 1 and set k = 1 for efficiency interpretability
    if (x_dsc)
    {
        if (is_binary(x))
        {
            k1 = 1;
            x = (x.array() == 1.0).select(x, 0.5);
        }
        else
        {
            k1 = dsc_getk(x);
        }
    }

    if (y_dsc)
    {
        if (is_binary(y))
        {
            k2 = 1;
            y = (y.array() == 1.0).select(y, 0.5);
        }
        else
        {
            k2 = dsc_getk(y);
        }
    }

    // COMPUTE AND SIDAK-CORRECT P-VALUES //////////////////////////////////////////////////////////////////////
    using std::vector;
    int W = k1 + k2 - 2;                                                // maximum resolution
    int W_valid = 0;                                                    // a total number of windows being scanned
    vector<vector<MatrixXd>> PSA(k1, vector<MatrixXd>(k2));             // uncorrected p-values for all windows
    vector<vector<MatrixXd>> PSA_cm_lodd(k1, vector<MatrixXd>(k2));     // estimates of common log odds ratios
    vector<vector<MatrixXd>> PSA_cm_lodd_sig(k1, vector<MatrixXd>(k2)); // estimates s.e. of common log odds ratios
    MatrixXd Pij = MatrixXd::Zero(k1, k2);                              // corrected p-values for partition (i,j)
    MatrixXd Lij = MatrixXd::Zero(k1, k2);                              // number of non-nan p-values for partition (i,j)
    VectorXd Pw = VectorXd::Zero(W + 1);                                // corrected p-values for resolutions w
    VectorXd Uw = VectorXd::Zero(W + 1);                                // number of non-nan p-values for resolution w
    double Pov = NaNd;                                                  // corrected p-value overall
    double pw_min = NaNd;                                               // minimum pw among all w = 0, 1, ..., W

    for (int w = 0; w <= W; ++w) // scan from low to high resolution w = 0, 1, ..., W
    {
        // std::cout << "DEBUG: --- w = " << w << ", k1 " << k1 << ", k2 " << k2 << " ---------------------------------------------------------" << std::endl;
        // std::cout.flush();

        double pij_min = NaNd;       // minimum among (i, j):i + j = w, L(i,j) > 0
        for (int j = 0; j < k2; ++j) // scan all partitions (i,j) of resolution w
        {
            int i = w - j;
            if (i < 0 || i >= k1)
                continue; // Ensure i is within bounds for PSA

            int twoi = 1 << i;
            int twoj = 1 << j;

            // std::cout << "DEBUG: ------ (i, j) = (" << i << ", " << j << ")" << std::endl;
            // std::cout.flush();
            PSA[i][j] = MatrixXd::Constant(twoi, twoj, NaNd);
            if (verbose)
            {
                PSA_cm_lodd[i][j] = MatrixXd::Constant(twoi, twoj, NaNd);
                PSA_cm_lodd_sig[i][j] = MatrixXd::Constant(twoi, twoj, NaNd);
            }

            double psa_min = NaNd;                                                                 // minimum among A \in \mcl{A}^{i,j}
            VectorXd x_steps = VectorXd::LinSpaced(twoi + 1, 0, twoi) / static_cast<double>(twoi); // {0, 1, 2, 3, ..., 2^i}/2^i, 2^i + 1 elements
            VectorXd y_steps = VectorXd::LinSpaced(twoj + 1, 0, twoj) / static_cast<double>(twoj); // {0, 1, 2, 3, ..., 2^j}/2^j, 2^j + 1 elements

            for (int ii = 1; ii < x_steps.size(); ++ii) // scan each window at partition (i,j)
            {
                for (int jj = 1; jj < y_steps.size(); ++jj)
                {
                    // ii = 1, 2, 3, ..., 2^i, jj = 1, 2, 3, ..., 2^j
                    // filter out x, y, z within the window
                    double xl = (ii == 1) ? 0.0 : x_steps(ii - 1);
                    double xr = x_steps(ii);
                    double yl = (jj == 1) ? 0.0 : y_steps(jj - 1);
                    double yr = y_steps(jj);
                    MaskArray x_mask = (x.array() > xl) && (x.array() <= xr);
                    MaskArray y_mask = (y.array() > yl) && (y.array() <= yr);
                    MaskArray xy_mask = x_mask && y_mask;
                    int nobs_A = xy_mask.count();
                    int TA = getT(nobs_A, sttm_n);
                    TA = std::min(TA, maxT);
                    // std::cout << "DEBUG: ------ nobsA " << nobs_A << ", TA " << TA << std::endl;
                    // std::cout.flush();
                    if (nobs_A <= scr_all || TA < 2)
                    {
                        // std::cout << "DEBUG: ------ PSA[" << i << "][" << j << "](" << ii - 1 << "," << jj - 1
                        //           << ") singleCMH screening triggered, p-value " << PSA[i][j](ii - 1, jj - 1) << std::endl;
                        // std::cout.flush();
                    }
                    else
                    {
                        double x_cut = 0.5 * (x_steps(ii) + x_steps(ii - 1));
                        double y_cut = 0.5 * (y_steps(jj) + y_steps(jj - 1));
                        // std::cout << "DEBUG: ------ xl = " << xl << ", xr = " << xr << ", x_cut = " << x_cut
                        //           << ", yl = " << yl << ", yr = " << yr << ", y_cut = " << y_cut << std::endl;
                        // std::cout.flush();
                        try
                        {
                            auto [pval, cm_lodd, cm_lodd_sig] = singleCMH(blsub_vec(x, xy_mask),
                                                                          blsub_vec(y, xy_mask),
                                                                          blsub_mat(Z, xy_mask),
                                                                          x_cut,
                                                                          y_cut,
                                                                          TA,
                                                                          scr_all,
                                                                          scr_mrg,
                                                                          stt_method,
                                                                          verbose);
                            PSA[i][j](ii - 1, jj - 1) = pval;
                            if (verbose)
                            {
                                PSA_cm_lodd[i][j](ii - 1, jj - 1) = cm_lodd;
                                PSA_cm_lodd_sig[i][j](ii - 1, jj - 1) = cm_lodd_sig;
                            }
                        }
                        catch (...)
                        {
                            // std::cout << "DEBUG: --- PSA[" << i << "][" << j << "](" << ii - 1 << "," << jj - 1
                            //           << ") singleCMH unknown error, p-value " << PSA[i][j](ii - 1, jj - 1) << std::endl;
                            // std::cout.flush();
                        }
                        psa_min = nanproof_min(psa_min, PSA[i][j](ii - 1, jj - 1));

                        if (!std::isnan(PSA[i][j](ii - 1, jj - 1)))
                        {
                            Lij(i, j)++;
                            // std::cout << "DEBUG: --------- PSA[" << i << "][" << j << "](" << ii - 1 << "," << jj - 1
                            //           << ") = " << PSA[i][j](ii - 1, jj - 1)
                            //           //   << ", psa_min " << psa_min
                            //           << std::endl;
                            // std::cout.flush();
                        }
                    }
                }
            }
            // Sidak's correction for partition (i,j)
            Pij(i, j) = (Lij(i, j) > 0) ? (1.0 - std::pow(1.0 - psa_min, Lij(i, j))) : NaNd;
            pij_min = nanproof_min(pij_min, Pij(i, j));
            Uw(w) += (Lij(i, j) > 0);
            // std::cout << "DEBUG: ------ Sidak's Pij(" << i << "," << j << ") = " << Pij(i, j)
            //           << ", L(" << i << "," << j << ") = " << Lij(i, j)
            //           //   << ", pij_min " << pij_min
            //           //   << ", Uw(" << w << ") " << Uw(w)
            //           << std::endl;
            // std::cout.flush();
        }
        // Sidak's correction for resolution w
        Pw(w) = (Uw(w) > 0) ? (1.0 - std::pow(1.0 - pij_min, Uw(w))) : NaNd;
        pw_min = nanproof_min(pw_min, Pw(w));
        W_valid += (Uw(w) > 0);
        // std::cout << "DEBUG: --- Sidak's Pw(" << w << ") = " << Pw(w)
        //           //   << ", pw_min " << pw_min
        //           //   << ", W_valid " << W_valid
        //           << ", U(" << w << ") = " << Uw(w)
        //           << std::endl;
        // std::cout.flush();
    }
    // Sidak's correction overall
    if (W_valid > 0)
    {
        Pov = 1.0 - std::pow(1.0 - pw_min, W_valid);
        // std::cout << "DEBUG: Oveall p-value = " << Pov << ", W_valid = " << W_valid
        //           << std::endl;
        // std::cout.flush();
    }
    else
        throw std::runtime_error("None of the windows yield a valid p-value");

    py::dict output;
    output["p_value"] = Pov;
    output["k1"] = k1;
    output["k2"] = k2;

    if (!verbose)
    {
        output["x_transformed"] = py::none();
        output["y_transformed"] = py::none();
        output["Z_transformed"] = py::none();
        output["windows"] = py::none();
        return (output);
    }

    // if verbose == true, return all the results including, for each window
    // i, j: a partition level
    // xl, xr, yl, yr: coordinates in eCDF scale
    // alpha: window-specific corrected significance level
    // pval: window-specific uncorrected p-value
    // cm_lodd: estimated common log odds ratio
    // cm_lodd_sig: estimated se of the common log odds ratio
    // and return a matrix of (num_wins x 10)
    size_t num_wins = 0; // the total number of windows in all (i, j) where there is at least one valid p-value
    MatrixXd windows(num_wins, 10);

    // COMPUTE SIDAK-CORRECT ALPHAS //////////////////////////////////////////////////////////////////////
    // alp(A) = 1 - (1-alp)^{1/(W+1) x 1/U(w) x 1/L(i,j)}

    for (int w = 0; w <= W_valid; ++w) // scan from low to high resolution w = 0, 1, ..., W
    {
        // retrieve U(w)
        double Uw_alp = Uw(w);
        if (Uw_alp == 0)
            continue;
        for (int j = 0; j < k2; ++j) // scan all partitions (i,j) of resolution w
        {
            int i = w - j;
            if (i < 0 || i >= k1)
                continue; // Ensure i is within bounds for PSA

            // retrieve L(i,j)
            double Lij_alp = Lij(i, j);
            if (Lij_alp == 0)
                continue;
            int twoi = 1 << i;
            int twoj = 1 << j;
            VectorXd x_steps = VectorXd::LinSpaced(twoi + 1, 0, twoi) / static_cast<double>(twoi); // {0, 1, 2, 3, ..., 2^i}/2^i, 2^i + 1 elements
            VectorXd y_steps = VectorXd::LinSpaced(twoj + 1, 0, twoj) / static_cast<double>(twoj); // {0, 1, 2, 3, ..., 2^j}/2^j, 2^j + 1 elements
            for (int ii = 1; ii < x_steps.size(); ++ii)                                            // scan each window at partition (i,j)
            {
                for (int jj = 1; jj < y_steps.size(); ++jj)
                {
                    // retrieve PSA[i][j](ii-1, jj-1) and compare with
                    // ALP[i][j](ii-1, jj-1) = 1 - (1-alp)^{1/W_valid x 1/U(w) x 1/L(i,j)}
                    // if significant (smaller), then store this
                    double pval_raw = PSA[i][j](ii - 1, jj - 1);
                    double alp_sidak = 1.0 - std::pow(1.0 - alp, 1.0 / (W_valid * Uw_alp * Lij_alp));
                    double xl = (ii == 1) ? 0.0 : x_steps(ii - 1);
                    double xr = x_steps(ii);
                    double yl = (jj == 1) ? 0.0 : y_steps(jj - 1);
                    double yr = y_steps(jj);
                    double cm_lodd = PSA_cm_lodd[i][j](ii - 1, jj - 1);
                    double cm_lodd_sig = PSA_cm_lodd_sig[i][j](ii - 1, jj - 1);
                    windows.conservativeResize(num_wins + 1, 10 + 3);
                    windows(num_wins, 0) = i;
                    windows(num_wins, 1) = j;
                    windows(num_wins, 2) = xl;
                    windows(num_wins, 3) = xr;
                    windows(num_wins, 4) = yl;
                    windows(num_wins, 5) = yr;
                    windows(num_wins, 6) = alp_sidak;
                    windows(num_wins, 7) = pval_raw;
                    windows(num_wins, 8) = cm_lodd;
                    windows(num_wins, 9) = cm_lodd_sig;
                    windows(num_wins, 10) = W_valid;
                    windows(num_wins, 11) = Uw_alp;
                    windows(num_wins, 12) = Lij_alp;
                    num_wins += 1;
                }
            }
        }
    }

    output["x_transformed"] = x;
    output["y_transformed"] = y;
    output["Z_transformed"] = Z;
    output["windows"] = windows;
    return (output);
}