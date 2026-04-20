#include <tuple>
#include "singleCMH.hpp"
#include "CMH.hpp"
#include "Tbl22.hpp"
#include "ward_fastcluster.hpp"
#include "median_tree.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Helper function to perform empirical CDF transformation on each column of a matrix.
// If a column of Z is discrete with ties, then we do a simple rank transform ignoring ties,
// so that [0,0, 1,1,1] is mapped to [0,0, 1,1,1] instead of [2/5, 2/5, 1,1,1].
// This way, we do not let the observed frequency of binary variables affect the (Euclidean) distance between two groups.

void vec_eCDF(Eigen::VectorXd &vec)
{
    int n = vec.size();
    if (n == 0)
        return;

    std::vector<std::pair<double, int>> indexed_vec(n);
    for (int i = 0; i < n; ++i)
    {
        indexed_vec[i] = {vec(i), i};
    }
    std::sort(indexed_vec.begin(), indexed_vec.end());

    Eigen::VectorXd ranked_vec(n);
    for (int i = 0; i < n; ++i)
    {
        int j = i;
        while (j < n && indexed_vec[j].first == indexed_vec[i].first)
        {
            j++;
        }
        for (int k = i; k < j; ++k)
        {
            ranked_vec(indexed_vec[k].second) =
                static_cast<double>(j) / static_cast<double>(n);
        }
        i = j - 1;
    }

    vec = ranked_vec; // overwrite input with eCDF values
}

void vec_rank(VectorXd vec)
{
    int n = vec.size();
    if (n == 0)
        return;

    std::vector<std::pair<double, int>> indexed_vec(n);
    for (int i = 0; i < n; ++i)
    {
        indexed_vec[i] = {vec(i), i};
    }
    std::sort(indexed_vec.begin(), indexed_vec.end());

    VectorXd ranks(n);
    double prev_val = indexed_vec[0].first;
    int rank = 0;
    ranks[indexed_vec[0].second] = rank;

    for (int i = 1; i < n; ++i)
    {
        double curr_val = indexed_vec[i].first;
        if (curr_val != prev_val)
        { // not ties
            ++rank;
            prev_val = curr_val;
        }
        ranks[indexed_vec[i].second] = rank; // if ties
    }

    vec = ranks / ranks.maxCoeff();
}

void mat_trans(MatrixXd &Z)
{
    int n = Z.rows();
    int p = Z.cols();
    for (int jj = 0; jj < p; ++jj)
    {
        std::vector<std::pair<double, int>> indexed_vec(n);
        bool has_ties = false;
        for (int i = 0; i < n; ++i)
            indexed_vec[i] = {Z(i, jj), i};
        std::sort(indexed_vec.begin(), indexed_vec.end());

        // check if Z.col(j) is discrete with ties or not
        for (int i = 1; i < n; ++i)
        {
            if (std::abs(indexed_vec[i].first - indexed_vec[i - 1].first) < 1e-9)
            {
                has_ties = true;
                break;
            }
        }

        if (has_ties) // rank transform ignoring ties
        {
            VectorXd ranks(n);
            double prev_val = indexed_vec[0].first;
            int rank = 0;
            ranks[indexed_vec[0].second] = rank;

            for (int i = 1; i < n; ++i)
            {
                double curr_val = indexed_vec[i].first;
                if (curr_val != prev_val)
                { // not ties
                    ++rank;
                    prev_val = curr_val;
                }
                ranks[indexed_vec[i].second] = rank; // if ties
            }

            Z.col(jj) = ranks / ranks.maxCoeff();
        }
        else // eCDF transform
        {
            VectorXd ranked_vec(n);
            for (int i = 0; i < n; ++i)
            {
                // for ties, find the end of the tied values
                int j = i;
                while (j < n && indexed_vec[j].first == indexed_vec[i].first)
                {
                    j++; // no ties: j=i+1, 2 ties: j=i+2, j will end at the last position of ties
                }
                for (int k = i; k < j; ++k)
                {
                    ranked_vec(indexed_vec[k].second) = static_cast<double>(j);
                }
                i = j - 1; // to make the next outer loop starts at j, -1 to offest ++i;
            }
            Z.col(jj) = ranked_vec / static_cast<double>(n);
        }
    }
}

std::tuple<double, double, double> singleCMH(
    const VectorXd &x,
    const VectorXd &y,
    const MatrixXd &Z_const,
    double x_cut,
    double y_cut,
    int T,
    int scr_all,
    int scr_mrg,
    int stt_method,
    bool verbose)
{
    // std::cout << "entered singleCMH " << std::endl;
    pybind11::scoped_ostream_redirect stream(
        std::cout, pybind11::module_::import("sys").attr("stdout"));
    pybind11::scoped_ostream_redirect error_stream(
        std::cerr, pybind11::module_::import("sys").attr("stderr"));

    // input validity check: x, y, x_cutoff, y_cutoff \in (0,1]
    bool allInRange = (x.minCoeff() > 0 && x.maxCoeff() <= 1) &&
                      (y.minCoeff() > 0 && y.maxCoeff() <= 1) &&
                      (x_cut > 0 && x_cut <= 1) &&
                      (y_cut > 0 && y_cut <= 1);
    if (!allInRange)
    {
        std::cerr << "Error: x, y and cutoff values must be in (0,1]. Returning NaN." << std::endl;
        return std::make_tuple(
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
    }

    // get the marginal table and check if any of the margins are empty
    /* marginal table (across all Z):
     * Y \ X |   0     1   |
     * ---------------------------
     * 1     |  n01   n11  |  n_1
     * 0     |  n00   n10  |  n_0
     *       |  n0_   n1_  |  n__
     */
    int n__ = x.size();
    int n0_ = (x.array() <= x_cut).count();
    int n1_ = n__ - n0_;
    int n_0 = (y.array() <= y_cut).count();
    int n_1 = n__ - n_0;

    // std::cout << "DEBUG: ------ n__:" << n__
    //           << ", n0_:" << n0_
    //           << ", n1_:" << n1_
    //           << ", n_0:" << n_0
    //           << ", n_1:" << n_1 << std::endl;

    // screening rules:
    // spit out NAN p-value (skip the test) if a window does not pass the screening rule
    // if n__ < scr_all or min(margins) < scr_mrg, then skip the CMH test
    if (n__ < scr_all || std::min({n_0, n_1, n0_, n1_}) < scr_mrg)
    {
        // std::cout << "DEBUG: singleCMH screening rule triggered. Returning NaN." << std::endl;
        // std::cout.flush();
        return std::make_tuple(
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
    }

    // 1) clustering to get tilZ
    // std::cout << "doing ward with T = " << T << std::endl;
    // do ecdf to Z for each window for more uniform clustering
    // MatrixXd Z = mat_trans(Z_const);
    Eigen::VectorXi tilZ;
    try
    {
        if (stt_method == 1)
        {
            tilZ = Ward_fastcluster(Z_const, T);
        }
        else if (stt_method == 2)
        {
            tilZ = MedTree_cluster(Z_const, T);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "DEBUG: singleCMH error during clustering: " << e.what() << ". Returning NaN." << std::endl;
        std::cout.flush();
        return std::make_tuple(
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
    }
    catch (...)
    {
        std::cout << "DEBUG: singleCMH unknown error during clustering" << std::endl;
        std::cout.flush();
        return std::make_tuple(
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
    }

    // 2) tabulate x, y, tilZ into std::vector<Tbl22> Tbl22T
    /* conditional table (at strata t):
     * Y \ X |   0      1    |  tildZ = t
     * ----------------------------------
     * 1     |  n01t   n11t  |  n_1t
     * 0     |  n00t   n10t  |  n_0t
     *       |  n0_t   n1_t  |  n__t
     */
    std::vector<Tbl22> Tbl22T;
    double n00t, n01t, n10t, n11t;
    for (int t = 0; t < T; ++t)
    {
        n00t = static_cast<double>(((x.array() <= x_cut) && (y.array() <= y_cut) && (tilZ.array() == t)).count());
        n01t = static_cast<double>(((x.array() <= x_cut) && (y.array() > y_cut) && (tilZ.array() == t)).count());
        n10t = static_cast<double>(((x.array() > x_cut) && (y.array() <= y_cut) && (tilZ.array() == t)).count());
        n11t = static_cast<double>(((x.array() > x_cut) && (y.array() > y_cut) && (tilZ.array() == t)).count());
        // Tbl22T.emplace_back(n01t, n11t, n00t, n10t);
        Tbl22T.emplace_back(n00t, n01t, n10t, n11t);
        // std::cout << "DEBUG: ------ t = " << t
        //           << " (" << n01t << ", " << n11t << ", " << n00t << ", " << n10t << "), sum "
        //           << n00t + n01t + n10t + n11t
        //           << std::endl;
        // std::cout.flush();
    }

    // 3) do CMH test to get p-value and return
    // std::cout << "doing CMH test with Tbl22T: " << std::endl;
    double pval = CMHtest(Tbl22T).second;

    // 4) add lines to do cm_lodd, cm_lodd_sig caculation
    double cm_lodd{std::numeric_limits<double>::quiet_NaN()};
    double cm_lodd_sig{std::numeric_limits<double>::quiet_NaN()};
    if (verbose)
        std::tie(cm_lodd, cm_lodd_sig) = Robins1986(Tbl22T);

    // std::cout << "single window p-value: " << pval << std::endl;
    return std::make_tuple(pval, cm_lodd, cm_lodd_sig);
}

py::dict singleCMH_detail(
    const VectorXd &x,
    const VectorXd &y,
    const MatrixXd &Z_const,
    double x_cut,
    double y_cut,
    int T,
    int scr_all,
    int scr_mrg,
    int stt_method)
{
    py::dict output;
    // std::cout << "entered singleCMH " << std::endl;
    // pybind11::scoped_ostream_redirect stream(
    //     std::cout, pybind11::module_::import("sys").attr("stdout"));
    // pybind11::scoped_ostream_redirect error_stream(
    //     std::cerr, pybind11::module_::import("sys").attr("stderr"));

    // input validity check: x, y, x_cutoff, y_cutoff \in (0,1]
    bool allInRange = (x.minCoeff() > 0 && x.maxCoeff() <= 1) &&
                      (y.minCoeff() > 0 && y.maxCoeff() <= 1) &&
                      (x_cut > 0 && x_cut <= 1) &&
                      (y_cut > 0 && y_cut <= 1);
    if (!allInRange)
    {
        std::cerr << "Error: x, y and cutoff values must be in (0,1]. Returning NaN." << std::endl;
        return output;
    }

    // get the marginal table and check if any of the margins are empty
    /* marginal table (across all Z):
     * Y \ X |   0     1   |
     * ---------------------------
     * 1     |  n01   n11  |  n_1
     * 0     |  n00   n10  |  n_0
     *       |  n0_   n1_  |  n__
     */
    int n__ = x.size();
    int n0_ = (x.array() <= x_cut).count();
    int n1_ = n__ - n0_;
    int n_0 = (y.array() <= y_cut).count();
    int n_1 = n__ - n_0;

    // std::cout << "DEBUG: ------ n__:" << n__
    //           << ", n0_:" << n0_
    //           << ", n1_:" << n1_
    //           << ", n_0:" << n_0
    //           << ", n_1:" << n_1 << std::endl;

    // screening rules:
    // spit out NAN p-value (skip the test) if a window does not pass the screening rule
    // if n__ < scr_all or min(margins) < scr_mrg, then skip the CMH test
    if (n__ < scr_all || std::min({n_0, n_1, n0_, n1_}) < scr_mrg)
    {
        // std::cout << "DEBUG: singleCMH screening rule triggered. Returning NaN." << std::endl;
        // std::cout.flush();
        return output;
    }

    // 1) clustering to get tilZ
    // std::cout << "doing ward with T = " << T << std::endl;
    // do ecdf to Z for each window for more uniform clustering
    // MatrixXd Z = mat_trans(Z_const);
    Eigen::VectorXi tilZ;
    try
    {
        if (stt_method == 1)
        {
            tilZ = Ward_fastcluster(Z_const, T);
        }
        else if (stt_method == 2)
        {
            tilZ = MedTree_cluster(Z_const, T);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "DEBUG: singleCMH error during clustering: " << e.what() << ". Returning NaN." << std::endl;
        std::cout.flush();
        return output;
    }
    catch (...)
    {
        std::cout << "DEBUG: singleCMH unknown error during clustering" << std::endl;
        std::cout.flush();
        return output;
    }

    // 2) tabulate x, y, tilZ into std::vector<Tbl22> Tbl22T
    /* conditional table (at strata t):
     * Y \ X |   0      1    |  tildZ = t
     * ----------------------------------
     * 1     |  n01t   n11t  |  n_1t
     * 0     |  n00t   n10t  |  n_0t
     *       |  n0_t   n1_t  |  n__t
     */
    // std::vector<Tbl22> Tbl22T;
    double n00t, n01t, n10t, n11t;
    Eigen::VectorXd slodds(T);
    Eigen::VectorXd sttm_size(T);
    Eigen::MatrixXd Z_mean(T, Z_const.cols());
    std::vector<Tbl22> Tbl22T;
    for (int t = 0; t < T; ++t)
    {
        // extract submatrix of Z_mean and compute column means
        std::vector<int> idx;
        for (int i = 0; i < tilZ.size(); ++i)
        {
            if (tilZ(i) == t)
                idx.push_back(i);
        }
        Eigen::MatrixXd Z_const_t(idx.size(), Z_const.cols());
        for (int i = 0; i < idx.size(); ++i)
        {
            Z_const_t.row(i) = Z_const.row(idx[i]);
        }
        Z_mean.row(t) = Z_const_t.colwise().mean();

        // compute sample log odds ratio = log (n00t+0.1)*(n11t+0.1)/((n10t+0.1)*(n01t+0.1))
        n00t = static_cast<double>(((x.array() <= x_cut) && (y.array() <= y_cut) && (tilZ.array() == t)).count());
        n01t = static_cast<double>(((x.array() <= x_cut) && (y.array() > y_cut) && (tilZ.array() == t)).count());
        n10t = static_cast<double>(((x.array() > x_cut) && (y.array() <= y_cut) && (tilZ.array() == t)).count());
        n11t = static_cast<double>(((x.array() > x_cut) && (y.array() > y_cut) && (tilZ.array() == t)).count());
        slodds(t) = std::log((n00t + 0.5) * (n11t + 0.5) / ((n10t + 0.5) * (n01t + 0.5)));
        sttm_size(t) = n00t + n01t + n10t + n11t;
        // Tbl22T.emplace_back(n01t, n11t, n00t, n10t);
        Tbl22T.emplace_back(n00t, n01t, n10t, n11t);
        // std::cout << "DEBUG: ------ t = " << t
        //           << " (" << n01t << ", " << n11t << ", " << n00t << ", " << n10t << "), sum "
        //           << n00t + n01t + n10t + n11t
        //           << std::endl;
        // std::cout.flush();
    }

    double cm_lodd{std::numeric_limits<double>::quiet_NaN()};
    double cm_lodd_sig{std::numeric_limits<double>::quiet_NaN()};
    std::tie(cm_lodd, cm_lodd_sig) = Robins1986(Tbl22T);

    // 3) do CMH test to get p-value and return
    // std::cout << "doing CMH test with Tbl22T: " << std::endl;
    // double pval = CMHtest(Tbl22T).second;

    // 4) add lines to do cm_lodd, cm_lodd_sig caculation
    // double cm_lodd{std::numeric_limits<double>::quiet_NaN()};
    // double cm_lodd_sig{std::numeric_limits<double>::quiet_NaN()};
    // if (verbose)
    // std::tie(cm_lodd, cm_lodd_sig) = Robins1986(Tbl22T);

    // std::cout << "single window p-value: " << pval << std::endl;
    output["sttm_size"] = sttm_size;
    output["slodds"] = slodds;
    output["Z_mean"] = Z_mean;
    output["cm_lodd"] = cm_lodd;
    return output;
}