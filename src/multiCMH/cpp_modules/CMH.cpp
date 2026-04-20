#include "CMH.hpp"

#include <cmath>    // std::sqrt (still useful for other places, or for Boost internal needs)
#include <iostream> // std::cerr
#include <limits>
#include <boost/math/distributions/chi_squared.hpp>
#include <Eigen/Dense>

double pchisq(double x, double df)
{
    if (df <= .0)
    {
        std::cerr << "DEBUG: pchisq df must be >= 1. Returning NaN" << std::endl;
        std::cout.flush();
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (std::isnan(x))
    {
        std::cerr << "DEBUG: pchisq x is NaN. Returning NaN" << std::endl;
        std::cout.flush();
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (x < .0)
        return .0;
    else
    {
        boost::math::chi_squared_distribution<> chi2(df);
        return boost::math::cdf(chi2, x);
    }
}

std::pair<double, double> CMHtest(const std::vector<Tbl22> &Tbl22T)
{
    if (Tbl22T.empty())
    {
        // std::cerr << "DEBUG: CMHtest empty strata. Returning NaN" << std::endl;
        // std::cout.flush();
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }

    double n00sum{.0}, mu0sum{.0}, nu0sum{.0};
    for (size_t i{0}; i < Tbl22T.size(); ++i)
    {
        n00sum += Tbl22T[i].n00;
        mu0sum += Tbl22T[i].mu0();
        nu0sum += Tbl22T[i].nu0();
    }
    double Mn = (n00sum - mu0sum) * (n00sum - mu0sum) / nu0sum;
    // std::cout << "DEBUG: ------ CMHtest Mn = " << Mn << std::endl;
    // std::cout.flush();
    if (std::isnan(Mn) || Mn < 0)
    {
        // std::cerr << "DEBUG: CMHtest std::isnan(Mn) || Mn < 0. Returning NaN" << std::endl;
        // std::cout.flush();
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    else
    {
        return {Mn, 1. - pchisq(Mn, 1)};
    }
}

std::pair<double, double> Robins1986(const std::vector<Tbl22> &Tbl22T)
{
    if (Tbl22T.empty())
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    // get common log odds ratio
    double A_mlt = 0.0;
    double A_summlt = 0.0;
    double B_mlt = 0.0;
    double B_summlt = 0.0;
    for (size_t t{0}; t < Tbl22T.size(); ++t)
    {
        /* conditional table (at strata t):
         * Y \ X |   0      1    |  tildZ = t
         * ----------------------------------
         * 1     |  n01t   n11t  |  n_1t
         * 0     |  n00t   n10t  |  n_0t
         *       |  n0_t   n1_t  |  n__t
         */
        double n__t = Tbl22T[t].n__();
        double At_sum = (Tbl22T[t].n01 + Tbl22T[t].n10) / n__t;
        double At_mlt = (Tbl22T[t].n01 * Tbl22T[t].n10) / n__t;
        double Bt_sum = (Tbl22T[t].n11 + Tbl22T[t].n00) / n__t;
        double Bt_mlt = (Tbl22T[t].n11 * Tbl22T[t].n00) / n__t;

        A_mlt += At_mlt;
        A_summlt += At_sum * At_mlt;
        B_mlt += Bt_mlt;
        B_summlt += Bt_sum * Bt_mlt;
    }

    double cm_lodd = -std::log(A_mlt / B_mlt);
    double cm_logg_sig = std::sqrt(
        0.5 * A_summlt / (A_mlt * A_mlt) +
        0.5 * B_summlt / (B_mlt * B_mlt) +
        0.5 * (A_summlt + B_summlt) / (A_mlt * B_mlt));

    return {cm_lodd, cm_logg_sig};
}