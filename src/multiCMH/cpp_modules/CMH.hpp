#pragma once

#include <vector>
#include "Tbl22.hpp"

/**
 * @brief Computes the cumulative distribution function (CDF) of the chi-squared distribution.
 *
 * @param x  The value at which to evaluate the CDF.
 * @param df Degrees of freedom.
 *
 * @return The probability P(X ≤ x) for a chi-squared distribution with the given degrees of freedom.
 */
double pchisq(double x, double df);

/**
 * @brief Performs the Cochran–Mantel–Haenszel (CMH) test across a 2x2xT table.
 *
 * @param Tbl22T A vector of Tbl22 objects, each representing a 2x2 table for a stratum.
 *
 * @return A pair containing:
 *         - The CMH test statistic (double)
 *         - The corresponding p-value (double)
 */
std::pair<double, double> CMHtest(const std::vector<Tbl22> &Tbl22T);

/**
 * @brief Computes an estimate of the common log odds ratio and its variance for a 2x2xT table. See Agresti 2013 p.229 for details.
 *
 * @param Tbl22T A vector of Tbl22 objects, each representing a 2x2 table for a stratum.
 *
 * @return A pair containing:
 *         - log of the esimated common odds ratio, as suggested in Mantel and Haenszel 1959 (double)
 *         - estimated standard error of the log of the common odds ratio estimate, as proposed in Robins 1986 (double)
 */
std::pair<double, double> Robins1986(const std::vector<Tbl22> &Tbl22T);