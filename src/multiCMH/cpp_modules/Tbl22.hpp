#pragma once

// include headers (#include) in .hpp only if the declarations herein absolutely depend on those inclusions
#include <string>    // For std::string
#include <stdexcept> // For std::invalid_argument

/**
 * @brief Represents a 2x2 contingency table for a single stratum.
 *
 * The table layout is as follows:
 * ```
 *   X \ Y |   0     1   |
 *   ---------------------
 *     0   |  n00   n01  |  n0_
 *     1   |  n10   n11  |  n1_
 *         |  n_0   n_1  |  n__
 * ```
 */
struct Tbl22
{
    double n00; ///< Count for cell (0,0)
    double n01; ///< Count for cell (0,1)
    double n10; ///< Count for cell (1,0)
    double n11; ///< Count for cell (1,1)

    /**
     * @brief Constructs a 2x2 table with specified cell counts.
     * @param val_n00 Count for cell (0,0)
     * @param val_n01 Count for cell (0,1)
     * @param val_n10 Count for cell (1,0)
     * @param val_n11 Count for cell (1,1)
     */
    Tbl22(double val_n00, double val_n01, double val_n10, double val_n11);

    /// @brief Returns the total count for row 0.
    double n0_() const;

    /// @brief Returns the total count for row 1.
    double n1_() const;

    /// @brief Returns the total count for column 0.
    double n_0() const;

    /// @brief Returns the total count for column 1.
    double n_1() const;

    /// @brief Returns the grand total count.
    double n__() const;

    /// @brief Computes the expected mean under the null hypothesis.
    double mu0() const;

    /// @brief Computes the variance under the null hypothesis.
    double nu0() const;

    /// @brief Returns a string representation of the table (for Python display).
    std::string printTbl22() const;
};