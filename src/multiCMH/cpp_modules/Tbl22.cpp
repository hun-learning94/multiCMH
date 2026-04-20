#include "Tbl22.hpp"

#include <iomanip> // For std::fixed, std::setprecision, std::setw
#include <sstream> // For std::ostringstream for the __repr__ method

Tbl22::Tbl22(double val_n00, double val_n01, double val_n10, double val_n11)
    : n00(val_n00), n01(val_n01), n10(val_n10), n11(val_n11)
{
    if (n00 < 0 || n01 < 0 || n10 < 0 || n11 < 0)
    {
        throw std::invalid_argument("Counts in the table must be non-negative.");
    }
}

double Tbl22::n0_() const { return n00 + n01; }
double Tbl22::n1_() const { return n10 + n11; }
double Tbl22::n_0() const { return n00 + n10; }
double Tbl22::n_1() const { return n01 + n11; }
double Tbl22::n__() const { return n00 + n01 + n10 + n11; }

double Tbl22::mu0() const
{
    if (n__() == 0)
        return .0; // empty tbl22
    return (n0_() * n_0()) / n__();
}

double Tbl22::nu0() const
{
    if (n__() == 0 || n__() == 1)
        return .0; // empty tbl22
    return (n0_() * n1_() * n_0() * n_1()) / (n__() * n__() * (n__() - 1));
}

std::string Tbl22::printTbl22() const
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    const int label_width = 9;
    const int data_width = 9;

    oss << "Class Tbl22, a 2 X 2 table: \n";
    oss << std::setw(label_width) << "" << " |"
        << std::setw(data_width) << "Y = 0" << " |"
        << std::setw(data_width) << "Y = 1" << " |"
        << std::setw(data_width) << "Total" << "\n";
    oss << std::string(label_width + (data_width + 2) * 3, '-') << "\n";
    oss << std::setw(label_width) << "X = 0" << " |"
        << std::setw(data_width) << n00 << " |"
        << std::setw(data_width) << n01 << " |"
        << std::setw(data_width) << n0_() << "\n";
    oss << std::setw(label_width) << "X = 1" << " |"
        << std::setw(data_width) << n10 << " |"
        << std::setw(data_width) << n11 << " |"
        << std::setw(data_width) << n1_() << "\n";
    oss << std::string(label_width + (data_width + 2) * 3, '-') << "\n";
    oss << std::setw(label_width) << "Total" << " |"
        << std::setw(data_width) << n_0() << " |"
        << std::setw(data_width) << n_1() << " |"
        << std::setw(data_width) << n__() << "\n";
    oss << "mu0 " << mu0() << ", nu0 " << nu0();

    return oss.str();
}