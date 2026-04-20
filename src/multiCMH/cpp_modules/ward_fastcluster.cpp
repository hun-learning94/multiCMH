#include "ward_fastcluster.hpp"
#include <Eigen/Dense> // For Eigen::MatrixXd and Eigen::VectorXd
#include <vector>
// #include <pybind11/pybind11.h>
// #include <pybind11/iostream.h>
using Eigen::MatrixXd;

// This is a helper function to calculate the condensed distance matrix.
// For Ward's method, we need the squared Euclidean distance.
std::vector<double> calculate_condensed_dist(const MatrixXd &X)
{
    int n = static_cast<int>(X.rows());
    int num_dists = n * (n - 1) / 2;
    std::vector<double> condensed_dist(num_dists);
    int k = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = i + 1; j < n; ++j)
        {
            condensed_dist[k++] = (X.row(i) - X.row(j)).squaredNorm();
        }
    }
    return condensed_dist;
}

Eigen::VectorXi Ward_fastcluster(
    const MatrixXd &data,
    int K)
{
    // pybind11::scoped_ostream_redirect stream(
    //     std::cout, pybind11::module_::import("sys").attr("stdout"));
    // pybind11::scoped_ostream_redirect error_stream(
    //     std::cerr, pybind11::module_::import("sys").attr("stderr"));

    int n = static_cast<int>(data.rows());

    // 2. Calculate the condensed distance matrix using squared Euclidean distance
    std::vector<double> distmat = calculate_condensed_dist(data);

    // 3. Allocate memory for the output
    std::vector<int> merge(2 * (n - 1));
    std::vector<double> height(n - 1);
    std::vector<int> labels(n);

    // 4. Call the fastcluster wrapper with Ward linkage
    // std::cout << "\nPerforming Ward linkage clustering..." << std::endl;
    // int result = fastcluster::hclust_fast(n, distmat.data(), fastcluster::HCLUST_METHOD_WARD, merge.data(), height.data());
    int result = hclust_fast(n, distmat.data(), HCLUST_METHOD_WARD, merge.data(), height.data());

    if (result != 0)
    {
        // std::cerr << "Error during clustering: invalid method." << std::endl;
        // return 1;
    }

    // 5. Cut the tree to get the final cluster labels
    // std::cout << "Cutting the dendrogram to form " << K << " clusters..." << std::endl;
    // fastcluster::cutree_k(n, merge.data(), K, labels.data());
    cutree_k(n, merge.data(), K, labels.data());

    // // 6. Print the results
    // std::cout << "\nFinal Cluster Labels:" << std::endl;
    // for (int i = 0; i < n; ++i)
    // {
    //     std::cout << "Data point " << i << ": Cluster " << labels[i] << std::endl;
    // }
    Eigen::VectorXi labels_out(n);
    for (int i = 0; i < n; ++i)
    {
        labels_out[i] = labels[i];
    }

    return labels_out;
}