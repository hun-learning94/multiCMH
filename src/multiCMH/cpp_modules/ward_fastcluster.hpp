#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint> // for int_fast32_t

// Forward declarations for types and enums used in fastcluster_dm.cpp
typedef int_fast32_t t_index;
typedef double t_float;

enum hclust_fast_methods
{
    HCLUST_METHOD_SINGLE = 0,
    HCLUST_METHOD_COMPLETE = 1,
    HCLUST_METHOD_AVERAGE = 2,
    HCLUST_METHOD_MEDIAN = 3,
    HCLUST_METHOD_WARD = 4
};

// Function declaration for hclust_fast
// This is the main clustering function.
int hclust_fast(int n, double *distmat, int method, int *merge, double *height);

// Function declaration for cutree_k
// This function cuts the dendrogram into k clusters.
void cutree_k(int n, const int *merge, int nclust, int *labels);

// Declaration for a helper function to order nodes.
void order_nodes(const int N, const int *const merge, const t_index *const node_size, int *const order);

// Main wrapper function for Ward linkage using fastcluster
Eigen::VectorXi Ward_fastcluster(
    const Eigen::MatrixXd &data,
    int K);