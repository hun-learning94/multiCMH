#pragma once
#include <vector>
#include <Eigen/Dense>
using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXi;

// container for each node
struct Node
{
    const MatrixXd *points; // pointer to full dataset
    Index start;
    Index end;
    size_t depth;
    Node *left_child = nullptr;
    Node *right_child = nullptr;
    size_t axis = 0;
    double split_point = 0.0;

    // constructor
    Node(const MatrixXd *pts, Index s, Index e, size_t d);

    // destructor
    ~Node();
};

// builds the whole tree
// returns the pointer of the constructed tree
Node *build_tree(const MatrixXd &points,
                 std::vector<Index> &indices,
                 Index start, Index end, size_t depth = 0);

// given the tree already constructed by build_kdtree
// partition data into K clusters by splitting from large to small nodes
std::vector<Node *> partition_tree(Node *root_node, size_t K);

// input: Z matrix, K
// output: labels
VectorXi MedTree_cluster(const MatrixXd &Z, size_t K);