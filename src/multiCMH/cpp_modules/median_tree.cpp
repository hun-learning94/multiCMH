#include "median_tree.hpp"
#include <numeric>   // std::iota
#include <algorithm> // std::stable_sort
#include <stdexcept> // std::invalid_argument
#include <memory>    // Required for std::unique_ptr

Node::Node(const MatrixXd *pts, Index s, Index e, size_t d)
    : points(pts), start(s), end(e), depth(d) {}

Node::~Node()
{
    // recursively delete all nodes
    delete left_child;
    delete right_child;
}

Node *build_tree(const MatrixXd &points,
                 std::vector<Index> &indices,
                 Index start,
                 Index end,
                 size_t depth)
{
    Index n = end - start;
    Index d = points.cols();
    if (n <= 0)
        return nullptr;
    Node *node = new Node(&points, start, end, depth); // deal with a pointer to a node
    node->axis = depth % d;
    if (n == 1) // nothing to split, terminal
        return node;

    Index med_idx = start + n / 2; // median split
    // std::stable_sort(indices.begin() + start, // stable sort only the relevant range [start, end)
    //                  indices.begin() + end,
    //                  [&](Index i1, Index i2)
    //                  {
    //                      return points(i1, node->axis) < points(i2, node->axis);
    //                  });
    // stable_sort is O(nlogn), sorting the whole sequence, whereas nth_element is O(n).
    // "... rearranges the elements in a range such that the element at the n-th position
    // is the element that would be in that position in a sorted sequence. All elements before it are less than or equal to it."
    std::nth_element(indices.begin() + start,
                     indices.begin() + med_idx,
                     indices.begin() + end,
                     [&](Index i1, Index i2)
                     {
                         double v1 = points(i1, node->axis);
                         double v2 = points(i2, node->axis);
                         if (v1 < v2)
                             return true;
                         if (v2 < v1)
                             return false;
                         return i1 < i2; // tie-break by original index
                     });
    node->split_point = points(indices[med_idx], node->axis); // store split value (first in right half)

    node->left_child = build_tree(points, indices, start, med_idx, depth + 1); // recurse without copying
    node->right_child = build_tree(points, indices, med_idx, end, depth + 1);
    return node;
}

std::vector<Node *> partition_tree(Node *root_node, size_t K)
{
    if (K == 0)
        return {};
    if (K == 1)
        return {root_node};

    std::vector<Node *> nodes = {root_node}; // a set nodes that each form a cluster

    while (nodes.size() < K)
    {
        // find the largest node (cluster) from root to leafs
        size_t largest_idx = 0;
        Index largest_size = nodes[0]->end - nodes[0]->start;
        for (size_t i = 1; i < nodes.size(); ++i)
        {
            Index size = nodes[i]->end - nodes[i]->start;
            if (size > largest_size)
            {
                largest_size = size;
                largest_idx = i;
            }
        }

        // mark the largest to split
        Node *largest_node = nodes[largest_idx];
        if (!largest_node->left_child && !largest_node->right_child)
            break; // break if the largest is terminal

        // delete the largest node (-1) and children (+1 or +2)
        nodes.erase(nodes.begin() + largest_idx);
        if (largest_node->left_child)
            nodes.push_back(largest_node->left_child);
        if (largest_node->right_child)
            nodes.push_back(largest_node->right_child);
    }

    return nodes;
}

VectorXi MedTree_cluster(const MatrixXd &Z, size_t K)
{
    Index n = Z.rows();
    if (K > static_cast<size_t>(n))
    {
        throw std::invalid_argument("K cannot be greater than number of points");
    }

    std::vector<Index> indices(n);
    std::iota(indices.begin(), indices.end(), 0); // original indices
    // Node *root = build_tree(Z, indices, 0, n);           // build the tree
    std::unique_ptr<Node> root(build_tree(Z, indices, 0, n));  // memory is released when root goes out of scope, no delete needed
    std::vector<Node *> nodes = partition_tree(root.get(), K); // partition along the tree

    Eigen::VectorXi labels(n);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        Node *cluster = nodes[i];
        for (Index pos = cluster->start; pos < cluster->end; ++pos)
        {
            labels(indices[pos]) = static_cast<int>(i);
        }
    }

    // delete root;
    return labels;
}
