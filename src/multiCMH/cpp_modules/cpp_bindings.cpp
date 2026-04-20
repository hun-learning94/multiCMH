#include "Tbl22.hpp"
#include "ward_fastcluster.hpp"
#include "median_tree.hpp"
#include "CMH.hpp"
#include "singleCMH.hpp"
#include "multiCMH.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // For converting std::vector and std::pair
#include <pybind11/eigen.h>    // For converting Eigen types
#include <pybind11/iostream.h> // REQUIRED FOR std::cout/cerr REDIRECTION TO PYTHON

namespace py = pybind11;

// --- Pybind11 Module Definition ---
// This macro creates the Python module.
// The first argument is the name you'll use to import it in Python.
PYBIND11_MODULE(multiCMH_cpp_module, m)
{
      m.doc() = "C++ module for multiCMH test";
      // py::add_ostream_redirect(m);

      //     // Bind CMH
      //     m.def("pchisq", &pchisq, "cdf of chi2 distribution", py::arg("x"), py::arg("df") = 1);
      //     m.def("CMHtest", &CMHtest, "do CMH test");

      //     // Bind Tbl22 struct. pybind11::class_<c++ name>(m, "python name")
      //     py::class_<Tbl22>(m, "Tbl22")
      //         .def(pybind11::init<double, double, double, double>(),
      //              pybind11::arg("n00"),
      //              pybind11::arg("n01"),
      //              pybind11::arg("n10"),
      //              pybind11::arg("n11"))
      //         .def_readwrite("n00", &Tbl22::n00)
      //         .def_readwrite("n01", &Tbl22::n01)
      //         .def_readwrite("n10", &Tbl22::n10)
      //         .def_readwrite("n11", &Tbl22::n11)
      //         .def("n0_", &Tbl22::n0_)
      //         .def("n1_", &Tbl22::n1_)
      //         .def("n_0", &Tbl22::n_0)
      //         .def("n_1", &Tbl22::n_1)
      //         .def("mu0", &Tbl22::mu0)
      //         .def("nu0", &Tbl22::nu0)
      //         .def("__repr__", &Tbl22::printTbl22);

      // Bind ward_fastcluster
      m.def("Ward_fastcluster",
            &Ward_fastcluster,
            "Ward linkage agglomerative clustering (using David Mullner's fastcluster)",
            py::arg("X"),
            py::arg("K"));

      // Bind median_tree
      m.def("MedTree_cluster",
            &MedTree_cluster,
            "Median tree (KD tree) clustering using forced median split for equal number of counts",
            py::arg("Z"),
            py::arg("K"));

      // Bind singleCMH_detail
      m.def("singleCMH_detail",
            &singleCMH_detail,
            "Return statum-specific information for a chosen window",
            py::arg("x"),
            py::arg("y"),
            py::arg("Z_const"),
            py::arg("x_cut"),
            py::arg("y_cut"),
            py::arg("T"),
            py::arg("scr_all"),
            py::arg("scr_mrg"),
            py::arg("stt_method"));

      // Bind multiCMH
      m.def("multiCMH_cpp",
            &multiCMH_cpp,
            "Compute the multi-CMH test statistics and returns the corrected p-value and else",
            py::arg("x"),
            py::arg("y"),
            py::arg("Z"),
            py::arg("x_dsc"),
            py::arg("y_dsc"),
            py::arg("k1"),
            py::arg("k2"),
            py::arg("alp"),
            py::arg("sttm_n"),
            py::arg("stt_method"),
            py::arg("scr_all"),
            py::arg("scr_mrg"),
            py::arg("maxT"),
            py::arg("verbose"));
}