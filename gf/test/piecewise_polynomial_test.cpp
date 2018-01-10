/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include "alps/gf/piecewise_polynomial.hpp"
#include <alps/testing/unique_file.hpp>

TEST(PiecewisePolynomial, Orthogonalization) {
    typedef double Scalar;
    const int n_section = 10, k = 8, n_basis = 3;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    boost::multi_array<Scalar,3> coeff(boost::extents[n_basis][n_section][k+1]);

    for (int s = 0; s < n_section + 1; ++s) {
        section_edges[s] = s*2.0/n_section - 1.0;
    }
    section_edges[0] = -1.0;
    section_edges[n_section] = 1.0;

    std::vector<pp_type> nfunctions;

    // x^0, x^1, x^2, ...
    for (int n = 0; n < n_basis; ++ n) {
        boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
        std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

        for (int s = 0; s < n_section; ++s) {
            double rtmp = 1.0;
            for (int l = 0; l < k + 1; ++l) {
                if (n - l < 0) {
                    break;
                }
                if (l > 0) {
                    rtmp /= l;
                    rtmp *= n + 1 - l;
                }
                coeff[s][l] = rtmp * std::pow(section_edges[s], n-l);
            }
        }

        nfunctions.push_back(pp_type(n_section, section_edges, coeff));
    }

    // Check if correctly constructed
    double x = 0.9;
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(nfunctions[n].compute_value(x), std::pow(x, n), 1e-8);
    }

    // Check overlap
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++ m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]), (std::pow(1.0,n+m+1)-std::pow(-1.0,n+m+1))/(n+m+1), 1e-8);
        }
    }


    // Check plus and minus
    for (int n = 0; n < n_basis; ++ n) {
        EXPECT_NEAR(4 * nfunctions[n].compute_value(x), (4.0*nfunctions[n]).compute_value(x), 1e-8);
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].compute_value(x) + nfunctions[m].compute_value(x),
                        (nfunctions[n] + nfunctions[m]).compute_value(x), 1e-8);
            EXPECT_NEAR(nfunctions[n].compute_value(x) - nfunctions[m].compute_value(x),
                        (nfunctions[n] - nfunctions[m]).compute_value(x), 1e-8);
        }
    }

    alps::gf::orthonormalize(nfunctions);
    for (int n = 0; n < n_basis; ++ n) {
        for (int m = 0; m < n_basis; ++m) {
            EXPECT_NEAR(nfunctions[n].overlap(nfunctions[m]),
                        n == m ? 1.0 : 0.0,
                        1e-8
            );
        }
    }

    //l = 0 should be x
    EXPECT_NEAR(nfunctions[1].compute_value(x) * std::sqrt(2.0/3.0), x, 1E-8);
}

TEST(PiecewisePolynomial, SaveLoad) {
    alps::testing::unique_file ufile("pp.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();
    const int n_section = 2, k = 3;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;
    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff), p2;
    {
        alps::hdf5::archive oar(filename,"w");
        p.save(oar,"/pp");

    }

    {
        alps::hdf5::archive iar(filename);
        p2.load(iar,"/pp");
    }

    ASSERT_TRUE(p == p2);
}

TEST(PiecewisePolynomial, SaveLoadStream) {
    alps::testing::unique_file ufile("pp.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();
    const int n_section = 2, k = 3;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;
    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff), p2;
    {
        alps::hdf5::archive oar(filename,"w");
        oar["/pp"]<<p;

    }

    {
        alps::hdf5::archive iar(filename);
        iar["/pp"]>>p2;
    }

    ASSERT_TRUE(p == p2);
}

TEST(PiecewisePolynomial, Copy) {
    const int n_section = 2, k = 3;
    typedef double Scalar;
    typedef alps::gf::piecewise_polynomial<Scalar> pp_type;

    std::vector<double> section_edges(n_section+1);
    section_edges[0] = -1.0;
    section_edges[1] =  0.0;
    section_edges[2] =  1.0;
    boost::multi_array<Scalar,2> coeff(boost::extents[n_section][k+1]);
    std::fill(coeff.origin(), coeff.origin()+coeff.num_elements(), 0.0);

    pp_type p(n_section, section_edges, coeff), p2;
    EXPECT_NO_THROW({p2 = p;});
    EXPECT_TRUE(p2 == p);
}
