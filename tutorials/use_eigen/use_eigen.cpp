/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>

// this include file brings in ALPSCore configuration information
#include <alps/config.hpp>

/**
 * This example shows how to test for Eigen's presence in ALPSCore
 * 
 */

// Test for Eigen presence
#if !defined(ALPS_HAVE_EIGEN_VERSION)
#error "Your version of ALPSCore library does not provide Eigen dependency"
#endif

// Use some Eigen header:
#include <Eigen/Dense>

#include <alps/utilities/stringify.hpp> // for ALPS_STRINGIFY

int main()
{
    std::cout << "ALPSCore provides Eigen v. " ALPS_STRINGIFY(ALPS_HAVE_EIGEN_VERSION)
              << "\n"
              << "also available from Eigen as: "
              << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION
              << std::endl;
}
