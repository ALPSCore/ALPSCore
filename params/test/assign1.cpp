/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>
#include "gtest/gtest.h"

int main(void)
{
    alps::params parms;
    // parms["char"] = static_cast<char>(1);
    // parms["signed char"] = static_cast<signed char>(1);
    // parms["unsigned char"] = static_cast<unsigned char>(1);
    // parms["short"] = static_cast<short>(1);
    // parms["unsigned short"] = static_cast<unsigned short>(1);
    parms["int"] = static_cast<int>(1);
    parms["unsigned"] = static_cast<unsigned>(1);
    parms["long"] = static_cast<long>(1);
    // parms["unsigned long"] = static_cast<unsigned long>(1);
    // parms["long long"] = static_cast<long long>(1);
    // parms["unsigned long long"] = static_cast<unsigned long long>(1);
    // parms["float"] = static_cast<float>(1);
    parms["double"] = static_cast<double>(1);
    // parms["long double"] = static_cast<long double>(1);
    parms["bool"] = static_cast<bool>(1);
    parms["std::string"] = std::string("asdf");

    std::cout << parms << std::endl;

    return 0;
}
