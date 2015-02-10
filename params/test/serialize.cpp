/** Serialization tests */

/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <fstream>

// Serialization headers:
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"

#include "alps/params.hpp"
#include "gtest/gtest.h"

void Test()
{
    const char* argv[]={ "", "--param1=111" };
    const int argc=sizeof(argv)/sizeof(*argv);
    alps::params p(argc,argv);

    p.description("Serialization test").
        define<int>("param1","integer 1").
        define<double>("param2",22.25,"double");
    p["param3"]=333;
           

    std::ofstream outs("/dev/stdout"); // FIXME: give a file name
    {
        boost::archive::text_oarchive ar(outs);
        ar << p;
    }

}

int main(int argc, char** argv)
{
    return Test(), 0;
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
