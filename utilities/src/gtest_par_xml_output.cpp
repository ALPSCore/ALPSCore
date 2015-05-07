/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <string>
#include <cstring>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

#include "alps/utilities/gtest_par_xml_output.hpp"

namespace alps {

    void gtest_par_xml_output::operator()(unsigned int irank, int argc, char** argv)
    {
        if (argc<2) return;

        const std::string option_prefix="--gtest_output=xml";
        const size_t prefix_len=option_prefix.size();
        const std::string srank=boost::lexical_cast<std::string>(irank);
        const size_t npos=std::string::npos;

        for (int i=1; i<argc; ++i) {
            std::string arg(argv[i]);
            if (arg.compare(0,prefix_len,option_prefix)!=0) continue; // starts with prefix? 
            arg.replace(0,prefix_len, "",0); // remove the prefix; arg is "argument" after "=xml"
            std::string arg_new;
            if (arg.empty()) { // "=xml"
                arg_new=":test_details"+srank+".xml";
            } else {
                if (arg[0]!=':') continue;
                if (arg[arg.size()-1]=='/') {
                    arg_new=arg.substr(0,arg.size()-1)+srank+"/";
                } else {
                    std::string ext=boost::filesystem::path(arg.substr(1,arg.size()-1)).extension().string();
                    arg_new=arg.substr(0,arg.size()-ext.size())
                            +srank+ext;
                }
            }
            std::string new_argv_i=option_prefix+arg_new;
            argv[i]=new char[new_argv_i.size()+1];
            // std::cerr << "DEBUG: gtest_par_xml_output() allocate ptr=" << (void*)argv[i] << "\n";
            keeper_.push_back(argv[i]);
            strcpy(argv[i],new_argv_i.c_str());
        }
    }

    gtest_par_xml_output::~gtest_par_xml_output()
    {
        BOOST_FOREACH(char* p, keeper_) {
            // std::cerr << "DEBUG: gtest_par_xml_output() deallocate ptr=" << (void*)p << "\n";
            delete[] p;
        }
    }
}
