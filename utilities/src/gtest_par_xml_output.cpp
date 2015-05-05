/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <string>
#include <cstring>
#include "boost/filesystem.hpp"
#include "boost/lexical_cast.hpp"

namespace alps {

    /// @warning Current implementation heap-allocates a char[] array
    /// for each "--gtest=xml..." command-line argument, that never gets deallocated.
    void gtest_par_xml_output(unsigned int irank, int argc, char** argv)
    {
        static const char option_prefix[]="--gtest_output=xml";
        static const size_t option_prefix_len=sizeof(option_prefix)-1;

        const std::string srank=boost::lexical_cast<std::string>(irank);
        if (argc<2) return;
        for (int i=1; i<argc; ++i) {
            if (std::string(argv[i]).find(option_prefix)!=0) continue;
            std::string arg(argv[i]+option_prefix_len);
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
            argv[i]=new char[option_prefix_len+arg_new.size()+1]; // NOTE: this memory will never be deallocated :(
            memcpy(argv[i], option_prefix, option_prefix_len);
            memcpy(argv[i]+option_prefix_len, arg_new.c_str(), arg_new.size()+1);
            // FIXME: An alternative would be a proper "Argv class" that would free all associated memory on destruction,
            // FIXME: possibly having Gtest_Mod_Argv as a derived class.
        }
    }
}
