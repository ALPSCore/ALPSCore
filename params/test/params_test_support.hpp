/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_test_support.hpp
    
    @brief Utility classes to support param testing
*/

#ifndef PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501
#define PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501

#include <boost/scoped_ptr.hpp>
#include <alps/params_new.hpp>
#include <alps/testing/unique_file.hpp>

#include <gtest/gtest.h>

#include <fstream>

class ParamsAndFile {
    alps::testing::unique_file uniqf_;
    boost::scoped_ptr<alps::params_new_ns::params> params_ptr_;

    void write_ini_(const std::string& content) const {
        std::ofstream outf(uniqf_.name().c_str());
        if (!outf) throw std::runtime_error("Can't open temporary file "+uniqf_.name());
        outf << content;
    }

    public:
    // Make param object from a given file content
    ParamsAndFile(const char* ini_content) : uniqf_("params.ini.", alps::testing::unique_file::REMOVE_AFTER), params_ptr_(0)
    {
        write_ini_(ini_content);
        params_ptr_.reset(new alps::params_new_ns::params(uniqf_.name()));
    }

    const std::string& fname() const { return uniqf_.name(); }
    alps::params_new_ns::params* get_params_ptr() const { return params_ptr_.get(); }
};


#endif /* PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501 */
