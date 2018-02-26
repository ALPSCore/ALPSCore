/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_test_support.hpp

    @brief Utility classes to support param testing
*/

#ifndef PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501
#define PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501

#include <boost/scoped_ptr.hpp>
#include <alps/params.hpp>
#include <alps/testing/unique_file.hpp>

#include <gtest/gtest.h>

#include <fstream>

/// Helper class to make `argc` & `argv` from argument strings
class arg_holder {
  private:
    std::vector<std::string> args_;
    std::vector<const char*> argv_;
  public:
    explicit arg_holder(const std::string progname="./program_name") {
        this->add(progname);
    }

    arg_holder& add(const std::string& arg) {
        args_.push_back(arg);
        return *this;
    }

    int argc() const { return args_.size(); }

    /** @warning The returned array may be invalidated after next call to `add()`
        @warning The returned array is surely invalidated after the next call to `argv()`
    **/
    const char* const * argv() {
        argv_.resize(args_.size());
        std::transform(args_.begin(), args_.end(), argv_.begin(),
                       [](const std::string& s) { return s.c_str(); });
        return argv_.data();
    }
};

/// Helper class to make INI file
class ini_maker {
  private:
    alps::testing::unique_file file_;
    std::ofstream fstream_;
  public:
    ini_maker(const std::string& prefix, alps::testing::unique_file::action_type action=alps::testing::unique_file::REMOVE_AFTER)
        : file_(prefix, action), fstream_(file_.name().c_str())
    {
        if (!fstream_) throw std::runtime_error("ini_maker: Unable to open "+file_.name());
    }

    ini_maker& add(const std::string& line) {
        fstream_ << line << std::endl;
        return *this;
    }

    const std::string& name() const { return file_.name(); }
};


class ParamsAndFile {
    alps::testing::unique_file uniqf_;
    boost::scoped_ptr<alps::params_ns::params> params_ptr_;

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
        params_ptr_.reset(new alps::params_ns::params(uniqf_.name()));
    }

    const std::string& fname() const { return uniqf_.name(); }
    alps::params_ns::params* get_params_ptr() const { return params_ptr_.get(); }
};


#endif /* PARAMS_TEST_PARAMS_TEST_SUPPORT_HPP_3f7fd28aba0945619cbf187223b2f501 */
