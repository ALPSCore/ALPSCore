/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file iniparser_interface.hpp
    Declares C++ class to work with C ini-file parser library */

#ifndef ALPS_PARAMS_DICT_INIPARSER_INTERFACE_HPP_eeee10ab39ff42f29a29ed6f682e58d9
#define ALPS_PARAMS_DICT_INIPARSER_INTERFACE_HPP_eeee10ab39ff42f29a29ed6f682e58d9

#include <boost/scoped_ptr.hpp>
#include <vector>
#include <string>
#include <utility> // for std::pair


namespace alps {
    namespace params_ns {
        namespace detail {

            class ini_dict_impl;

            class iniparser {
                boost::scoped_ptr<ini_dict_impl> ini_dict_ptr_;
              public:
                typedef std::pair<std::string, std::string> kv_pair;
                typedef std::vector<kv_pair> kv_container_type;
                
                /// Reads the INI file and parses it
                iniparser(const std::string& inifile);

                ~iniparser();

                /// Returns a container (actually, vector of pairs) of keys and values
                kv_container_type operator()() const;
            };
        } // ::detail
    } // ::params_ns
} // ::alps

#endif /* ALPS_PARAMS_DICT_INIPARSER_INTERFACE_HPP_eeee10ab39ff42f29a29ed6f682e58d9 */
