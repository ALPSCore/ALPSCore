/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dict_exceptions.hpp Defines exceptions that can be thrown by the dictionary */

#ifndef ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71
#define ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71

namespace alps {
    namespace params_new_ns {

        namespace exception {
            /// General exception (base class)
            class exception_base : public std::runtime_error {
                std::string name_; ///< name of the option that caused the error
            public:
                exception_base(const std::string& a_name, const std::string& a_reason)
                    : std::runtime_error("Key '"+a_name+"': "+a_reason),
                      name_(a_name)
                {}

                std::string name() const { return name_; }

                ~exception_base() throw() {}
            };
            
            /// Exception for using uninitialized value
            struct uninitialized_value : public exception_base {
                uninitialized_value (const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {};
            };
            
        } // ::exception
    } // ::params_ns
} // ::alps

#endif /* ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71 */
