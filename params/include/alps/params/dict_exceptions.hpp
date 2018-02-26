/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file dict_exceptions.hpp Defines exceptions that can be thrown by the dictionary */

#ifndef ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71
#define ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71

namespace alps {
    namespace params_ns {

        namespace exception {
            /// General exception (base class)
            class exception_base : public std::runtime_error {
                std::string name_; ///< name of the option that caused the error
                mutable std::string what_; ///< explanation of the error
            public:
                exception_base(const std::string& a_name, const std::string& a_reason)
                    : std::runtime_error(a_reason),
                      name_(a_name), what_(a_reason)
                {}

                std::string name() const { return name_; }
                void set_name(const std::string& name) { name_=name; }

                virtual const char* what() const throw() {
                    const std::string key(name_.empty() ? std::string("Unknown_key") : ("Key '"+name_+"'"));
                    what_=key+": "+std::runtime_error::what();
                    return what_.c_str();
                }

                ~exception_base() throw() {}
            };
            
            /// Exception for using uninitialized value
            struct uninitialized_value : public exception_base {
                uninitialized_value (const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {}
            };
            
            /// Exception for type mismatch
            struct type_mismatch : public exception_base {
                type_mismatch(const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {}
            };
            
            /// Exception for value mismatch
            struct value_mismatch : public exception_base {
                value_mismatch(const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {}
            };

            // /// Exception for duplicated definition
            struct double_definition : public exception_base {
                double_definition(const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {}
            };
            
        } // ::exception
    } // ::params_ns
} // ::alps

#endif /* ALPS_PARAMS_DICT_EXCEPTIONS_HPP_046b246d331b4e54bedf6bfef6e54a71 */
