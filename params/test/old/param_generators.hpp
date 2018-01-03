/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef PARAMS_TEST_PARAM_GENERATORS_HPP_INCLUDED
#define PARAMS_TEST_PARAM_GENERATORS_HPP_INCLUDED

/** @file param_generators.hpp
    @brief Utility classes to generate alps::params objects for testing purposes.
*/

/* FIXME: the code can be used for testing of MPI and serialization too. */
/* FIXME: uninitialized parameters, "trigger" type and vector types are not considered */
/* FIXME: constructing from INI files is not considered */
/* FIXME: constructing from archive is not considered */

#include "alps/params.hpp"

namespace alps {
    namespace params_ns {
        namespace testing {

            /// Utility function: cast a value as an input string
            template <typename T>
            std::string toInputString(T val)
            {
                return boost::lexical_cast<std::string>(val); // good for numeric types and bool
            }

            std::string toInputString(const std::string& val)
            {
                return "\""+val+"\"";
            }

            template <typename T>
            std::string toInputString(const std::vector<T>& vec)
            {
                typename std::vector<T>::const_iterator it=vec.begin(), end=vec.end();
                std::string out;
                while (it!=end) {
                    out += toInputString(*it);
                    ++it;
                    if (it!=end)  out+=",";
                }
                return out;
            }

            
            /// Utility function: generate a parameter from string as if from a command line
            inline params gen_param(const std::string& name, const std::string& val)
            {
                std::string arg="--"+name+"="+val;
                const char* argv[]={ "program_name", arg.c_str() };
                const int argc=sizeof(argv)/sizeof(*argv);
                return params(argc,argv);
            }

            /// Utility function: generate a test value of scalar type T
            template <typename T>
            inline void gen_data(T& val) { val=T(321.125); } // good enough for any integer and floating point

#define ALPS_TEST_DEF_GENERATOR(atype,aval) inline void gen_data(atype& val) { val=aval; }
            ALPS_TEST_DEF_GENERATOR(std::string,"hello, world!");
            ALPS_TEST_DEF_GENERATOR(bool,true);
#undef ALPS_TEST_DEF_GENERATOR

            /// Utility function: generate a test value of std::vector<T>
            template <typename T>
            inline void gen_data(std::vector<T>& val)
            {
                T x;
                gen_data(x);
                std::vector<T> vec(3,x);
                val=vec;
            }
          
            /// Utility function: generate "another" test value of type T (not equal to one from `gen_data<T>()`)
            template <typename T>
            inline void gen_other_data(T& val) { gen_data(val); val+=boost::lexical_cast<T>(1); } // works for scalars and strings
            
            inline void gen_other_data(bool& val) { gen_data(val); val=!val; }

            /// Base class: stores name and value associated with a parameter of type T
            template <typename T>
            class BasicParameter
            {
                public:
                typedef T value_type;
                std::string name_;
                value_type val_;

                /// Constructor, initializes name and value
                BasicParameter(const std::string& name): name_(name)
                {
                    gen_data(val_);
                }

                /// Returns associated value
                T data() const { return val_; } 

                /// Returns a value that is not equal to the stored one
                T other_data() const { T v; gen_other_data(v); return v; } // data "other than" stored value
            };

            /// Generates a parameter object as if from a command line
            template <typename T>
            class CmdlineParameter : public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                CmdlineParameter(const std::string& name): B(name) {}

                /// Returns parameters object generated from a command line
                alps::params params() const
                {
                    alps::params p=gen_param(B::name_, toInputString(B::data()));
                    return p.define<typename B::value_type>(B::name_,"some parameter");
                }
            };

            /// Generate a parameter object as if from a command line, with the option having a default value
            template <typename T>
            class CmdlineParameterWithDefault: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                CmdlineParameterWithDefault(const std::string& name): B(name) {}

                /// Returns parameters object generated from a command line, but having a default value
                alps::params params() const
                {
                    alps::params p=gen_param(B::name_, toInputString(B::data()));
                    return p.define<typename B::value_type>(B::name_, B::other_data(), "some parameter");
                }
            };

            /// Generate a parameter object without a default value, missing from a command line
            template <typename T>
            class MissingParameterNoDefault: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                MissingParameterNoDefault(const std::string& name): B(name) {}

                /// Returns parameters object w/o default
                alps::params params() const
                {
                    // generate parameter from cmdline with other-name and other-data
                    alps::params p=gen_param(B::name_+"_other",toInputString(B::other_data()));
                    return p.define<typename B::value_type>(B::name_, "some parameter");
                }
            };

            /// Generate a parameter object with a default value, missing from a command line
            template <typename T>
            class MissingParameterWithDefault: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                MissingParameterWithDefault(const std::string& name): B(name) {}

                /// Returns parameters object with default
                alps::params params() const
                {
                    // generate parameter from cmdline with other-name and other-data
                    alps::params p=gen_param(B::name_+"_other",boost::lexical_cast<std::string>(this->other_data()));
                    return p.define<typename B::value_type>(B::name_, this->other_data(), "some parameter");
                }
            };

            /// Generate a parameter object with the option having a default value, without any command line
            template <typename T>
            class ParameterWithDefault: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                ParameterWithDefault(const std::string& name): B(name) {}

                /// Returns parameters object having a default value
                alps::params params() const
                {
                    alps::params p;
                    return p.define<typename B::value_type>(B::name_, B::data(), "some parameter");
                }
            };

            /// Generate a parameter object without a default value, without any command line
            template <typename T>
            class ParameterNoDefault: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                ParameterNoDefault(const std::string& name): B(name) {}

                /// Returns parameters object w/o default
                alps::params params() const
                {
                    alps::params p;
                    return p.define<typename B::value_type>(B::name_, "some parameter");
                }
            };

            /// Generate a parameter object by direct assignment
            template <typename T>
            class AssignedParameter: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                AssignedParameter(const std::string& name): B(name) {}

                /// Returns parameters object generated by a direct assignment 
                alps::params params() const
                {
                    alps::params p;
                    p[B::name_]=B::val_;
                    return p;
                }
            };

            /// Generate a parameter object as if from command line, then assign
            template <typename T>
            class OverriddenParameter: public BasicParameter<T>
            {
                public:
                typedef BasicParameter<T> B;
                OverriddenParameter(const std::string& name): B(name) {}
                /// Returns parameters object generated from a command line and then assigned to
                alps::params params() const
                {
                    alps::params p=gen_param(B::name_, toInputString(B::other_data()));
                    p.define<typename B::value_type>(B::name_, "some parameter");
                    p[B::name_]=B::val_;
                    return p;
                }
            };
        } // namespace testing
    } // namespace params_ns
} // namespace alps

#endif // PARAMS_TEST_PARAM_GENERATORS_HPP_INCLUDED
