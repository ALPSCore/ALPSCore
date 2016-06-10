/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_detail.hpp Contains private implementation-specific classes and functions */

#ifndef ALPS_PARAMS_PARAMS_DETAIL_HPP_INCLUDED_c2521f700b98472599377dee5f7772e7
#define ALPS_PARAMS_PARAMS_DETAIL_HPP_INCLUDED_c2521f700b98472599377dee5f7772e7

#include "./option_type.hpp"

namespace alps {
    namespace params_ns {

        /// Namespace hiding boring/awkward implementation details
        namespace detail {

            /// Service cast-via-validate function from a string to (presumably scalar) type T
            template <typename T>
            static T validate_cast(const std::string& sval)
            {
                using boost::program_options::validate;
                std::vector<std::string> sval_vec(1);
                sval_vec[0]=sval;
                boost::any outval;
                validate(outval, sval_vec, (T*)0, 0);
                return boost::any_cast<T>(outval);
            }


            /// Validator for strings, used by boost::program_options
            void validate(boost::any& outval, const std::vector<std::string>& strvalues,
                          string_container*, int);

            /// Validator for vectors, used by boost::program_options
            template <typename T>
            void validate(boost::any& outval, const std::vector<std::string>& strvalues,
                          vector_tag<T>*, int)
            {
                namespace po=boost::program_options;
                namespace pov=po::validators;
                typedef std::vector<std::string> strvec;
                typedef boost::char_separator<char> charsep;

                // std::cerr << "***DEBUG: entering validate() (templated) ***" << std::endl;

                pov::check_first_occurrence(outval); // check that this option has not yet been assigned
                const std::string in_str=pov::get_single_string(strvalues); // check that this option is passed a single value

                // Now, do parsing
                boost::tokenizer<charsep> tok(in_str,charsep(" ;,"));
                const strvec tokens(tok.begin(),tok.end());
                std::vector<T> typed_outval(tokens.size());
                std::transform(tokens.begin(), tokens.end(), typed_outval.begin(), detail::validate_cast<T>);

                outval=boost::any(typed_outval);
                // std::cerr << "***DEBUG: returning from validate() (templated) ***" << std::endl;
            }
        } // detail::
    } // params_ns::
} // alps::

#endif /* ALPS_PARAMS_PARAMS_DETAIL_HPP_INCLUDED_c2521f700b98472599377dee5f7772e7 */
