/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_PARAMS_PARAM_TYPES_INCLUDED
#define ALPS_PARAMS_PARAM_TYPES_INCLUDED

/** @file param_types.hpp Defines data types acceptable for alps::params parameters,
    as well as alps::param type machinery deried from these types.

    This version uses Boost preprocessor macros rather than MPL.
*/

#include <stdexcept>
#include <vector>
#include <iostream>

#include "boost/preprocessor/array/to_seq.hpp"
#include "boost/preprocessor/seq/transform.hpp"
#include "boost/preprocessor/seq/enum.hpp"
#include "boost/preprocessor/seq/for_each.hpp"

#include "boost/variant.hpp"
#include "boost/optional.hpp"

// FIXME: will go away with acceptance of boost::TypeIndex
#include "alps/params/typeindex.hpp"

namespace alps {
    namespace params_ns {
        namespace detail {
            
	    // Allowed basic numerical types.
	    // NOTE 1: do not forget to change "6" to the correct number if editing!
	    // NOTE 2: currently, not more than (20-1)/2 = 9 types are supported
	    //         (20 is boost::variant limit; we have std::vector<T> for each of
	    //         these basic types, plus None.)
            // NOTE 3: we support both `int` and `long`; supporting only `long` would be enough,
            //         but it's convenient for a user to use <int> when an integer parameter is needed.
            //         Although it's possible to "translate" <int> to <long> when needed,
            //         it is deemed to be an unnecessary complication.
            // NOTE 4: If new types are introduced, add them to type ranking in `param_types_ranking.hpp` also,
            //         to define what type is "convertible" to what.
#define	    ALPS_PARAMS_DETAIL_STYPES_VEC (6,(int,            \
                                              long,           \
					      double,         \
                                              char,           \
                                              bool,           \
					      std::string))

            /// "Empty value" type
            struct None {};

            /// Output operator for the "empty value" (@throws runtime_error always)
            inline std::ostream& operator<<(std::ostream& s, const None&)
            {
                throw std::runtime_error("Attempt to print uninitialized option value");
            }


	    // BOOST-PP Sequence of numerical types types
#define     ALPS_PARAMS_DETAIL_STYPES_SEQ BOOST_PP_ARRAY_TO_SEQ(ALPS_PARAMS_DETAIL_STYPES_VEC)

	    // Macro to make derived types
#define     ALPS_PARAMS_DETAIL_MAKE_TYPE(s,atype,elem) atype< elem >

            // Sequence of std::vector<T> types (aka "vector types")
#define     ALPS_PARAMS_DETAIL_VTYPES_SEQ BOOST_PP_SEQ_TRANSFORM(ALPS_PARAMS_DETAIL_MAKE_TYPE, std::vector, ALPS_PARAMS_DETAIL_STYPES_SEQ)

            /// Tag type to indicate "trigger" option type (FIXME: must be reacher)
            struct trigger_tag {};

            inline std::ostream& operator<<(std::ostream&, const trigger_tag&)
            {
                throw std::logic_error("Attempt to use undefined operator<< for trigger_tag");
            }
          
             // Sequence of trigger type, scalar and vector types
#define     ALPS_PARAMS_DETAIL_ALLTYPES_SEQ ALPS_PARAMS_DETAIL_STYPES_SEQ(trigger_tag)ALPS_PARAMS_DETAIL_VTYPES_SEQ

            // FIXME: will go away with acceptance of boost::TypeIndex
            // Generate a pretty-name specialization for scalar and vector types
#define     ALPS_PARAMS_DETAIL_GEN_TYPID(s, data, elem) ALPS_PARAMS_DETAIL_TYPID_NAME(elem)
            BOOST_PP_SEQ_FOR_EACH(ALPS_PARAMS_DETAIL_GEN_TYPID,~,ALPS_PARAMS_DETAIL_ALLTYPES_SEQ);
            // Generate a few more pretty-names, for frequently-needed types
            ALPS_PARAMS_DETAIL_TYPID_NAME(char *);
            ALPS_PARAMS_DETAIL_TYPID_NAME(const char *);
          
            // Sequence of `boost::optional<T>` types for all supported types
#define     ALPS_PARAMS_DETAIL_OTYPES_SEQ BOOST_PP_SEQ_TRANSFORM(ALPS_PARAMS_DETAIL_MAKE_TYPE, boost::optional, ALPS_PARAMS_DETAIL_ALLTYPES_SEQ)

            /// A variant of all types and None as the 1st type
	    typedef boost::variant< None, BOOST_PP_SEQ_ENUM(ALPS_PARAMS_DETAIL_OTYPES_SEQ) > variant_all_type;

            /// An output operator for optionals of any type (FIXME! throws unconditionally)
            template <typename T>
            inline std::ostream& operator<<(std::ostream& , const boost::optional<T>&)
            {
                throw std::logic_error("Attempt to use undefined operator<< for boost::optional<T>");
            }

            // /// A variant of the trigger_tag and optionals of all types
            // typedef boost::variant<BOOST_PP_SEQ_ENUM(ALPS_PARAMS_DETAIL_OTYPES_SEQ), trigger_tag> variant_all_optional_type;
        }

        // Elevate choosen generated types:
        using detail::variant_all_type;

	// Undefine local macros
#undef  ALPS_PARAMS_DETAIL_STYPES_VEC
#undef  ALPS_PARAMS_DETAIL_STYPES_SEQ
#undef  ALPS_PARAMS_DETAIL_MAKE_TYPE
#undef  ALPS_PARAMS_DETAIL_VTYPES_SEQ
#undef  ALPS_PARAMS_DETAIL_ALLTYPES_SEQ
#undef  ALPS_PARAMS_DETAIL_GEN_TYPID
#undef  ALPS_PARAMS_DETAIL_OTYPES_SEQ
	
    } // params_ns
}// alps

// The following is needed for serialization support
namespace boost {
    namespace serialization {
        /// Serialization function for the "empty value" (does nothing)
        template<class Archive>
        inline void serialize(Archive & ar, alps::params_ns::detail::None&, const unsigned int)
        { }

        /// Serialization function for the "trigger type" (does nothing)
        template<class Archive>
        inline void serialize(Archive & ar, alps::params_ns::detail::trigger_tag&, const unsigned int)
        { }
    } // serialization
} // boost


#endif // ALPS_PARAMS_PARAM_TYPES_INCLUDED
