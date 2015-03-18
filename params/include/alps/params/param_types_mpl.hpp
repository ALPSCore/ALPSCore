#ifndef ALPS_PARAMS_PARAM_TYPES_MPL_INCLUDED
#define ALPS_PARAMS_PARAM_TYPES_MPL_INCLUDED

/* This file is not in use as of 3/17/2015. This MPL-based code is replaced by a Boost PP-using one. */

#include <vector>

#include "boost/mpl/vector.hpp"
#include "boost/mpl/set.hpp"
#include "boost/mpl/insert.hpp"
#include "boost/mpl/transform.hpp"
#include "boost/mpl/fold.hpp"
#include "boost/mpl/placeholders.hpp"
#include "boost/mpl/bool.hpp"
#include <boost/mpl/and.hpp>
#include <boost/mpl/logical.hpp>

#include "boost/variant.hpp"
#include "boost/optional.hpp"

namespace alps {
    namespace params_ns {
        namespace detail {
            
            // Have namespaces handy
            namespace mpl=::boost::mpl;
            namespace mplh=::boost::mpl::placeholders;

            /// "Empty value" type
            struct None {};

            /// Output operator for the "empty value" (@throws runtime_error always)
            inline std::ostream& operator<<(std::ostream& s, const None&)
            {
                throw std::runtime_error("Attempt to print uninitialized option value");
            }

            // Vector of allowed scalar types:
            typedef mpl::vector<int,
				unsigned int,
                                // float,
				double,
                                // long double,
                                long int, unsigned long int,
                                // long long int, unsigned long long int,
                                char, // signed char, unsigned char,
                                // short int, unsigned short int,
                                bool>::type scalar_types_vec;

            // /// Make a set of allowed types (for fast look-up)
            // typedef mpl::fold< scalar_types_vec,
            //                    mpl::set<>, // empty set
            //                    mpl::insert<mplh::_1,mplh::_2>
            //                    >::type scalar_types_set;

            // Vector of std::vector<T> types (aka "vector types")
            typedef mpl::transform< scalar_types_vec, std::vector<mplh::_1> >::type vector_types_subvec;

            // Add std::string to the vector of the "vector types"
            typedef mpl::push_front<vector_types_subvec, std::string>::type vector_types_vec;

            // /// Make a set of "vector types" (for fast look-up):
            // typedef mpl::fold< vector_types_vec,
            //                    mpl::set<>, // empty set
            //                    mpl::insert<mplh::_1,mplh::_2>
            //                    >::type vector_types_set;

            // // Make a set of all types (for fast look-up):
            // typedef mpl::fold< vector_types_set,
            //                    scalar_types_set, 
            //                    mpl::insert<mplh::_1,mplh::_2>
            //                    >::type all_types_set;

            // Make a vector of all types (for boost::variant, starting with the scalar types)
            typedef mpl::fold< vector_types_vec, 
                               scalar_types_vec, 
                               mpl::push_back<mplh::_1, mplh::_2>
                               >::type all_types_vec;

            /// A variant of all types, including None (as the first one)
            typedef boost::make_variant_over< mpl::push_front<all_types_vec,None>::type >::type variant_all_type;

            /// A vector of `optional` types for each of the vectors and scalar types
            typedef mpl::transform< all_types_vec, boost::optional<mplh::_1> >::type optional_types_vec;

            /// An output operator for optionals of any type (throws unconditionally)
            template <typename T>
            inline std::ostream& operator<<(std::ostream& , const boost::optional<T>&)
            {
                throw std::logic_error("Attempt to use undefined operator<< for boost::optional<T>");
            }

            /// Tag type to indicate "trigger" option type (FIXME: it's a hack and must be redone)
            struct trigger_tag {};

            inline std::ostream& operator<<(std::ostream&, const trigger_tag&)
            {
                throw std::logic_error("Attempt to use undefined operator<< for trigger_tag");
            }

            /// A variant of the trigger_tag and optionals of all types
            typedef boost::make_variant_over< mpl::push_back<optional_types_vec,trigger_tag>::type >::type variant_all_optional_type;
            
            // /// A meta-function determining if both types are scalar
            // template <typename T, typename U>
            // struct both_scalar
            //     : mpl::and_< mpl::has_key<scalar_types_set,U>,
            //                  mpl::has_key<scalar_types_set,T> >
            // {};


        }

        // Elevate choosen generated types:
        // using detail::scalar_types_set;
        // using detail::vector_types_set;
        // using detail::all_types_set;
        using detail::variant_all_type;
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



#endif // ALPS_PARAMS_PARAM_TYPES_MPL_INCLUDED
