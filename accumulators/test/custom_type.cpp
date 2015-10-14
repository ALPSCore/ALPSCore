/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <stdexcept>

/** @file custom_type.cpp: Test for using a custom type */

/// A custom type. FIXME: Has to be declared before the first inclusion of config.h
template <typename T> struct my_custom_type;


#define ALPS_ACCUMULATOR_VALUE_TYPES float, double, long double, std::vector<float>, std::vector<double>, std::vector<long double>, my_custom_type<double>
#define ALPS_ACCUMULATOR_VALUE_TYPES_SIZE 7

/// A custom type.
/** @FIXME: Has to be declared before the first inclusion of config.h
    
    @FIXME: Has to be fully defined before the first inclusion of alps/accumulators.hpp,
            otherwise alps::has_value_type< my_custom_type<T> > returns wrong result!
*/
template <typename T>
struct my_custom_type {
    T my_value;
    typedef T my_value_type;

    /// This type is needed (among other things?) for mean accumulator to work: treated as math-scalar type
    /** @detail Returned by `alps::element_type<...>` metafunction. */
    typedef my_value_type value_type;
    
    /// Unary minus (negation) operator.
    my_custom_type operator-() const { throw std::runtime_error("unary operator- is not implemented for this type"); }
    /// Addition operator
    my_custom_type operator+(my_custom_type) const { throw std::runtime_error("operator+ is not implemented for this type"); }
    /// Subtraction operator
    my_custom_type operator-(my_custom_type) const { throw std::runtime_error("operator- is not implemented for this type"); }
    /// Multiplication operator
    my_custom_type operator*(my_custom_type) const { throw std::runtime_error("operator* is not implemented for this type"); }
    /// Division operator
    my_custom_type operator/(my_custom_type) const { throw std::runtime_error("operator/ is not implemented for this type"); }

    /// Add-assign operator with the same type at RHS (needed for mean calculation)
    void operator+=(const my_custom_type&) { throw std::logic_error("operator+=: Not implemented"); }

    /// Divide operator with scalar_type<my_custom_type> at RHS (needed for mean calculation)
    /** WARNING: T is assumed to be scalar! */
    my_custom_type operator/(const my_value_type&) const { throw std::logic_error("operator/(scalar_type): Not implemented"); }

    /// Multiply operator with scalar_type<my_custom_type> at RHS (needed for mean calculation)
    /** WARNING: T is assumed to be scalar!
        WARNING: It is assumed that multiplication is an inverse of division, that is:
            custom_type val=...;
            T count=...;
            custom_type mean=val/count;
            assert(mean*count == val);
    */
    my_custom_type operator*(const my_value_type&) const { throw std::logic_error("operator*(scalar_type): Not implemented"); }

    /// Add-Assign operator with scalar
    my_custom_type& operator+=(const my_value_type&) { throw std::runtime_error("operator+= is not implemented for this type"); }
    
    /// Add operator with scalar
    my_custom_type operator+(const my_value_type&) const { throw std::runtime_error("operator+ is not implemented for this type"); }
    
    /// Subtract operator with scalar
    my_custom_type operator-(const my_value_type&) const { throw std::runtime_error("operator- is not implemented for this type"); }
};

namespace alps {
    namespace numeric {
        /// Setting "negative" values to zero (needed for autocorrelation). Already implemented by ALPSCore for sequences.
        /** FIXME: Has to be done before including "accumulators.hpp" */
        template <typename T>
        void set_negative_0(my_custom_type<T>& x)
        {
            throw std::logic_error("set_negative_0() value is not yet implemented for this type");
        }

        // a set of standard math functions for the custom type.
        /** FIXME: Has to be done before including "accumulators.hpp" */
        template <typename T> my_custom_type<T>  sin(my_custom_type<T>) { throw std::runtime_error("Function sin() is not implemented for this type."); }
        template <typename T> my_custom_type<T>  cos(my_custom_type<T>) { throw std::runtime_error("Function cos() is not implemented for this type."); }
        template <typename T> my_custom_type<T>  tan(my_custom_type<T>) { throw std::runtime_error("Function tan() is not implemented for this type."); }
        template <typename T> my_custom_type<T> sinh(my_custom_type<T>) { throw std::runtime_error("Function sinh() is not implemented for this type."); }
        template <typename T> my_custom_type<T> cosh(my_custom_type<T>) { throw std::runtime_error("Function cosh() is not implemented for this type."); }
        template <typename T> my_custom_type<T> tanh(my_custom_type<T>) { throw std::runtime_error("Function tanh() is not implemented for this type."); }
        template <typename T> my_custom_type<T> asin(my_custom_type<T>) { throw std::runtime_error("Function asin() is not implemented for this type."); }
        template <typename T> my_custom_type<T> acos(my_custom_type<T>) { throw std::runtime_error("Function acos() is not implemented for this type."); }
        template <typename T> my_custom_type<T> atan(my_custom_type<T>) { throw std::runtime_error("Function atan() is not implemented for this type."); }
        template <typename T> my_custom_type<T>  abs(my_custom_type<T>) { throw std::runtime_error("Function abs() is not implemented for this type."); }
        template <typename T> my_custom_type<T> sqrt(my_custom_type<T>) { throw std::runtime_error("Function sqrt() is not implemented for this type."); }
        template <typename T> my_custom_type<T>  log(my_custom_type<T>) { throw std::runtime_error("Function log() is not implemented for this type."); }
        template <typename T> my_custom_type<T> cbrt(my_custom_type<T>) { throw std::runtime_error("Function cbrt() is not implemented for this type."); }
        
    } // numeric::
}




#include "alps/accumulators.hpp"
#include "gtest/gtest.h"

namespace alps {
    namespace hdf5 {
        /// Specialization of alps::hdf5::scalar_type<T> for the custom_type<...>
        /** FIXME: should better be called `numeric::scalar_type` */
        template <typename T>
        struct scalar_type< my_custom_type<T> > {
            typedef typename my_custom_type<T>::my_value_type type;
        };

        /// Specialization of alps::hdf5::is_content_continuous<T> for the custom_type<...>
        template <typename T>
        struct is_content_continuous< my_custom_type<T> >
            : public is_continuous<T> {};

        /// Specialization of alps::hdf5::is_continuous<T> for the custom_type<...>
        template <typename T>
        struct is_continuous< my_custom_type<T> >
            : public is_content_continuous< my_custom_type<T> > {}; // the type is continuous if its content is continuous

        /// Overload of load() for the custom_type<...>
        template <typename T>
        void load(archive& ar, const std::string& path,
                  my_custom_type<T>& value,
                  std::vector<std::size_t> size   =std::vector<std::size_t>(),
                  std::vector<std::size_t> chunk  =std::vector<std::size_t>(),
                  std::vector<std::size_t> offset =std::vector<std::size_t>())
        {
            throw std::logic_error("load(custom_type) is not yet implemented");
        }

        namespace detail {
            /// Overload of get_pointer<custom_type>
            template<typename T> struct get_pointer< my_custom_type<T> > {
                static T* apply(my_custom_type<T>& value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value.my_value);
                }
            };
        } // detail::
  
        /// Overload of save() for the custom_type<...>
        template <typename T>
        void save(archive& ar, const std::string& path,
                  const my_custom_type<T>& value,
                  std::vector<std::size_t> size   =std::vector<std::size_t>(),
                  std::vector<std::size_t> chunk  =std::vector<std::size_t>(),
                  std::vector<std::size_t> offset =std::vector<std::size_t>())
        {
            if (ar.is_group(path)) ar.delete_group(path);
            if (is_continuous<T>::value) {
                std::vector<std::size_t> extent(get_extent(value));
                std::copy(extent.begin(), extent.end(), std::back_inserter(size));
                std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
                std::fill_n(std::back_inserter(offset), extent.size(), 0);
                ar.write(path, get_pointer(value), size, chunk, offset);
            } else {
                throw std::invalid_argument("Can save only custom_type<continuous_type>");
            }
        }
        
    } // hdf5::

    namespace numeric {
        /// This must be specialized to give the notion of "infinity" (for autocorrelation)
        /** The type should be default-constructible and convertible to custom_type */
        template <typename T>
        struct inf< my_custom_type<T> > {
            operator my_custom_type<T>() const {
                throw std::logic_error("The infinite value is not yet implemented for this type");
            }
        };
    
    } // numeric::

    /// Declare that the type is not a sequence, despite the presence of value_type
    template <typename T>
    struct is_sequence< my_custom_type<T> > : public boost::false_type {};
    
} // alps::


/// Stream-output operator
template <typename T>
std::ostream& operator<<(std::ostream& s, const my_custom_type<T>& obj)
{
    s << "[Custom type: value=" << obj.my_value << "]";
    return s;
}

/// Right division operator.
template <typename T>
my_custom_type<T> operator/(const T& lhs, const my_custom_type<T>& rhs)
{
    throw std::logic_error("operator/ (right div): Not implemented");
}

TEST(accumulators, CustomType) {
    using namespace alps::accumulators;
    typedef my_custom_type<double> dbl_custom_type;

    accumulator_set m;
    m << MeanAccumulator<dbl_custom_type>("mean");
    m << NoBinningAccumulator<dbl_custom_type>("nobin");
    m << LogBinningAccumulator<dbl_custom_type>("logbin");
    m << FullBinningAccumulator<dbl_custom_type>("fullbin");
}

// template <typename T>
// std::ostream& operator<<(std::ostream& s, const std::vector<T>& v)
// {
//     s << alps::short_print(v);
//     return s;
// }

// TEST(accumulators, VectorType) {
//     using namespace alps::accumulators;
//     typedef std::vector<double> dvec;
//     accumulator_set m;
//     m << MeanAccumulator<dvec>("custom");

//     double val[]={1., 3.};
//     m["custom"] << dvec(val, val+sizeof(val)/sizeof(*val));

//     std::cout << "Vector accumulator: " << m["custom"] << std::endl;

//     result_set res(m);
//     std::cout << "Vector mean: " << res["custom"].mean<dvec>() << std::endl;
//     std::cout << "Vector 5*mean: " << (res["custom"]*5.0).mean<dvec>() << std::endl;
//     res["custom"]+=5.0;
//     std::cout << "Vector +=5 mean: " << res["custom"].mean<dvec>() << std::endl;
// }
