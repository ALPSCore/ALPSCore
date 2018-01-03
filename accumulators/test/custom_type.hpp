/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file custom_type.hpp: "Custom" sequence-like type to be used as a data for an Accumulator. Declarations and template definitions. */

#ifndef ALPS_ACCUMULATORS_TEST_CUSTOM_TYPE_HPP
#define ALPS_ACCUMULATORS_TEST_CUSTOM_TYPE_HPP

#include <stdexcept>
#include <iostream>
#include <cmath>
#include <boost/type_traits/is_same.hpp>
#include <boost/optional.hpp>

/// Custom value types for accumulators: must be defined before inclusion of "config.hpp"
#define ALPS_ACCUMULATOR_USER_TYPES my_custom_type<double>

#include "alps/config.hpp"

/// A custom type forward declaration.
template <typename T> struct my_custom_type;

namespace alps {
    namespace numeric {
        // to allow friend declaration in the class
        template <typename T> my_custom_type<T> sqrt(my_custom_type<T>);

        // this is used later (FIXME: this should be avoided by defining my T::inf() outside the class)
        template <typename T> class inf;
    }

    namespace hdf5 {
        namespace detail {
            // to allow friend declaration in the classes
            template<typename T> class get_pointer;
            template<typename T> class get_extent;
        }
    }
}

/// We use alps::hdf5 types here, so this must be included:
#include "alps/hdf5/archive.hpp"


/// A custom type.
/** Has to be fully defined before the first inclusion of
    alps/accumulators.hpp, where it gets instantiated.
*/

template <typename T>
class my_custom_type {
    // this implementation of sqrt() needs friend access
    template <typename X> friend my_custom_type<X> alps::numeric::sqrt(my_custom_type<X>);

    // allow HDF5 serialization access the internals
    friend class alps::hdf5::detail::get_pointer<my_custom_type>;
    friend class alps::hdf5::detail::get_pointer<const my_custom_type>;
    friend class alps::hdf5::detail::get_extent<my_custom_type>;

    private:
    boost::optional<T> my_value_;  // `optional` to catch uninitialized use

    explicit my_custom_type(T v): my_value_(v) {} // `private` to catch conversions from T

    /// Accessor to catch uninitialized value access
    T& value() {
        if (!my_value_) throw std::runtime_error("Use of uninitialized my_custom_type");
        return *my_value_;
    }

    /// Const-Accessor to catch uninitialized value access
    const T& value() const {
        if (!my_value_) throw std::runtime_error("Use of uninitialized my_custom_type");
        return *my_value_;
    }

    public:

    /// Accessor to inspect the stored value
    T get_my_value() const { return this->value(); }
    
    /// "Constituent" type (for archiving and MPI). The "content" of the object should be of the "constituent" type!
    typedef T my_constituent_type;

    // "Math" Scalar type: should behave as a scalar type, with arithmetics and conversion from scalar types.
    // (NOTE: it also has to be a C++ scalar, from the point of view of boost::is_scalar!)
    typedef T my_scalar_type; // happens to be the same as T in this implementation

    // "Element type" (as a sequence)
    typedef T my_element_type;

    /// Default constructor, leaving the object in an **initialized** state @REQUIRED
    my_custom_type() : my_value_(T()) { }
    
    
    // If it is a sequence and the default template specializations are used, the following 3 needs to be implemented:
    
    /// Size of the sequence @REQUIRED_SEQUENCE
    std::size_t size() const { return 1; }

    /// Access to sequence elements @REQUIRED_SEQUENCE
    my_element_type& operator[](std::size_t i) {
        if (i!=0) throw std::runtime_error("Out-of-bound access to my_custom_type<T>");
        return this->value();
    }

    /// const-Access to sequence elements @REQUIRED_SEQUENCE
    const my_element_type& operator[](std::size_t i) const {
        if (i!=0) throw std::runtime_error("Out-of-bound access to my_custom_type<T>");
        return this->value();
    }

    // If the type is not continuous, it is supposed to have vector-like interface (even if it's not actually vectorizable)

    /// const-iterator type @REQUIRED_NONCONTINUOUS
    typedef const my_element_type* const_iterator;

    /// value type @REQUIRED_NONCONTINUOUS
    typedef my_element_type value_type;

    /// iterator to the beginning of the contents @REQUIRED_NONCONTINUOUS
    const_iterator begin() const {
        // throw std::logic_error("custom_type::begin() Not implemented");
        return &(*this)[0];
    }

    /// iterator to beyond the end of the contents @REQUIRED_NONCONTINUOUS
    const_iterator end() const {
        // throw std::logic_error("custom_type::end() Not implemented");
        return &(*this)[0]+size();
    }

    /// Add-assign operator with the same type at RHS. @REQUIRED (for mean calculation)
    my_custom_type& operator+=(const my_custom_type& rhs) {
        this->value() += rhs.value();
        return *this;
    }
    
    /// Divide operator with scalar_type<my_custom_type> at RHS. @REQUIRED (for many calculations)
    my_custom_type operator/(const my_scalar_type& c) const {
        my_custom_type r=*this;
        r.value() /= c;
        return r;
    }

    /// Multiply operator with scalar_type<my_custom_type> at RHS. @REQUIRED (for many calculations)
    /** WARNING: It is assumed that multiplication is an inverse of division, that is:
                custom_type val=...;
                T count=...;
                custom_type mean=val/count;
                assert(mean*count == val);
    */
    my_custom_type operator*(const my_scalar_type& c) const {
        my_custom_type r=*this;
        r.value() *= c;
        return r;
    }

    /// Addition operator. @REQUIRED (for many calculations, must be consistent with +=)
    // FIXME: use boost operators, define via +=
    my_custom_type operator+(const my_custom_type& rhs) const {
        my_custom_type r=*this;
        r.value() += rhs.value();
        return r;
    }

    /// Subtraction operator @REQUIRED (for many calculations, must be consistent with +)
    my_custom_type operator-(const my_custom_type& rhs) const {
        my_custom_type r=*this;
        r.value() -= rhs.value();
        return r;
    }

    /// Multiplication operator @REQUIRED (must be element-wise to make sense)
    my_custom_type operator*(const my_custom_type& rhs) const {
        my_custom_type r=*this;
        r.value() *= rhs.value();
        return r;
    }
    /// Division operator @REQUIRED (must be element-wise to make sense)
    my_custom_type operator/(const my_custom_type& rhs) const {
        my_custom_type r=*this;
        r.value() /= rhs.value();
        return r;
    }

    /// Unary minus (negation) operator. @REQUIRED
    my_custom_type operator-() const {
        my_custom_type r(-this->value());
        return r;
    }

    /// Add operator with scalar @REQUIRED (semantics: adds scaled identity custom_type)
    my_custom_type operator+(const my_scalar_type& s) const {
        my_custom_type r=*this;
        r.value() += s;
        return r;
    }
    
    /// Subtract operator with scalar @REQUIRED (semantics: subtracts scaled identity custom_type)
    my_custom_type operator-(const my_scalar_type& s) const {
        my_custom_type r=*this;
        r.value() -= s;
        return r;
    }

    /// Right-division (divide scalar by *this)
    /** Used in operator/; member method to avoid quirks with friend templates */
    my_custom_type right_div(const my_scalar_type& lhs) const {
        my_custom_type r(lhs/this->value());
        return r;
    }

    /// Print to a stream
    /** Used in operator<<; member method to avoid quirks with friend templates */
    std::ostream& print(std::ostream& s) const {
        s << "[Custom type: value=" << this->value() << "]";
        return s;
    }

    /// Verify if the value can be loaded
    static bool can_load(alps::hdf5::archive& ar, const std::string& path) {
        std::string attr_path=path+"/@c++_type";
        if (!ar.is_attribute(attr_path)) return false;
        std::string attr_val;
        ar[attr_path] >> attr_val;
        return (attr_val == "my_custom_type" && ar.is_datatype<T>(path));
    }

    /// Generate infinite value
    static my_custom_type inf() {
        my_custom_type r=my_custom_type(alps::numeric::inf<T>(T()));
        return r;
    }

    /// Generate a value with given content
    /** This is just a fancy way to construct the object:
        it's done this way to verify we never need a constructor from scalar type
        anywhere inside Accumulators
    */
    static my_custom_type generate(T data) {
        return my_custom_type(data);
    }
};


/// Stream-output operator @REQUIRED
template <typename T>
std::ostream& operator<<(std::ostream& s, const my_custom_type<T>& obj)
{
    return obj.print(s);
}

/// Right division operator. @REQUIRED (Needed for error bars in trigonometric functions)
template <typename T>
my_custom_type<T> operator/(const typename my_custom_type<T>::my_scalar_type& lhs, const my_custom_type<T>& rhs)
{
    return rhs.right_div(lhs);
}


// needed for the traits specialized below
#include "alps/type_traits/is_sequence.hpp"
#include "alps/type_traits/element_type.hpp"
#include "alps/numeric/scalar.hpp"
#include "alps/numeric/inf.hpp"

namespace alps {
    /// Declare that the type is a sequence, despite a possible absence of value_type @REQUIRED
    template <typename T>
    struct is_sequence< my_custom_type<T> > : public boost::true_type {};
    
    /// Declare the element type @REQUIRED
    template <typename T>
    struct element_type< my_custom_type<T> > {
        typedef typename my_custom_type<T>::my_element_type type;
    };
    
    
    namespace numeric {
        // /// Setting "negative" values to zero @REQUIRED (for autocorrelation). Already implemented by ALPSCore for sequences.
        // /** NOTE: Has to be done before including "accumulators.hpp" */
        // template <typename T>
        // void set_negative_0(my_custom_type<T>& x)
        // {
        //     throw std::logic_error("set_negative_0() value is not yet implemented for this type");
        // }
        
        namespace {
            inline std::runtime_error not_implemented(const std::string& fname) {
                return std::runtime_error("Function "+fname+"() is not implemented for this type.");
            }
        }

        /// Square root @REQUIRED (for error bar calculations). Must be element-wise to make sense.
        template <typename T>
        my_custom_type<T> sqrt(my_custom_type<T> x) {
            using std::sqrt;
            x.value() = sqrt(x.value());
            return x;
        }

        // a set of standard math functions for the custom type. @REQUIRED (for the corresponding math)
        /* NOTE: Has to be done before including "accumulators.hpp" */
        template <typename T> inline my_custom_type<T>  sin(my_custom_type<T>) { throw not_implemented("sin"); }
        template <typename T> inline my_custom_type<T>  cos(my_custom_type<T>) { throw not_implemented("cos"); }
        template <typename T> inline my_custom_type<T>  tan(my_custom_type<T>) { throw not_implemented("tan"); }
        template <typename T> inline my_custom_type<T> sinh(my_custom_type<T>) { throw not_implemented("sinh"); }
        template <typename T> inline my_custom_type<T> cosh(my_custom_type<T>) { throw not_implemented("cosh"); }
        template <typename T> inline my_custom_type<T> tanh(my_custom_type<T>) { throw not_implemented("tanh"); }
        template <typename T> inline my_custom_type<T> asin(my_custom_type<T>) { throw not_implemented("asin"); }
        template <typename T> inline my_custom_type<T> acos(my_custom_type<T>) { throw not_implemented("acos"); }
        template <typename T> inline my_custom_type<T> atan(my_custom_type<T>) { throw not_implemented("atan"); }
        template <typename T> inline my_custom_type<T>  abs(my_custom_type<T>) { throw not_implemented("abs"); }
        template <typename T> inline my_custom_type<T>  log(my_custom_type<T>) { throw not_implemented("log"); }
        template <typename T> inline my_custom_type<T> cbrt(my_custom_type<T>) { throw not_implemented("cbrt"); }

        /// if element_type<T> is not its scalar type, define scalar here, before "accumulators.hpp" (it gets instantiated inside)
        template <typename T>
        struct scalar< my_custom_type<T> > {
            typedef typename my_custom_type<T>::my_scalar_type type;
        };
    } // numeric::
}


// Contains definitions of traits specialized below
#include "alps/hdf5/archive.hpp"

namespace alps {
    namespace hdf5 {
        /// Specialization of alps::hdf5::scalar_type<T> for the custom_type<...>
        template <typename T>
        struct scalar_type< my_custom_type<T> > {
            typedef typename my_custom_type<T>::my_constituent_type type;
        };

        /// Specialization of alps::hdf5::is_content_continuous<T> for the custom_type<...>
        template <typename T>
        struct is_content_continuous< my_custom_type<T> >
            : public is_continuous<typename my_custom_type<T>::my_constituent_type> {};

        /// Specialization of alps::hdf5::is_continuous<T> for the custom_type<...>
        template <typename T>
        struct is_continuous< my_custom_type<T> >
            : public boost::false_type {}; // the type is NOT continuous: an array of this type does not contain array of T!

        namespace detail {
            /// Specialization of get_pointer<custom_type>
            template<typename T> struct get_pointer< my_custom_type<T> > {
                static typename my_custom_type<T>::my_constituent_type* apply(my_custom_type<T>& x) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(x.value());
                }
            };
            
            /// Specialization of get_pointer<const custom_type>
            template<typename T> struct get_pointer< const my_custom_type<T> > {
                static const typename my_custom_type<T>::my_constituent_type* apply(const my_custom_type<T>& x) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(x.value());
                }
            };
            
            /// Specialization of get_extent<custom_type>
            template<typename T> struct get_extent< my_custom_type<T> > {
                static std::vector<std::size_t> apply(const my_custom_type<T>& /*x*/) {
                    using alps::hdf5::get_extent;
                    std::vector<std::size_t> ext(1,1); // 1D object of size 1
                    return ext;
                }
            };
            
            /// Specialization of set_extent<custom_type>
            template<typename T> struct set_extent< my_custom_type<T> > {
                static void apply(const my_custom_type<T>& /*x*/, const std::vector<std::size_t>& ext) {
                    using alps::hdf5::set_extent;
                    if (ext.size()!=1 || ext[0]!=1) { // expecting 1D object of size 1
                        throw std::runtime_error("Unexpected extent values");
                    }
                }
            };

            /// Specialization of is_vectorizable<custom_type>
            template<typename T> struct is_vectorizable< my_custom_type<T> > {
                static bool apply(const my_custom_type<T>& /*x*/) {
                    // return false;
                    return true;
                }
            };

        } // detail::
  
        /// Overload of load() for the custom_type<...>
        template <typename T>
        void load(archive& ar, const std::string& path,
                  my_custom_type<T>& obj,
                  std::vector<std::size_t> chunk  =std::vector<std::size_t>(),
                  std::vector<std::size_t> offset =std::vector<std::size_t>())
        {
            if (!is_continuous<T>::value) {
                throw std::invalid_argument("Can load only my_custom_type<continuous_type>");
            }

            if (!my_custom_type<T>::can_load(ar,path)) {
                throw std::runtime_error("Attempt to load a value of a wrong type");
            }

            std::vector<std::size_t> extent=get_extent(obj);
            std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
            std::fill_n(std::back_inserter(offset), extent.size(), 0);
            typename my_custom_type<T>::my_constituent_type* ptr=get_pointer(obj);
            load(ar, path, *ptr, chunk, offset);
        }

        /// Overload of save() for the custom_type<...>
        template <typename T>
        void save(archive& ar, const std::string& path,
                  const my_custom_type<T>& obj,
                  std::vector<std::size_t> size   =std::vector<std::size_t>(),
                  std::vector<std::size_t> chunk  =std::vector<std::size_t>(),
                  std::vector<std::size_t> offset =std::vector<std::size_t>())
        {
            if (!is_continuous<T>::value) {
                throw std::invalid_argument("Can save only my_custom_type<continuous_type>");
            }
            std::vector<std::size_t> extent=get_extent(obj);
            std::copy(extent.begin(), extent.end(), std::back_inserter(size));
            std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
            std::fill_n(std::back_inserter(offset), extent.size(), 0);
            const typename my_custom_type<T>::my_constituent_type* ptr=get_pointer(obj);
            save(ar, path, *ptr, size, chunk, offset);
            ar[path+"/@c++_type"]="my_custom_type";
        }
    } // hdf5::
    
    namespace numeric {
        /// This must be specialized to give the notion of "infinity" (for "undefined" error bars)
        /** The type should be default-constructible and convertible to custom_type */
        template <typename T>
        struct inf< my_custom_type<T> > {
            typedef my_custom_type<T> value_type;
            inf(const value_type&) {}
            operator my_custom_type<T>() const {
                return value_type::inf();
            }
        };
    
    } // numeric::

} // alps::

// To declare traits specialized below:
#include "alps/accumulators/archive_traits.hpp"

namespace alps {
    namespace accumulators {
        namespace detail {
            template <typename T>
            struct archive_trait< my_custom_type<T> > {
                static bool can_load(hdf5::archive& ar,
                                     const std::string& name,
                                     std::size_t dim)
                {
                    return ar.is_data(name)
                           && my_custom_type<T>::can_load(ar,name)
                           && ar.dimensions(name)==dim;
                }
            };
        } // detail::
    } // accumulators::
} // alps::


#endif /* ALPS_ACCUMULATORS_TEST_CUSTOM_TYPE_HPP */
