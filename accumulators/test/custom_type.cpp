/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <stdexcept>
#include <complex>
#include <iostream>
#include <cmath>
#include <boost/type_traits/is_same.hpp>
#include "gtest/gtest.h"

/** @file custom_type.cpp: Test for using a custom type */

/// A custom type. FIXME: Has to be declared before the first inclusion of config.h
template <typename T> struct my_custom_type;


#define ALPS_ACCUMULATOR_VALUE_TYPES float, double, long double, std::vector<float>, std::vector<double>, std::vector<long double>, my_custom_type<double>
#define ALPS_ACCUMULATOR_VALUE_TYPES_SIZE 7

namespace alps {
    namespace numeric {
        // to allow friend declaration in the class
        template <typename T> my_custom_type<T> sqrt(my_custom_type<T>);

        // this is used later (FIXME: this should be avoided by defining my T::inf() outside the class)
        template <typename T> class inf;
    }

    namespace hdf5 {
        namespace detail {
            // to allow friend declaration in the class
            template<typename T> class get_pointer;
        }
    }
}

/// A custom type.
/** @FIXME: Has to be declared before the first inclusion of config.h
    
    Has to be fully defined before the first inclusion of alps/accumulators.hpp,
    where it gets instantiated.
*/

template <typename T>
class my_custom_type {
    // this implementation of sqrt() needs friend access
    template <typename X> friend my_custom_type<X> alps::numeric::sqrt(my_custom_type<X>);

    // allow HDF5 serialization access the internals
    friend class alps::hdf5::detail::get_pointer<my_custom_type>;

    private:
    T my_value;

    public:
    /// "Constituent" type (for archiving and MPI). The "content" of the object should be of the "constituent" type!
    typedef T my_constituent_type;

    // "Math" Scalar type: should behave as a scalar type, with arithmetics and conversion from scalar types.
    // (FIXME: it seems to have to be a C++ scalar, from the point of view of boost::is_scalar!)
    typedef T my_scalar_type; // happens to be the same as T in this implementation

    // // "Element type" (as a sequence, which it is not)
    // typedef my_custom_type my_element_type;

    // "Element type" (as a sequence)
    typedef T my_element_type;
    
    /// Add-assign operator with the same type at RHS (needed for mean calculation)
    my_custom_type& operator+=(const my_custom_type& rhs) {
        my_value += rhs.my_value;
        return *this;
    }
    
    /// Divide operator with scalar_type<my_custom_type> at RHS (needed for many calculations)
    my_custom_type operator/(const my_scalar_type& c) const {
        // throw std::logic_error("operator/(scalar_type): Not implemented");
        my_custom_type r=*this;
        r.my_value /= c;
        return r;
    }

  /// Multiply operator with scalar_type<my_custom_type> at RHS (needed for many calculations)
    /** WARNING: It is assumed that multiplication is an inverse of division, that is:
                custom_type val=...;
                T count=...;
                custom_type mean=val/count;
                assert(mean*count == val);
    */
    my_custom_type operator*(const my_scalar_type& c) const {
        // throw std::logic_error("operator*(scalar_type): Not implemented");
        my_custom_type r=*this;
        r.my_value *= c;
        return r;
    }

    /// Addition operator (needed for many calculations, must be consistent with +=)
    // FIXME: use boost operators, define via +=
    my_custom_type operator+(const my_custom_type& rhs) const {
        // throw std::runtime_error("operator+ is not implemented for this type");
        my_custom_type r=*this;
        r.my_value += rhs.my_value;
        return r;
    }

    /// Subtraction operator (needed for many calculations, must be consistent with +)
    my_custom_type operator-(const my_custom_type& rhs) const {
        // throw std::runtime_error("operator- is not implemented for this type");
        my_custom_type r=*this;
        r.my_value -= rhs.my_value;
        return r;
    }

    /// Multiplication operator (must be element-wise to make sense)
    my_custom_type operator*(const my_custom_type& rhs) const {
        // throw std::runtime_error("operator* is not implemented for this type");
        my_custom_type r=*this;
        r.my_value *= rhs.my_value;
        return r;
    }
    /// Division operator (must be element-wise to make sense)
    my_custom_type operator/(const my_custom_type& rhs) const {
        // throw std::runtime_error("operator/ is not implemented for this type");
        my_custom_type r=*this;
        r.my_value /= rhs.my_value;
        return r;
    }

    /// Unary minus (negation) operator.
    my_custom_type operator-() const {
        // throw std::runtime_error("unary operator- is not implemented for this type");
        my_custom_type r;
        r.my_value=-my_value;
        return r;
    }

    // /// Add-Assign operator with scalar
    // my_custom_type& operator+=(const my_scalar_type&) {
    //     throw std::runtime_error("operator+= is not implemented for this type");
    // }
    
    /// Add operator with scalar (semantics: adds scaled identity custom_type)
    my_custom_type operator+(const my_scalar_type& s) const {
        // throw std::runtime_error("operator+ is not implemented for this type");
        my_custom_type r=*this;
        r.my_value += s;
        return r;
    }
    
    /// Subtract operator with scalar (semantics: subtracts scaled identity custom_type)
    my_custom_type operator-(const my_scalar_type& s) const {
        // throw std::runtime_error("operator- is not implemented for this type");
        my_custom_type r=*this;
        r.my_value -= s;
        return r;
    }

    /// Right-division (divide scalar by *this)
    /** Used in operator/; member method to avoid quirks with friend templates */
    my_custom_type right_div(const my_scalar_type& lhs) const {
        my_custom_type r;
        r.my_value = lhs/my_value;
        return r;
    }

    /// Print to a stream
    /** Used in operator<<; member method to avoid quirks with friend templates */
    std::ostream& print(std::ostream& s) const {
        s << "[Custom type: value=" << my_value << "]";
        return s;
    }

    /// Generate infinite value
    static my_custom_type inf() {
        my_custom_type r;
        r.my_value = alps::numeric::inf<T>();
    }
};


/// Stream-output operator
template <typename T>
std::ostream& operator<<(std::ostream& s, const my_custom_type<T>& obj)
{
    return obj.print(s);
}

/// Right division operator. (Needed for error bars in trigonometric functions)
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
    // /// Declare that the type is not a sequence, despite a possible presence of value_type
    // template <typename T>
    // struct is_sequence< my_custom_type<T> > : public boost::false_type {};

    /// Declare that the type is a sequence, despite a possible absence of value_type
    template <typename T>
    struct is_sequence< my_custom_type<T> > : public boost::true_type {};

    // /// Declare the element type (must be the same as the enclosing type, because the latter is not a sequence)
    // template <typename T>
    // struct element_type< my_custom_type<T> > {
    //     typedef typename my_custom_type<T>::my_element_type type;
    // };
    
    /// Declare the element type
    template <typename T>
    struct element_type< my_custom_type<T> > {
        typedef typename my_custom_type<T>::my_element_type type;
    };
    
    
    namespace numeric {
        /// Setting "negative" values to zero (needed for autocorrelation). Already implemented by ALPSCore for sequences.
        /** FIXME: Has to be done before including "accumulators.hpp" */
        template <typename T>
        void set_negative_0(my_custom_type<T>& x)
        {
            throw std::logic_error("set_negative_0() value is not yet implemented for this type");
        }

        namespace {
            inline void not_implemented(const std::string& fname) {
                throw std::runtime_error("Function "+fname+"() is not implemented for this type.");
            }
        }

        /// Square root (needed for error bar calculations). Must be element-wise to make sense.
        template <typename T>
        my_custom_type<T> sqrt(my_custom_type<T> x) {
            using std::sqrt;
            x.my_value = sqrt(x.my_value);
            return x;
        }

        // a set of standard math functions for the custom type.
        /** FIXME: Has to be done before including "accumulators.hpp" */
        template <typename T> inline my_custom_type<T>  sin(my_custom_type<T>) { not_implemented("sin"); }
        template <typename T> inline my_custom_type<T>  cos(my_custom_type<T>) { not_implemented("cos"); }
        template <typename T> inline my_custom_type<T>  tan(my_custom_type<T>) { not_implemented("tan"); }
        template <typename T> inline my_custom_type<T> sinh(my_custom_type<T>) { not_implemented("sinh"); }
        template <typename T> inline my_custom_type<T> cosh(my_custom_type<T>) { not_implemented("cosh"); }
        template <typename T> inline my_custom_type<T> tanh(my_custom_type<T>) { not_implemented("tanh"); }
        template <typename T> inline my_custom_type<T> asin(my_custom_type<T>) { not_implemented("asin"); }
        template <typename T> inline my_custom_type<T> acos(my_custom_type<T>) { not_implemented("acos"); }
        template <typename T> inline my_custom_type<T> atan(my_custom_type<T>) { not_implemented("atan"); }
        template <typename T> inline my_custom_type<T>  abs(my_custom_type<T>) { not_implemented("abs"); }
        template <typename T> inline my_custom_type<T>  log(my_custom_type<T>) { not_implemented("log"); }
        template <typename T> inline my_custom_type<T> cbrt(my_custom_type<T>) { not_implemented("cbrt"); }

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
        /** FIXME: should better be called `numeric::scalar_type` */
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
                static typename my_custom_type<T>::my_constituent_type* apply(my_custom_type<T>& value) {
                    using alps::hdf5::get_pointer;
                    return get_pointer(value.my_value); // get_pointer(*value.data());
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
                throw std::invalid_argument("Can save only my_custom_type<continuous_type>");
            }
        }
        
    } // hdf5::

    namespace numeric {
        /// This must be specialized to give the notion of "infinity" (for "undefined" error bars)
        /** The type should be default-constructible and convertible to custom_type */
        template <typename T>
        struct inf< my_custom_type<T> > {
            operator my_custom_type<T>() const {
                // throw std::logic_error("The infinite value is not yet implemented for this type");
                return my_custom_type<T>::inf();
            }
        };
    
        
    } // numeric::

} // alps::


// Defines the notion of accumulator, instantiates templates defined above.
#include "alps/accumulators.hpp"


/****************** TEST SECTION ***********************/


template <template<typename> class A, typename T>
struct AccumulatorTypeGenerator {
    typedef A<double> dbl_accumulator_type;
    typedef A<T> accumulator_type;
};

template <typename G>
struct CustomTypeAccumulatorTest : public testing::Test {
    typedef typename G::accumulator_type acc_type;
    typedef typename acc_type::accumulator_type raw_acc_type;
    typedef typename alps::accumulators::value_type<raw_acc_type>::type value_type;
    typedef typename G::dbl_accumulator_type dbl_acc_type;
    
    alps::accumulators::accumulator_set aset;
    alps::accumulators::result_set* rset_p;

    // Ugly, but should work
    static const bool is_mean_acc=boost::is_same<alps::accumulators::MeanAccumulator<value_type>,
                                                 acc_type>::value;
    static const bool is_nobin_acc=boost::is_same<alps::accumulators::NoBinningAccumulator<value_type>,
                                                  acc_type>::value;
    
    CustomTypeAccumulatorTest() {
        aset << acc_type("data");
        aset << dbl_acc_type("scalar");
        aset["data"] << value_type();
        aset["scalar"] << 2.0;
        rset_p=new alps::accumulators::result_set(aset);
    }

    ~CustomTypeAccumulatorTest() {
        delete rset_p;
    }

    void TestH5ScalarType() {
        typedef typename alps::hdf5::scalar_type<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_constituent_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestNumScalarType() {
        typedef typename alps::numeric::scalar<value_type>::type stype;
        EXPECT_EQ(typeid(typename value_type::my_scalar_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestElementType() {
        typedef typename alps::element_type<value_type>::type stype;
        EXPECT_TRUE(alps::is_sequence<value_type>::value);
        EXPECT_EQ(typeid(typename value_type::my_element_type), typeid(stype)) << "type is: " << typeid(stype).name();
    }

    void TestMean() {
        (*rset_p)["data"].mean<value_type>();
    }

    void TestError() {
        if (is_mean_acc) return;
        (*rset_p)["data"].error<value_type>();
    }

    // void TestTau() {
    //     if (is_mean_acc || is_nobin_acc) return;
    //     aset["data"].extract<raw_acc_type>().autocorrelation();
    // }

    void TestScaleConst() {
        (*rset_p)["data"]*2;
    }

    void TestScale() {
        (*rset_p)["data"]*(*rset_p)["scalar"];
    }

    void TestAddConst() {
        (*rset_p)["data"]+2;
    }

    void TestAddEqConst() {
        (*rset_p)["data"]+=2;
    }

    void TestAdd() {
        (*rset_p)["data"]+(*rset_p)["data"];
    }

    void TestAddEq() {
        (*rset_p)["data"]+=(*rset_p)["data"];
    }

    void TestAddScalar() {
        (*rset_p)["data"]+(*rset_p)["scalar"];
    }

    void TestAddEqScalar() {
        (*rset_p)["data"]+=(*rset_p)["scalar"];
    }

};

typedef my_custom_type<double> dbl_custom_type;
typedef ::testing::Types<
    AccumulatorTypeGenerator<alps::accumulators::MeanAccumulator, dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::NoBinningAccumulator,dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::LogBinningAccumulator,dbl_custom_type>,
    AccumulatorTypeGenerator<alps::accumulators::FullBinningAccumulator,dbl_custom_type>
    > MyTypes;

TYPED_TEST_CASE(CustomTypeAccumulatorTest, MyTypes);

#define MAKE_TEST(_name_) TYPED_TEST(CustomTypeAccumulatorTest, _name_)  { this->TestFixture::_name_(); }

MAKE_TEST(TestH5ScalarType)
MAKE_TEST(TestNumScalarType)
MAKE_TEST(TestElementType)

MAKE_TEST(TestMean)
MAKE_TEST(TestError)
// MAKE_TEST(TestTau)

MAKE_TEST(TestScaleConst)
MAKE_TEST(TestScale)
MAKE_TEST(TestAddConst)
MAKE_TEST(TestAdd)
MAKE_TEST(TestAddScalar)
MAKE_TEST(TestAddEqConst)
MAKE_TEST(TestAddEq)
MAKE_TEST(TestAddEqScalar)

