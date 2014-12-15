/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_WRAPED_ACCUMULATORS_HPP
#define ALPS_WRAPED_ACCUMULATORS_HPP

#include <alps/config.hpp>

#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/archive.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/utilities/boost_mpi.hpp>
#endif

#include <alps/accumulators/wrapper_set.hpp>

#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <string>
#include <iostream>
#include <typeinfo>
#include <stdexcept>

#ifndef ALPS_ACCUMULATOR_VALUE_TYPES_SEQ
    #define ALPS_ACCUMULATOR_VALUE_TYPES_SEQ BOOST_PP_TUPLE_TO_SEQ(ALPS_ACCUMULATOR_VALUE_TYPES_SIZE, (ALPS_ACCUMULATOR_VALUE_TYPES))
#endif

namespace alps {
    namespace accumulators {

        class ALPS_DECL result_wrapper;
        class ALPS_DECL accumulator_wrapper;

        namespace wrapped {

            template<typename accumulator_type> class ALPS_DECL virtual_result_wrapper {
                public:

                    // default constructor
                    virtual_result_wrapper();

                    // constructor from raw accumulator
                    virtual_result_wrapper(result_wrapper * arg);

                    // copy constructor
                    virtual_result_wrapper(virtual_result_wrapper const & rhs);

                    // constructor from hdf5
                    virtual_result_wrapper(hdf5::archive & ar);

                    virtual ~virtual_result_wrapper();


                // // default constructor
                //     result_wrapper() 
                //         : m_variant()
                //     {}

                // // constructor from raw result
                //     template<typename T> result_wrapper(T arg)
                //         : m_variant(typename detail::add_base_wrapper_pointer<typename value_type<T>::type>::type(
                //             new derived_result_wrapper<T>(arg))
                //           )
                //     {}

                // // constructor from base_wrapper
                //     template<typename T> result_wrapper(base_wrapper<T> * arg)
                //         : m_variant(typename detail::add_base_wrapper_pointer<T>::type(arg))
                //     {}

            //     // copy constructor
            //     private:
            //         struct copy_visitor: public boost::static_visitor<> {
            //             copy_visitor(detail::variant_type & s): self(s) {}
            //             template<typename T> void operator()(T const & arg) const {
            //                 const_cast<detail::variant_type &>(self) = T(arg->clone());
            //             }
            //             detail::variant_type & self;
            //         };
            //     public:
            //         result_wrapper(result_wrapper const & rhs)
            //             : m_variant()
            //         {
            //             copy_visitor visitor(m_variant);
            //             boost::apply_visitor(visitor, rhs.m_variant);
            //         }

            //     // constructor from hdf5
            //         result_wrapper(hdf5::archive & ar) {
            //             ar[""] >> *this;
            //         }

            //     // operator=
            //     private:
            //         struct assign_visitor: public boost::static_visitor<> {
            //             assign_visitor(result_wrapper * s): self(s) {}
            //             template<typename T> void operator()(T & arg) const {
            //                 self->m_variant = arg;
            //             }
            //             mutable result_wrapper * self;
            //         };
            //     public:
            //         result_wrapper & operator=(boost::shared_ptr<result_wrapper> const & rhs) {
            //             boost::apply_visitor(assign_visitor(this), rhs->m_variant);
            //             return *this;
            //         }

            //     // get
            //     private:
            //         template<typename T> struct get_visitor: public boost::static_visitor<> {
            //             template<typename X> void operator()(X const & arg) {
            //                 throw std::runtime_error(std::string("Canot cast observable") + typeid(X).name() + " to base type: " + typeid(T).name() + ALPS_STACKTRACE);
            //             }
            //             void operator()(typename detail::add_base_wrapper_pointer<T>::type const & arg) { value = arg; }
            //             typename detail::add_base_wrapper_pointer<T>::type value;
            //         };
            //     public:
            //         template <typename T> base_wrapper<T> & get() {
            //             get_visitor<T> visitor;
            //             boost::apply_visitor(visitor, m_variant);
            //             return *visitor.value;
            //         }

            //     // extract
            //     private:
            //         template<typename A> struct extract_visitor: public boost::static_visitor<> {
            //             template<typename T> void operator()(T const & arg) { value = &arg->template extract<A>(); }
            //             A * value;
            //         };
            //     public:
            //         template <typename A> A & extract() {
            //             extract_visitor<A> visitor;
            //             boost::apply_visitor(visitor, m_variant);
            //             return *visitor.value;
            //         }
            //         template <typename A> A const & extract() const {
            //             extract_visitor<A> visitor;
            //             boost::apply_visitor(visitor, m_variant);
            //             return *visitor.value;
            //         }

                    // count
                    boost::uint64_t count() const;

                private:

                    #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)  \
                        T mean_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_MEAN_IMPL

                    #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)  \
                        T error_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_ERROR_IMPL

                public:

                    // mean
                    template<typename T> T mean() const {
                        return mean_impl(T());
                    }

                    // error
                    template<typename T> T error() const {
                        return mean_impl(T());
                    }

                    // save
                    void save(hdf5::archive & ar) const;

                    // load
                    void load(hdf5::archive & ar);

                    // print
                    void print(std::ostream & os) const;


            //     // transform(T F(T))
            //     private:
            //         template<typename T> struct transform_1_visitor: public boost::static_visitor<> {
            //             transform_1_visitor(boost::function<T(T)> f) : op(f) {}
            //             template<typename X> void apply(typename boost::enable_if<
            //                 typename detail::is_valid_argument<T, typename value_type<X>::type>::type, X &
            //             >::type arg) const {
            //                 arg.transform(op);
            //             }
            //             template<typename X> void apply(typename boost::disable_if<
            //                 typename detail::is_valid_argument<T, typename value_type<X>::type>::type, X &
            //             >::type arg) const {
            //                 throw std::logic_error(std::string("cannot convert: ") + typeid(T).name() + " to " + typeid(typename value_type<X>::type).name() + ALPS_STACKTRACE);
            //             }
            //             template<typename X> void operator()(X & arg) const {
            //                 apply<typename X::element_type>(*arg);
            //             }
            //             boost::function<T(T)> op;
            //         };
            //     public:
            //         template<typename T> result_wrapper transform(boost::function<T(T)> op) const {
            //             result_wrapper clone(*this);
            //             boost::apply_visitor(transform_1_visitor<T>(op), clone.m_variant);
            //             return clone;
            //         }
            //         template<typename T> result_wrapper transform(T(*op)(T)) const {
            //             return transform(boost::function<T(T)>(op));
            //         }

            //     // unary plus
            //     public:
            //         result_wrapper operator+ () const {
            //             return result_wrapper(*this);
            //         }

            //     // unary minus
            //     private:
            //         struct unary_add_visitor: public boost::static_visitor<> {
            //             template<typename X> void operator()(X & arg) const {
            //                 arg->negate();
            //             }
            //         };
            //     public:
            //         result_wrapper operator- () const {
            //             result_wrapper clone(*this);
            //             unary_add_visitor visitor;
            //             boost::apply_visitor(visitor, clone.m_variant);
            //             return clone;
            //         }

            //     // operators
            //     #define ALPS_ACCUMULATOR_OPERATOR_PROXY(OPNAME, AUGOPNAME, AUGOP, FUN)                                  \
            //         private:                                                                                            \
            //             template<typename T> struct FUN ## _arg_visitor: public boost::static_visitor<> {               \
            //                 FUN ## _arg_visitor(T & v): value(v) {}                                                     \
            //                 template<typename X> void apply(X const &) const {                                          \
            //                     throw std::logic_error("only results with equal value types are allowed in operators"   \
            //                         + ALPS_STACKTRACE);                                                                 \
            //                 }                                                                                           \
            //                 void apply(T const & arg) const {                                                           \
            //                     const_cast<T &>(value) AUGOP arg;                                                       \
            //                 }                                                                                           \
            //                 template<typename X> void operator()(X const & arg) const {                                 \
            //                     apply(*arg);                                                                            \
            //                 }                                                                                           \
            //                 T & value;                                                                                  \
            //             };                                                                                              \
            //             struct FUN ## _self_visitor: public boost::static_visitor<> {                                   \
            //                 FUN ## _self_visitor(result_wrapper const & v): value(v) {}                                 \
            //                 template<typename X> void operator()(X & self) const {                                      \
            //                     FUN ## _arg_visitor<typename X::element_type> visitor(*self);                           \
            //                     boost::apply_visitor(visitor, value.m_variant);                                         \
            //                 }                                                                                           \
            //                 result_wrapper const & value;                                                               \
            //             };                                                                                              \
            //             struct FUN ## _double_visitor: public boost::static_visitor<> {                                 \
            //                 FUN ## _double_visitor(double v): value(v) {}                                               \
            //                 template<typename X> void operator()(X & arg) const {                                       \
            //                     *arg AUGOP value;                                                                       \
            //                 }                                                                                           \
            //                 double value;                                                                               \
            //             };                                                                                              \
            //         public:                                                                                             \
            //             result_wrapper & AUGOPNAME (result_wrapper const & arg) {                                       \
            //                 FUN ## _self_visitor visitor(arg);                                                          \
            //                 boost::apply_visitor(visitor, m_variant);                                                   \
            //                 return *this;                                                                               \
            //             }                                                                                               \
            //             result_wrapper & AUGOPNAME (double arg) {                                                       \
            //                 FUN ## _double_visitor visitor(arg);                                                        \
            //                 boost::apply_visitor(visitor, m_variant);                                                   \
            //                 return *this;                                                                               \
            //             }                                                                                               \
            //             result_wrapper OPNAME (result_wrapper const & arg) const {                                      \
            //                 result_wrapper clone(*this);                                                                \
            //                 clone AUGOP arg;                                                                            \
            //                 return clone;                                                                               \
            //             }                                                                                               \
            //             result_wrapper OPNAME (double arg) const {                                                      \
            //                 result_wrapper clone(*this);                                                                \
            //                 clone AUGOP arg;                                                                            \
            //                 return clone;                                                                               \
            //             }
            //     ALPS_ACCUMULATOR_OPERATOR_PROXY(operator+, operator+=, +=, add)
            //     ALPS_ACCUMULATOR_OPERATOR_PROXY(operator-, operator-=, -=, sub)
            //     ALPS_ACCUMULATOR_OPERATOR_PROXY(operator*, operator*=, *=, mul)
            //     ALPS_ACCUMULATOR_OPERATOR_PROXY(operator/, operator/=, /=, div)
            //     #undef ALPS_ACCUMULATOR_OPERATOR_PROXY

            //     // inverse
            //     private:
            //         struct inverse_visitor: public boost::static_visitor<> {
            //             template<typename T> void operator()(T & arg) const { arg->inverse(); }
            //         };
            //     public:
            //         result_wrapper inverse() const {
            //             result_wrapper clone(*this);
            //             boost::apply_visitor(inverse_visitor(), m_variant);
            //             return clone;
            //         }

            //     #define ALPS_ACCUMULATOR_FUNCTION_PROXY(FUN)                                \
            //         private:                                                                \
            //             struct FUN ## _visitor: public boost::static_visitor<> {            \
            //                 template<typename T> void operator()(T & arg) const {           \
            //                     arg-> FUN ();                                               \
            //                 }                                                               \
            //             };                                                                  \
            //         public:                                                                 \
            //             result_wrapper FUN () const {                                       \
            //                 result_wrapper clone(*this);                                    \
            //                 boost::apply_visitor( FUN ## _visitor(), clone.m_variant);      \
            //                 return clone;                                                   \
            //             }
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(sin)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(cos)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(tan)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(sinh)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(cosh)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(tanh)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(asin)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(acos)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(atan)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(abs)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(sqrt)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(log)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(sq)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(cb)
            //     ALPS_ACCUMULATOR_FUNCTION_PROXY(cbrt)
            //     #undef ALPS_ACCUMULATOR_FUNCTION_PROXY

                private:

                    std::ptrdiff_t * m_cnt;
                    result_wrapper * m_ptr;
            };
            // inline result_wrapper operator+(double arg1, result_wrapper const & arg2) {
            //     return arg2 + arg1;
            // }
            // inline result_wrapper operator-(double arg1, result_wrapper const & arg2) {
            //     return -arg2 + arg1;
            // }
            // inline result_wrapper operator*(double arg1, result_wrapper const & arg2) {
            //     return arg2 * arg1;
            // }
            // inline result_wrapper operator/(double arg1, result_wrapper const & arg2) {
            //     return arg2.inverse() * arg1;
            // }

            // inline std::ostream & operator<<(std::ostream & os, const result_wrapper & arg) {
            //     arg.print(os);
            //     return os;
            // }

            // template <typename A> A & extract(result_wrapper & m) {
            //     return m.extract<A>();
            // }

            // #define EXTERNAL_FUNCTION(FUN)                          \
            //     result_wrapper FUN (result_wrapper const & arg);

            //     EXTERNAL_FUNCTION(sin)
            //     EXTERNAL_FUNCTION(cos)
            //     EXTERNAL_FUNCTION(tan)
            //     EXTERNAL_FUNCTION(sinh)
            //     EXTERNAL_FUNCTION(cosh)
            //     EXTERNAL_FUNCTION(tanh)
            //     EXTERNAL_FUNCTION(asin)
            //     EXTERNAL_FUNCTION(acos)
            //     EXTERNAL_FUNCTION(atan)
            //     EXTERNAL_FUNCTION(abs)
            //     EXTERNAL_FUNCTION(sqrt)
            //     EXTERNAL_FUNCTION(log)
            //     EXTERNAL_FUNCTION(sq)
            //     EXTERNAL_FUNCTION(cb)
            //     EXTERNAL_FUNCTION(cbrt)

            // #undef EXTERNAL_FUNCTION


            class ALPS_DECL virtual_accumulator_wrapper {
                public:
                    // default constructor
                    virtual_accumulator_wrapper();

                    // constructor from raw accumulator
                    virtual_accumulator_wrapper(accumulator_wrapper * arg);

                    // copy constructor
                    virtual_accumulator_wrapper(virtual_accumulator_wrapper const & rhs);

                    // constructor from hdf5
                    virtual_accumulator_wrapper(hdf5::archive & ar);

                    virtual ~virtual_accumulator_wrapper();

                    // operator(), operator<<
                    #define ALPS_ACCUMULATOR_OPERATOR_CALL(r, data, T)              \
                        virtual_accumulator_wrapper & operator()(T const & value);  \
                        virtual_accumulator_wrapper & operator<<(T const & value) { \
                            (*this)(value);                                         \
                            return (*this);                                         \
                        }
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_OPERATOR_CALL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_OPERATOR_CALL

                    /// Merge another accumulator into this one. @param rhs  accumulator to merge.
                    void merge(const virtual_accumulator_wrapper & rhs);

                    virtual_accumulator_wrapper & operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs);

                    // get
                    // template <typename T> base_wrapper<T> & get() {
                    //     get_visitor<T> visitor;
                    //     boost::apply_visitor(visitor, m_variant);
                    //     return *visitor.value;
                    // }

                    // extract
                    // template <typename A> A & extract() {
                    //     throw std::logic_error(std::string("unknown type : ") + typeid(A).name() + ALPS_STACKTRACE);
                    // }
                    // template <> MeanAccumulatorDouble & extract<MeanAccumulatorDouble>();

                    // count
                    boost::uint64_t count() const;

                private:

                    #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)  \
                        T mean_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_MEAN_IMPL

                    #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)  \
                        T error_impl(T) const;
                    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
                    #undef ALPS_ACCUMULATOR_ERROR_IMPL

                public:

                    // mean
                    template<typename T> T mean() const {
                        return mean_impl(T());
                    }

                    // error
                    template<typename T> T error() const {
                        return mean_impl(T());
                    }

                    // save
                    void save(hdf5::archive & ar) const;

                    // load
                    void load(hdf5::archive & ar);

                    // reset
                    void reset() const;

                    // result
                    boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> > result() const;

                    // print
                    void print(std::ostream & os) const;

#ifdef ALPS_HAVE_MPI
                    // collective_merge
                    void collective_merge(boost::mpi::communicator const & comm, int root);
                    void collective_merge(boost::mpi::communicator const & comm, int root) const;
#endif

                private:

                    std::ptrdiff_t * m_cnt;
                    accumulator_wrapper * m_ptr;
            };

            inline std::ostream & operator<<(std::ostream & os, const virtual_accumulator_wrapper & arg) {
                arg.print(os);
                return os;
            }

            // template <typename A> A & extract(virtual_accumulator_wrapper & m) {
            //     return m.extract<A>();
            // }

            inline void ALPS_DECL reset(virtual_accumulator_wrapper & arg) {
                return arg.reset();
            }

        }
    }

    // TODO: take type from variant type
    template<typename T> struct MeanAccumulator {
        public:
            MeanAccumulator(std::string const & name): m_name(name) {}
            std::string const & name() const { return m_name; }
        private:
            std::string m_name;
    };

    typedef accumulators::impl::wrapper_set<accumulators::wrapped::virtual_accumulator_wrapper> accumulator_set;
    typedef accumulators::impl::wrapper_set<accumulators::wrapped::virtual_result_wrapper<accumulators::wrapped::virtual_accumulator_wrapper> > result_set;

    #define ALPS_ACCUMULATOR_ADD_ACCUMULATOR(r, data, T)  \
        accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator< T > & arg);
    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ADD_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
    #undef ALPS_ACCUMULATOR_ADD_ACCUMULATOR
}

#endif