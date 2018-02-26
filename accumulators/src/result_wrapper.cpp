/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators/accumulator.hpp>
#include <sstream>

namespace alps {
    namespace accumulators {

        //
        // constructors
        //

        result_wrapper::result_wrapper()
            : m_variant()
        {}

        struct copy_visitor: public boost::static_visitor<> {
            copy_visitor(detail::variant_type & s): self(s) {}
            template<typename T> void operator()(T const & arg) const {
                const_cast<detail::variant_type &>(self) = T(arg->clone());
            }
            detail::variant_type & self;
        };
        result_wrapper::result_wrapper(result_wrapper const & rhs)
            : m_variant()
        {
            copy_visitor visitor(m_variant);
            boost::apply_visitor(visitor, rhs.m_variant);
        }
        result_wrapper::result_wrapper(hdf5::archive & ar) {
            ar[""] >> *this;
        }

        //
        // operator=
        //

        struct result_wrapper::assign_visitor: public boost::static_visitor<> {
            assign_visitor(result_wrapper * s): self(s) {}
            template<typename T> void operator()(T & arg) const {
                self->m_variant = arg;
            }
            mutable result_wrapper * self;
        };
        result_wrapper & result_wrapper::operator=(boost::shared_ptr<result_wrapper> const & rhs) {
            boost::apply_visitor(assign_visitor(this), rhs->m_variant);
            return *this;
        }

        //
        // count
        //

        struct count_visitor: public boost::static_visitor<boost::uint64_t> {
            template<typename T> boost::uint64_t operator()(T const & arg) const {
                return arg->count();
            }
        };
        boost::uint64_t result_wrapper::count() const {
            count_visitor visitor;
            return boost::apply_visitor(visitor, m_variant);
        }

        //
        // save
        //

        struct save_visitor: public boost::static_visitor<> {
            save_visitor(hdf5::archive & a): ar(a) {}
            template<typename T> void operator()(T & arg) const { ar[""] = *arg; }
            hdf5::archive & ar;
        };
        void result_wrapper::save(hdf5::archive & ar) const {
            boost::apply_visitor(save_visitor(ar), m_variant);
        }

        //
        // load
        //

        struct load_visitor: public boost::static_visitor<> {
            load_visitor(hdf5::archive & a): ar(a) {}
            template<typename T> void operator()(T const & arg) const { ar[""] >> *arg; }
            hdf5::archive & ar;
        };
        void result_wrapper::load(hdf5::archive & ar) {
            boost::apply_visitor(load_visitor(ar), m_variant);
        }

        //
        // print
        //

        struct print_visitor: public boost::static_visitor<> {
            print_visitor(std::ostream & o, bool t): os(o), terse(t) {}
            template<typename T> void operator()(T const & arg) const { arg->print(os, terse); }
            std::ostream & os;
            bool terse;
        };
        void result_wrapper::print(std::ostream & os, bool terse) const {
            boost::apply_visitor(print_visitor(os, terse), m_variant);
        }

        //
        // unary plus
        //

        result_wrapper result_wrapper::operator+ () const {
            return result_wrapper(*this);
        }

        //
        // unary minus
        //

        struct unary_add_visitor : public boost::static_visitor<> {
            template<typename X> void operator()(X & arg) const {
                arg->negate();
            }
        };
        result_wrapper result_wrapper::operator- () const {
            result_wrapper clone(*this);
            unary_add_visitor visitor;
            boost::apply_visitor(visitor, clone.m_variant);
            return clone;
        }

        //
        // Arithmetical operations
        //

        // Naming conventions:
        //   Operation is `lhs_var AUGOP rhs_var`, where AUGOP is `+=` , `-=` etc.
        //   lhsvar contains a variant over LHSPT types
        //   rhsvar contains a variant over RHSPT types
        //   LHSPT: lhs (pointer) type, which is shared_ptr<LHSWT>
        //   LHSWT: lhs (base_wrapper<...>) type
        //   RHSPT: rhs (pointer) type, which is shared_ptr<RHSWT>
        //   RHSWT: rhs (base_wrapper<...>) type
        #define ALPS_ACCUMULATOR_OPERATOR_PROXY(OPNAME, AUGOPNAME, AUGOP, FUN)                              \
            template<typename LHSWT> struct FUN ## _arg_visitor: public boost::static_visitor<> {           \
                FUN ## _arg_visitor(LHSWT & v): lhs_value(v) {}                                             \
                template<typename RHSWT>                                                                    \
                void apply(const RHSWT&,                                                                    \
                            typename std::enable_if<!detail::is_compatible_op<LHSWT,RHSWT>::value >::type* =0) const { \
                    throw std::logic_error("only results with compatible value types are allowed in operators"   \
                        + ALPS_STACKTRACE);                                                                 \
                }                                                                                           \
                template<typename RHSWT>                                                                    \
                void apply(const RHSWT& rhs_value,                                                          \
                            typename std::enable_if<detail::is_compatible_op<LHSWT,RHSWT>::value >::type* =0) { \
                    lhs_value AUGOP rhs_value;                                                              \
                }                                                                                           \
                void apply(LHSWT const & rhs_value) {                                                       \
                    lhs_value AUGOP rhs_value;                                                              \
                }                                                                                           \
                template<typename RHSPT> void operator()(RHSPT const & rhs_ptr) {                           \
                    apply(*rhs_ptr);                                                                        \
                }                                                                                           \
                LHSWT & lhs_value;                                                                          \
            };                                                                                              \
            struct result_wrapper:: FUN ## _self_visitor: public boost::static_visitor<> {                  \
                FUN ## _self_visitor(result_wrapper const & v): rhs_value(v) {}                             \
                template<typename LHSPT> void operator()(LHSPT & self) const {                              \
                    FUN ## _arg_visitor<typename LHSPT::element_type> visitor(*self);                       \
                    boost::apply_visitor(visitor, rhs_value.m_variant);                                     \
                }                                                                                           \
                result_wrapper const & rhs_value;                                                           \
            };                                                                                              \
            /** @brief Visitor to do AUGOP with a constant value */                                         \
            /* @note `long double` is chosen as the widest scalar numeric type */                           \
            /* @note No use of template: calls non-templatable virtual function. */                         \
            struct FUN ## _ldouble_visitor: public boost::static_visitor<> {                                \
                FUN ## _ldouble_visitor(long double v): value(v) {}                                         \
                template<typename X> void operator()(X & arg) const {                                       \
                    *arg AUGOP value;                                                                       \
                }                                                                                           \
                long double value;                                                                          \
            };                                                                                              \
            /** @brief Do AUGOP with another result   */                                                    \
            result_wrapper & result_wrapper:: AUGOPNAME (result_wrapper const & rhs) {                      \
                FUN ## _self_visitor visitor(rhs);                                                          \
                boost::apply_visitor(visitor, m_variant);                                                   \
                return *this;                                                                               \
            }                                                                                               \
            /** @brief Do AUGOP with a constant value */                                                    \
            result_wrapper & result_wrapper:: AUGOPNAME (long double arg) {                                 \
                FUN ## _ldouble_visitor visitor(arg);                                                       \
                boost::apply_visitor(visitor, m_variant);                                                   \
                return *this;                                                                               \
            }                                                                                               \
            result_wrapper result_wrapper:: OPNAME (result_wrapper const & arg) const {                     \
                result_wrapper clone(*this);                                                                \
                clone AUGOP arg;                                                                            \
                return clone;                                                                               \
            }                                                                                               \
            /** @brief Visitor to do OP with RHS constant value */                                          \
            result_wrapper result_wrapper:: OPNAME (long double arg) const {                                \
                result_wrapper clone(*this);                                                                \
                clone AUGOP arg;                                                                            \
                return clone;                                                                               \
            }
        ALPS_ACCUMULATOR_OPERATOR_PROXY(operator+, operator+=, +=, add)
        ALPS_ACCUMULATOR_OPERATOR_PROXY(operator-, operator-=, -=, sub)
        ALPS_ACCUMULATOR_OPERATOR_PROXY(operator*, operator*=, *=, mul)
        ALPS_ACCUMULATOR_OPERATOR_PROXY(operator/, operator/=, /=, div)
        #undef ALPS_ACCUMULATOR_OPERATOR_PROXY

        //
        // inverse
        //

        struct inverse_visitor: public boost::static_visitor<> {
            template<typename T> void operator()(T & arg) const { arg->inverse(); }
        };
        result_wrapper result_wrapper::inverse() const {
            result_wrapper clone(*this);
            boost::apply_visitor(inverse_visitor(), clone.m_variant);
            return clone;
        }

        //
        // Math functions
        //

        #define ALPS_ACCUMULATOR_FUNCTION_PROXY(FUN)                            \
            struct FUN ## _visitor: public boost::static_visitor<> {            \
                template<typename T> void operator()(T & arg) const {           \
                    arg-> FUN ();                                               \
                }                                                               \
            };                                                                  \
            result_wrapper result_wrapper:: FUN () const {                      \
                result_wrapper clone(*this);                                    \
                boost::apply_visitor( FUN ## _visitor(), clone.m_variant);      \
                return clone;                                                   \
            }
        ALPS_ACCUMULATOR_FUNCTION_PROXY(sin)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(cos)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(tan)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(sinh)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(cosh)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(tanh)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(asin)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(acos)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(atan)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(abs)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(sqrt)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(log)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(sq)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(cb)
        ALPS_ACCUMULATOR_FUNCTION_PROXY(cbrt)
        #undef ALPS_ACCUMULATOR_FUNCTION_PROXY

        //
        // Free functions
        //

#define EXTERNAL_FUNCTION(FUN)                                  \
        result_wrapper FUN (result_wrapper const & arg) {       \
            return arg. FUN ();                                 \
        }
        EXTERNAL_FUNCTION(sin)
        EXTERNAL_FUNCTION(cos)
        EXTERNAL_FUNCTION(tan)
        EXTERNAL_FUNCTION(sinh)
        EXTERNAL_FUNCTION(cosh)
        EXTERNAL_FUNCTION(tanh)
        EXTERNAL_FUNCTION(asin)
        EXTERNAL_FUNCTION(acos)
        EXTERNAL_FUNCTION(atan)
        EXTERNAL_FUNCTION(abs)
        EXTERNAL_FUNCTION(sqrt)
        EXTERNAL_FUNCTION(log)
        EXTERNAL_FUNCTION(sq)
        EXTERNAL_FUNCTION(cb)
        EXTERNAL_FUNCTION(cbrt)

#undef EXTERNAL_FUNCTION

        detail::printable_type short_print(const result_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,true);
            return ostr.str();
        }

        detail::printable_type full_print(const result_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,false);
            return ostr.str();
        }

        std::ostream & operator<<(std::ostream & os, const result_wrapper & arg) {
            arg.print(os, true); // terse printing by default
            return os;
        }

    }
}
