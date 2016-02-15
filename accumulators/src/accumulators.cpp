/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators_.hpp>
#include <alps/accumulators/accumulator.hpp>
#include <alps/accumulators/namedaccumulators.hpp>


namespace alps {
    namespace accumulators {
        namespace wrapped {


            // default constructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper()
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(new result_wrapper()) 
            {}

            // constructor from raw accumulator
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(result_wrapper * arg)
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(arg) 
            {}

            // copy constructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(virtual_result_wrapper const & rhs)
                : m_cnt(rhs.m_cnt)
                , m_ptr(rhs.m_ptr)
            {
                ++(*m_cnt);
            }

            // constructor from hdf5
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::virtual_result_wrapper(hdf5::archive & ar)
                : m_cnt(new std::ptrdiff_t(1))
                , m_ptr(new result_wrapper(ar))
            {}

            // destructor
            template<> virtual_result_wrapper<virtual_accumulator_wrapper>::~virtual_result_wrapper() {
                if (!--(*m_cnt)) {
                    delete m_cnt;
                    delete m_ptr;
                }
            }

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
            template<> boost::uint64_t virtual_result_wrapper<virtual_accumulator_wrapper>::count() const{
                return m_ptr->count();
            }

            // mean
            #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)                                                      \
                template<> T virtual_result_wrapper<virtual_accumulator_wrapper>::mean_impl(T) const {          \
                    return m_ptr->mean<T>();                                                                    \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_MEAN_IMPL

            // error
            #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)                                                     \
                template<> T virtual_result_wrapper<virtual_accumulator_wrapper>::error_impl(T) const {         \
                    return m_ptr->error<T>();                                                                   \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_ERROR_IMPL

            // save
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::save(hdf5::archive & ar) const {
                m_ptr->save(ar);
            }

            // load
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::load(hdf5::archive & ar){
                m_ptr->load(ar);
            }

            // print
            template<> void virtual_result_wrapper<virtual_accumulator_wrapper>::print(std::ostream & os) const {
                m_ptr->print(os);
            }

/* ********
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
******** */



            // default constructor
			virtual_accumulator_wrapper::virtual_accumulator_wrapper()
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(new accumulator_wrapper()) 
			{}

            // constructor from raw accumulator
            virtual_accumulator_wrapper::virtual_accumulator_wrapper(accumulator_wrapper * arg)
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(arg) 
            {}

            // copy constructor
			virtual_accumulator_wrapper::virtual_accumulator_wrapper(virtual_accumulator_wrapper const & rhs)
				: m_cnt(rhs.m_cnt)
				, m_ptr(rhs.m_ptr)
			{
				++(*m_cnt);
			}

            // constructor from hdf5
			virtual_accumulator_wrapper::virtual_accumulator_wrapper(hdf5::archive & ar)
				: m_cnt(new std::ptrdiff_t(1))
				, m_ptr(new accumulator_wrapper(ar))
            {}

			// destructor
			virtual_accumulator_wrapper::~virtual_accumulator_wrapper() {
				if (!--(*m_cnt)) {
					delete m_cnt;
					delete m_ptr;
				}
            }

            // operator()
            #define ALPS_ACCUMULATOR_OPERATOR_CALL(r, data, T)                                              \
                virtual_accumulator_wrapper & virtual_accumulator_wrapper::operator()(T const & value) {    \
                    (*m_ptr)(value);                                                                        \
                    return (*this);                                                                         \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_OPERATOR_CALL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_OPERATOR_CALL

            /// Merge another accumulator into this one. @param rhs_acc  accumulator to merge.
			void virtual_accumulator_wrapper::merge(const virtual_accumulator_wrapper & rhs){
            	m_ptr->merge(*(rhs.m_ptr));
            }

            virtual_accumulator_wrapper & virtual_accumulator_wrapper::operator=(boost::shared_ptr<virtual_accumulator_wrapper> const & rhs){
            	(*m_ptr) = *(rhs->m_ptr);
            	return *this;
            }

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
            boost::uint64_t virtual_accumulator_wrapper::count() const{
            	return m_ptr->count();
            }

            // mean
            #define ALPS_ACCUMULATOR_MEAN_IMPL(r, data, T)                  \
                T virtual_accumulator_wrapper::mean_impl(T) const {         \
                    return m_ptr->mean<T>();                                \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_MEAN_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_MEAN_IMPL

            // error
            #define ALPS_ACCUMULATOR_ERROR_IMPL(r, data, T)                 \
                T virtual_accumulator_wrapper::error_impl(T) const {        \
                    return m_ptr->error<T>();                               \
                }
            BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ERROR_IMPL, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
            #undef ALPS_ACCUMULATOR_ERROR_IMPL

            // save
            void virtual_accumulator_wrapper::save(hdf5::archive & ar) const {
            	m_ptr->save(ar);
            }

            // load
            void virtual_accumulator_wrapper::load(hdf5::archive & ar){
            	m_ptr->load(ar);
            }

            // reset
            void virtual_accumulator_wrapper::reset() const {
            	m_ptr->reset();
            }

            // result
            boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> > virtual_accumulator_wrapper::result() const {
                return boost::shared_ptr<virtual_result_wrapper<virtual_accumulator_wrapper> >(
                    new virtual_result_wrapper<virtual_accumulator_wrapper>(new result_wrapper(*(m_ptr->result())))
                );
            }

            // print
            void virtual_accumulator_wrapper::print(std::ostream & os) const {
            	m_ptr->print(os);
            }

#ifdef ALPS_HAVE_MPI
            // collective_merge
            void virtual_accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) {
            	m_ptr->collective_merge(comm, root);
            }
            void virtual_accumulator_wrapper::collective_merge(alps::mpi::communicator const & comm, int root) const {
            	m_ptr->collective_merge(comm, root);
            }
#endif
		}
	}

    #define ALPS_ACCUMULATOR_ADD_ACCUMULATOR(r, type, T)                                                                \
        accumulator_set & operator<<(accumulator_set & set, const MeanAccumulator< T > & arg) {                         \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::mean_tag, accumulators::impl::Accumulator<                                    \
                            T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >                       \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const NoBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::error_tag, accumulators::impl::Accumulator<                                   \
                            T , accumulators::mean_tag, accumulators::impl::Accumulator<                                \
                                T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >                   \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const LogBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::binning_analysis_tag, accumulators::impl::Accumulator<                        \
                            T , accumulators::error_tag, accumulators::impl::Accumulator<                               \
                                T , accumulators::mean_tag, accumulators::impl::Accumulator<                            \
                                    T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >               \
                                >                                                                                       \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }                                                                                                               \
        accumulator_set & operator<<(accumulator_set & set, const FullBinningAccumulator< T > & arg) {                    \
            set.insert(arg.name(), boost::shared_ptr<accumulators::wrapped::virtual_accumulator_wrapper>(               \
                new accumulators::wrapped::virtual_accumulator_wrapper(new accumulators::accumulator_wrapper(           \
                    accumulators::impl::Accumulator<                                                                    \
                        T , accumulators::max_num_binning_tag, accumulators::impl::Accumulator<                         \
                            T , accumulators::binning_analysis_tag, accumulators::impl::Accumulator<                    \
                                T , accumulators::error_tag, accumulators::impl::Accumulator<                           \
                                    T , accumulators::mean_tag, accumulators::impl::Accumulator<                        \
                                        T , accumulators::count_tag, accumulators::impl::AccumulatorBase< T >           \
                                    >                                                                                   \
                                >                                                                                       \
                            >                                                                                           \
                        >                                                                                               \
                    >()                                                                                                 \
                )                                                                                                       \
            )));                                                                                                        \
            return set;                                                                                                 \
        }
    BOOST_PP_SEQ_FOR_EACH(ALPS_ACCUMULATOR_ADD_ACCUMULATOR, ~, ALPS_ACCUMULATOR_VALUE_TYPES_SEQ)
    #undef ALPS_ACCUMULATOR_ADD_ACCUMULATOR
}
