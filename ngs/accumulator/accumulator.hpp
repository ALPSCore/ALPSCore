/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2011 - 2013 by Mario Koenz <mkoenz@ethz.ch>                       *
 *                              Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_ACCUMULATOR_ACCUMULATOR_HPP
#define ALPS_NGS_ACCUMULATOR_ACCUMULATOR_HPP

#include <alps/ngs/accumulator/wrappers.hpp>
#include <alps/ngs/accumulator/feature/weight_impl.hpp>

// TODO: move inside features
#include <alps/type_traits/covariance_type.hpp>

#include <alps/hdf5/archive.hpp>

#include <boost/shared_ptr.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <typeinfo>
#include <stdexcept>

namespace alps {
    namespace accumulator {

        // TODO: merge with accumulator_wrapper
        class ALPS_DECL result_wrapper {
            public:
                result_wrapper() 
                    : m_base()
                {}

                template<typename T> result_wrapper(T arg)
                    : m_base(new derived_result_wrapper<T>(arg))
                {}

                result_wrapper(base_wrapper * arg)
                    : m_base(arg) 
                {}

                result_wrapper(result_wrapper const & arg)
                    : m_base(arg.m_base->clone()) 
                {}

                result_wrapper(hdf5::archive & ar) {
                    ar[""] >> *this;
                }

                template<typename T> void operator()(T const & value) {
                    (*m_base)(&value, typeid(T));
                }
                template<typename T> result_wrapper & operator<<(T const & value) {
                    (*this)(value);
                    return (*this);
                }

                template<typename T, typename W> void operator()(T const & value, W const & weight) {
                    (*m_base)(&value, typeid(T), &weight, typeid(W));
                }

                result_wrapper & operator=(boost::shared_ptr<result_wrapper> const & ptr) {
                    m_base = ptr->m_base;
                    return *this;
                }

                template<typename T> result_type_wrapper<T> const & get() const {
                    return m_base->get<T>();
                }

                template <typename A> A & extract() {
                    return m_base->extract<A>();
                }

                boost::uint64_t count() const {
                    return m_base->count();
                }

                // TODO: add all member functions
                template<typename T> typename mean_type<result_type_wrapper<T> >::type mean() const { return get<T>().mean(); }
                template<typename T> typename error_type<result_type_wrapper<T> >::type error() const { return get<T>().error(); }
                template<typename T> typename covariance_type<T>::type accurate_covariance(result_wrapper const & rhs) const { return typename covariance_type<T>::type(); } // TODO: implement!
                template<typename T> typename covariance_type<T>::type covariance(result_wrapper const & rhs) const { return typename covariance_type<T>::type(); } // TODO: implement!

                void save(hdf5::archive & ar) const {
                    ar[""] = *m_base;
                }

                void load(hdf5::archive & ar) {
                    ar[""] >> *m_base;
                }

                void print(std::ostream & os) const {
                    m_base->print(os);
                }

                #define OPERATOR_PROXY(OPNAME, AUGOPNAME, AUGOP)                        \
                    result_wrapper & AUGOPNAME (result_wrapper const & arg) {           \
                        *this->m_base AUGOP *arg.m_base;                                \
                        return *this;                                                   \
                    }                                                                   \
                    result_wrapper OPNAME (result_wrapper const & arg) {                \
                        result_wrapper clone(*this);                                    \
                        clone AUGOP arg;                                                \
                        return clone;                                                   \
                    }                                                                   \
                    result_wrapper & AUGOPNAME (double arg) {                           \
                        *this->m_base AUGOP arg;                                        \
                        return *this;                                                   \
                    }                                                                   \
                    result_wrapper OPNAME (double arg) {                                \
                        result_wrapper clone(*this);                                    \
                        clone AUGOP arg;                                                \
                        return clone;                                                   \
                    }
                OPERATOR_PROXY(operator+, operator+=, +=)
                OPERATOR_PROXY(operator-, operator-=, -=)
                OPERATOR_PROXY(operator*, operator*=, *=)
                OPERATOR_PROXY(operator/, operator/=, /=)
                #undef OPERATOR_PROXY

                result_wrapper inverse() {
                    result_wrapper clone(*this);
                    clone.inverse();
                    return clone;
                }

                #define FUNCTION_PROXY(FUN)            \
                    result_wrapper FUN () const {      \
                        result_wrapper clone(*this);   \
                        clone.m_base-> FUN ();         \
                        return clone;                  \
                    }
                FUNCTION_PROXY(sin)
                FUNCTION_PROXY(cos)
                FUNCTION_PROXY(tan)
                FUNCTION_PROXY(sinh)
                FUNCTION_PROXY(cosh)
                FUNCTION_PROXY(tanh)
                FUNCTION_PROXY(asin)
                FUNCTION_PROXY(acos)
                FUNCTION_PROXY(atan)
                FUNCTION_PROXY(abs)
                FUNCTION_PROXY(sqrt)
                FUNCTION_PROXY(log)
                FUNCTION_PROXY(sq)
                FUNCTION_PROXY(cb)
                FUNCTION_PROXY(cbrt)
                #undef FUNCTION_PROXY

            private:
                boost::shared_ptr<base_wrapper> m_base;
        };

        inline std::ostream & operator<<(std::ostream & os, const result_wrapper & arg) {
            arg.print(os);
            return os;
        }

        template <typename A> A & extract(result_wrapper & m) {
            return m.extract<A>();
        }

        #define EXTERNAL_FUNCTION(FUN)                          \
            result_wrapper FUN (result_wrapper const & arg);

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

        class ALPS_DECL accumulator_wrapper {
            public:
                accumulator_wrapper() 
                    : m_base()
                {}

                template<typename T> accumulator_wrapper(T arg)
                    : m_base(new derived_accumulator_wrapper<T>(arg))
                {}

                accumulator_wrapper(accumulator_wrapper const & arg)
                    : m_base(arg.m_base->clone())
                {}

                accumulator_wrapper(hdf5::archive & ar) {
                    ar[""] >> *this;
                }

                template<typename T> void operator()(T const & value) {
                    (*m_base)(&value, typeid(T));
                }
                template<typename T> accumulator_wrapper & operator<<(T const & value) {
                    (*this)(value);
                    return (*this);
                }
                template<typename T, typename W> void operator()(T const & value, W const & weight) {
                    (*m_base)(&value, typeid(T), &weight, typeid(W));
                }                

                accumulator_wrapper & operator=(boost::shared_ptr<accumulator_wrapper> const & ptr) {
                    m_base = ptr->m_base;
                    return *this;
                }

                template<typename T> result_type_wrapper<T> const & get() const {
                    return m_base->get<T>();
                }

                template <typename A> A & extract() {
                    return m_base->extract<A>();
                }

                boost::uint64_t count() const {
                    return m_base->count();
                }

                // TODO: add all member functions
                template<typename T> typename mean_type<result_type_wrapper<T> >::type mean() const { return get<T>().mean(); }
                template<typename T> typename error_type<result_type_wrapper<T> >::type error() const { return get<T>().error(); }
                template<typename T> typename covariance_type<T>::type accurate_covariance(result_wrapper const & rhs) const { return typename covariance_type<T>::type(); } // TODO: implement!
                template<typename T> typename covariance_type<T>::type covariance(result_wrapper const & rhs) const { return typename covariance_type<T>::type(); } // TODO: implement!

                void save(hdf5::archive & ar) const {
                    ar[""] = *m_base;
                }

                void load(hdf5::archive & ar) {
                    ar[""] >> *m_base;
                }

                inline void reset() {
                    m_base->reset();
                }

                boost::shared_ptr<result_wrapper> result() const {
                   return boost::shared_ptr<result_wrapper>(new result_wrapper(m_base->result()));
                }

                void print(std::ostream & os) const {
                    m_base->print(os);
                }

#ifdef ALPS_HAVE_MPI
                inline void collective_merge(
                      boost::mpi::communicator const & comm
                    , int root
                ) {
                    m_base->collective_merge(comm, root);
                }

                inline void collective_merge(
                      boost::mpi::communicator const & comm
                    , int root
                ) const {
                    m_base->collective_merge(comm, root);
                }
#endif

            private:

                boost::shared_ptr<base_wrapper> m_base;
        };

        inline std::ostream & operator<<(std::ostream & os, const accumulator_wrapper & arg) {
            arg.print(os);
            return os;
        }

        template <typename A> A & extract(accumulator_wrapper & m) {
            return m.extract<A>();
        }

        inline void ALPS_DECL reset(accumulator_wrapper & arg) {
            return arg.reset();
        }

        namespace detail {
            
            template<typename T> struct serializable_type {
                virtual ~serializable_type() {}
                virtual std::size_t rank() const = 0;
                virtual bool can_load(hdf5::archive & ar) const = 0;
                virtual T * create(hdf5::archive & ar) const = 0;
            };

            template<typename T, typename A> struct serializable_type_impl : public serializable_type<T> {
                std::size_t rank() const {
                    return A::rank();
                }
                bool can_load(hdf5::archive & ar) const {
                    return A::can_load(ar);
                }
                T * create(hdf5::archive & ar) const {
                    return new T(A());
                }
            };
        }

        namespace detail {

            inline void register_predefined_serializable_type();

        }

        namespace impl {

            template <typename T> class wrapper_set {

                public: 
                    typedef T value_type;

                    typedef typename std::map<std::string, boost::shared_ptr<T> >::iterator iterator;
                    typedef typename std::map<std::string, boost::shared_ptr<T> >::const_iterator const_iterator;

                    template <typename U> wrapper_set(wrapper_set<U> const & arg, typename boost::disable_if<boost::is_same<result_wrapper, U>, void *>::type = NULL) {
                        for (typename wrapper_set<U>::const_iterator it = arg.begin(); it != arg.end(); ++it)
                            insert(it->first, it->second->result());
                    }

                    wrapper_set() {
                        if (m_types.empty())
                            detail::register_predefined_serializable_type();
                    }
                    wrapper_set(wrapper_set const &) {} // TODO: how do we handle that?

                    T & operator[](std::string const & name) {
                        if (!has(name))
                            m_storage.insert(make_pair(name, boost::shared_ptr<T>(new T())));
                        return *(m_storage.find(name)->second);
                    }

                    T const & operator[](std::string const & name) const {
                        if (!has(name))
                            throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
                        return *(m_storage.find(name)->second);
                    }

                    bool has(std::string const & name) const{
                        return m_storage.find(name) != m_storage.end();
                    }
                    
                    void insert(std::string const & name, boost::shared_ptr<T> ptr){
                        if (has(name))
                            throw std::out_of_range("There exists alrady an accumulator with the name: " + name + ALPS_STACKTRACE);
                        m_storage.insert(make_pair(name, ptr));
                    }

                    std::size_t size() const {
                        return m_storage.size();
                    }

                    void save(hdf5::archive & ar) const {
                        for(const_iterator it = begin(); it != end(); ++it)
                            ar[it->first] = *(it->second);
                    }

                    void load(hdf5::archive & ar) {
                        std::vector<std::string> list = ar.list_children("");
                        for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                            ar.set_context(*it);
                            for (typename std::vector<boost::shared_ptr<detail::serializable_type<T> > >::const_iterator jt = m_types.begin()
                                ; jt != m_types.end()
                                ; ++jt
                            )
                                if ((*jt)->can_load(ar)) {
                                    operator[](*it) = boost::shared_ptr<T>((*jt)->create(ar));
                                    break;
                                }
                            if (!has(*it))
                                throw std::logic_error("The Accumulator/Result " + *it + " cannot be unserilized" + ALPS_STACKTRACE);
                            operator[](*it).load(ar);
                            ar.set_context("..");
                        }
                    }

                    template<typename A> static void register_serializable_type(bool known = false) {
                        if (!known && m_types.empty())
                            detail::register_predefined_serializable_type();
                        m_types.push_back(boost::shared_ptr<detail::serializable_type<T> >(new detail::serializable_type_impl<T, A>));
                        for (std::size_t i = m_types.size(); i > 1 && m_types[i - 1]->rank() > m_types[i - 2]->rank(); --i)
                            m_types[i - 1].swap(m_types[i - 2]);
                    }

                    void merge(wrapper_set const &) {}

                    void print(std::ostream & os) const {
                        for(const_iterator it = begin(); it != end(); ++it)
                            os << it->first << ": " << *(it->second) << std::endl;
                    }

                    void reset(bool=true /* deprecated flag */) { // TODO: Do we really want this flag?
                        for(iterator it = begin(); it != end(); ++it)
                            it->second->reset();
                    }
                    
                    iterator begin() { return m_storage.begin(); }
                    iterator end() { return m_storage.end(); }

                    const_iterator begin() const { return m_storage.begin(); }
                    const_iterator end() const { return m_storage.end(); }
                    
                    void clear() { m_storage.clear(); }

                private:
                    std::map<std::string, boost::shared_ptr<T> > m_storage;
                    static std::vector<boost::shared_ptr<detail::serializable_type<T> > > m_types;
            };
            template<typename T> std::vector<boost::shared_ptr<detail::serializable_type<T> > > wrapper_set<T>::m_types;

            template<typename T> inline std::ostream & operator<<(std::ostream & os, const wrapper_set<T> & arg) {
                arg.print(os);
                return os;
            }
        }
        typedef impl::wrapper_set<accumulator_wrapper> accumulator_set;
        typedef impl::wrapper_set<result_wrapper> result_set;

        // TODO: make this nicer ...
        namespace detail {
    
            template<typename T> struct PredefinedObservableBase {
                typedef T accumulator_type;
                typedef typename T::result_type result_type;

                template<typename ArgumentPack> PredefinedObservableBase(ArgumentPack const& args) 
                    : name(args[accumulator_name])
                    , wrapper(new accumulator_wrapper(T(args)))
                {}

                std::string name;
                boost::shared_ptr<accumulator_wrapper> wrapper;
            };


            template<typename T> struct PredefinedObservable : public PredefinedObservableBase<T> {
                BOOST_PARAMETER_CONSTRUCTOR(
                    PredefinedObservable, 
                    (PredefinedObservableBase<T>),
                    accumulator_keywords,
                        (required (_accumulator_name, (std::string)))
                        (optional 
                            (_max_bin_number, (std::size_t))
                        )
                )
            };

            template<typename T> inline accumulator_set & operator<<(accumulator_set & set, const PredefinedObservable<T> & arg) {
                set.insert(arg.name, arg.wrapper);
                return set;
            }

            template<typename T> struct simple_observable_type
                : public impl::Accumulator<T, error_tag, impl::Accumulator<T, mean_tag, impl::Accumulator<T, count_tag, impl::AccumulatorBase<T> > > >
            {
                simple_observable_type(): base_type() {}
                template<typename A> simple_observable_type(A const & arg): base_type(arg) {}
                private:
                    typedef impl::Accumulator<T, error_tag, impl::Accumulator<T, mean_tag, impl::Accumulator<T, count_tag, impl::AccumulatorBase<T> > > > base_type;
            };

            template<typename T> struct observable_type
                : public impl::Accumulator<T, autocorrelation_tag, impl::Accumulator<T, max_num_binning_tag, simple_observable_type<T> > >
            {
                observable_type(): base_type() {}
                template<typename A> observable_type(A const & arg): base_type(arg) {}
                private:
                    typedef impl::Accumulator<T, autocorrelation_tag, impl::Accumulator<T, max_num_binning_tag, simple_observable_type<T> > > base_type;
            };

            template<typename T> struct signed_observable_type
                : public impl::Accumulator<T, weight_holder_tag<simple_observable_type<T> >, observable_type<T> >
            {
                signed_observable_type(): base_type() {}
                template<typename A> signed_observable_type(A const & arg): base_type(arg) {}
                private:
                    typedef impl::Accumulator<T, weight_holder_tag<simple_observable_type<T> >, observable_type<T> > base_type;
            };

            template<typename T> struct signed_simple_observable_type
                : public impl::Accumulator<T, weight_holder_tag<simple_observable_type<T> >, simple_observable_type<T> >
            {
                signed_simple_observable_type(): base_type() {}
                template<typename A> signed_simple_observable_type(A const & arg): base_type(arg) {}
                private:
                    typedef impl::Accumulator<T, weight_holder_tag<simple_observable_type<T> >, simple_observable_type<T> > base_type;
            };
        }

        typedef detail::PredefinedObservable<detail::simple_observable_type<double> > SimpleRealObservable;
        typedef detail::PredefinedObservable<detail::simple_observable_type<std::vector<double> > > SimpleRealVectorObservable;

        typedef detail::PredefinedObservable<detail::observable_type<double> > RealObservable;
        typedef detail::PredefinedObservable<detail::observable_type<std::vector<double> > > RealVectorObservable;

        typedef detail::PredefinedObservable<detail::signed_observable_type<double> > SignedRealObservable;
        typedef detail::PredefinedObservable<detail::signed_observable_type<std::vector<double> > > SignedRealVectorObservable;

        typedef detail::PredefinedObservable<detail::signed_simple_observable_type<double> > SignedSimpleRealObservable;
        typedef detail::PredefinedObservable<detail::signed_simple_observable_type<std::vector<double> > > SignedSimpleRealVectorObservable;

        // TODO implement: RealTimeSeriesObservable

        namespace detail {

            inline void register_predefined_serializable_type() {
                accumulator_set::register_serializable_type<SimpleRealObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<SimpleRealVectorObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<RealObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<RealVectorObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<SignedRealObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<SignedRealVectorObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<SignedSimpleRealObservable::accumulator_type>(true);
                accumulator_set::register_serializable_type<SignedSimpleRealVectorObservable::accumulator_type>(true);

                result_set::register_serializable_type<SimpleRealObservable::result_type>(true);
                result_set::register_serializable_type<SimpleRealVectorObservable::result_type>(true);
                result_set::register_serializable_type<RealObservable::result_type>(true);
                result_set::register_serializable_type<RealVectorObservable::result_type>(true);
                result_set::register_serializable_type<SignedRealObservable::result_type>(true);
                result_set::register_serializable_type<SignedRealVectorObservable::result_type>(true);
                result_set::register_serializable_type<SignedSimpleRealObservable::result_type>(true);
                result_set::register_serializable_type<SignedSimpleRealVectorObservable::result_type>(true);
            }
        }
    }
}

 #endif