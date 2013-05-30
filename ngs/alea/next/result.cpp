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

#include <alps/ngs/alea/next/accumulator.hpp>
#include <alps/ngs/alea/next/feature/mean.hpp>
#include <alps/ngs/alea/next/feature/count.hpp>

#include <alps/hdf5.hpp>

#include <boost/shared_ptr.hpp>

#ifdef ALPS_HAVE_MPI
    #include <alps/ngs/boost_mpi.hpp>
#endif

#include <typeinfo>
#include <stdexcept>

namespace alps {
	namespace accumulator {

		template<typename T> struct result_type_wrapper;
		template<typename A> struct derived_wrapper;

		class base_wrapper : public impl::BaseWrapper<tag::mean, impl::BaseWrapper<tag::count, impl::Noop> > {

			public:
				virtual ~base_wrapper() {}

            	virtual void operator()(void const * value, std::type_info const & info) = 0;

                virtual void save(hdf5::archive & ar) const = 0;
                virtual void load(hdf5::archive & ar) = 0;

				virtual void print(std::ostream & os) const = 0;

				virtual base_wrapper* clone() const = 0;

                template<typename T> inline result_type_wrapper<T> const & get() const {
                    return dynamic_cast<result_type_wrapper<T> const &>(*this);
                }

                template<typename A> inline A & extract() {
                    return dynamic_cast<derived_wrapper<A>& >(*this).extract();
                }
		};

		template<typename T> struct result_type_wrapper : public impl::ResultTypeWrapper<T, tag::mean, impl::ResultTypeWrapper<T, tag::count, base_wrapper> > {};

		namespace detail {
			template<typename A> class accumulator_holder : public result_type_wrapper<typename value_type<A>::type> {

				public:
					accumulator_holder(A const & arg): m_acc(arg) {}

				protected:
					A m_acc;
			};
		}

		template<typename A> void add_value(A & acc, typename value_type<A>::type const & value) {
			acc(value);
		}

		template<typename A> struct derived_wrapper : public impl::DerivedWrapper<A, tag::mean, impl::DerivedWrapper<A, tag::count, detail::accumulator_holder<A> > > {
			public:
				derived_wrapper(): impl::DerivedWrapper<A, tag::mean, impl::DerivedWrapper<A, tag::count, detail::accumulator_holder<A> > >() {}
				derived_wrapper(A const & arg): impl::DerivedWrapper<A, tag::mean, impl::DerivedWrapper<A, tag::count, detail::accumulator_holder<A> > >(arg) {}

                inline A & extract()  {
                    return this->m_acc;
                }

	 			void operator()(void const * value, std::type_info const & info) {
		            if (&info != &typeid(typename value_type<A>::type) &&
			            #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
			                std::strcmp(info.name(), typeid(typename value_type<A>::type).name()) != 0
			            #else
			                info != typeid(typename value_type<A>::type)
			            #endif
		             )
		                throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
		            add_value(this->m_acc, *static_cast<typename value_type<A>::type const *>(value));
		        }

                void save(hdf5::archive & ar) const { ar[""] = this->m_acc; }
                void load(hdf5::archive & ar) { ar[""] >> this->m_acc; }

				void print(std::ostream & os) const {
                    this->m_acc.print(os);
                }

				base_wrapper * clone() const { 
					return new derived_wrapper<A>(this->m_acc); 
				}
		};

        class result_wrapper {
            public:
                template<typename T> result_wrapper(T arg) 
                	: m_base(new derived_wrapper<T>(arg)) 
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

                template<typename T> result_type_wrapper<T> const & get() const {
                    return m_base->get<T>();
                }

                template <typename A> A & extract() {
                    return m_base->extract<A>();
                }

                boost::uint64_t count() const {
                    return m_base->count();
                }

                void save(hdf5::archive & ar) const {
                    ar[""] = *m_base;
                }

                void load(hdf5::archive & ar) {
                	// TODO: make logic to find right accumulator:
                	impl::Accumulator<double, tag::mean, impl::Accumulator<double, tag::count, impl::Noop> > acc;
                	m_base =  boost::shared_ptr<base_wrapper>(new derived_wrapper<impl::Accumulator<double, tag::mean, impl::Accumulator<double, tag::count, impl::Noop> > >(acc));
                    ar[""] >> *m_base;
                }

				void print(std::ostream & os) const {
                    m_base->print(os);
                }

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

	    template <typename A> void reset(A & arg) {
	        return arg.reset();
	    }

        class accumulator_set {

            public: 
                typedef std::map<std::string, boost::shared_ptr<accumulator_wrapper> >::iterator iterator;
                typedef std::map<std::string, boost::shared_ptr<accumulator_wrapper> >::const_iterator const_iterator;

		        accumulator_wrapper & operator[](std::string const & name) {
		            if (!has(name))
		                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
		            return *(m_storage.find(name)->second);
		        }

		        accumulator_wrapper const & operator[](std::string const & name) const {
		            if (!has(name))
		                throw std::out_of_range("No observable found with the name: " + name + ALPS_STACKTRACE);
		            return *(m_storage.find(name)->second);
		        }

		        bool has(std::string const & name) const{
		            return m_storage.find(name) != m_storage.end();
		        }
		        
		        void insert(std::string const & name, boost::shared_ptr<accumulator_wrapper> ptr){
		            if (has(name))
		                throw std::out_of_range("There exists alrady an accumulator with the name: " + name + ALPS_STACKTRACE);
		            m_storage.insert(make_pair(name, ptr));
		        }

		        void save(hdf5::archive & ar) const {
		            for(const_iterator it = begin(); it != end(); ++it)
		            	ar[it->first] = *(it->second);
		        }

		        void load(hdf5::archive & ar) {
				    std::vector<std::string> list = ar.list_children("");
				    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
				    	ar.set_context(*it);
				    	insert(*it, boost::shared_ptr<accumulator_wrapper>(new accumulator_wrapper(ar)));
				    	ar.set_context("..");
				    }
		        }

		        void merge(accumulator_set const &) {}

		        void print(std::ostream & os) const {
					for(const_iterator it = begin(); it != end(); ++it)
						os << it->first << ": " << *(it->second) << std::endl;
		        }

		        void reset() {
		            for(iterator it = begin(); it != end(); ++it)
		                it->second->reset();
		        }
		        
		        iterator begin() { return m_storage.begin(); }
		        iterator end() { return m_storage.end(); }

		        const_iterator begin() const { return m_storage.begin(); }
		        const_iterator end() const { return m_storage.end(); }
		        
		        void clear() { m_storage.clear(); }

            private:
                std::map<std::string, boost::shared_ptr<accumulator_wrapper> > m_storage;
        };

        inline std::ostream & operator<<(std::ostream & os, const accumulator_set & arg) {
            arg.print(os);
            return os;
        }
	}
}

 #endif