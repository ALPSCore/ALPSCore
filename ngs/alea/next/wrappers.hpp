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

#ifndef ALPS_NGS_ACCUMULATOR_WRAPPER_HPP
#define ALPS_NGS_ACCUMULATOR_WRAPPER_HPP

#include <alps/ngs/alea/next/feature/mean.hpp>
#include <alps/ngs/alea/next/feature/error.hpp>
#include <alps/ngs/alea/next/feature/count.hpp>
#include <alps/ngs/alea/next/feature/weight.hpp>
#include <alps/ngs/alea/next/feature/max_num_binning.hpp>

#include <alps/hdf5.hpp>

#include <typeinfo>
#include <stdexcept>

namespace alps {
	namespace accumulator {

		template<typename T> class result_type_wrapper;
		template<typename A> class derived_wrapper;

		// TODO: make for macro for that ...
		class base_wrapper : public 
			impl::BaseWrapper<tag::weight, 
			impl::BaseWrapper<tag::max_num_binning, 
			impl::BaseWrapper<tag::error, 
			impl::BaseWrapper<tag::mean, 
			impl::BaseWrapper<tag::count, 
		impl::BaseWrapper<void, void> > > > > > {
			public:
				virtual ~base_wrapper() {}

            	virtual void operator()(void const * value, std::type_info const & value_info) = 0;
            	virtual void operator()(void const * value, std::type_info const & value_info, void const * weight, std::type_info const & weight_info) = 0;

                virtual void save(hdf5::archive & ar) const = 0;
                virtual void load(hdf5::archive & ar) = 0;

				virtual void print(std::ostream & os) const = 0;
                virtual void reset() = 0;

				virtual base_wrapper* clone() const = 0;
				virtual base_wrapper* result() const = 0;

                template<typename T> result_type_wrapper<T> const & get() const {
                    return dynamic_cast<result_type_wrapper<T> const &>(*this);
                }

                template<typename A> A & extract() {
                    return dynamic_cast<derived_wrapper<A>& >(*this).extract();
                }
		};

		namespace detail {
			template<typename T, typename B> struct value_type_wrapper : public B {
				typedef T value_type;
			};
		}

		template<typename T> class result_type_wrapper : public 
			impl::ResultTypeWrapper<T, tag::weight, 
			impl::ResultTypeWrapper<T, tag::max_num_binning, 
			impl::ResultTypeWrapper<T, tag::error, 
			impl::ResultTypeWrapper<T, tag::mean, 
			impl::ResultTypeWrapper<T, tag::count, 
		detail::value_type_wrapper<T, base_wrapper> > > > > > {};

		namespace detail {
			template<typename A> class foundation_wrapper : public result_type_wrapper<typename value_type<A>::type> {

				public:
					foundation_wrapper(A const & arg): m_data(arg) {}

				protected:
					A m_data;
			};
		}

		template<typename T> void add_value(T & arg, typename value_type<T>::type const & value) {
			arg(value);
		}

		template<typename T> void add_value(T & arg, typename value_type<T>::type const & value, typename value_type<typename weight_type<T>::type>::type const & weight) {
			arg(value, weight);
		}

 		template<typename A> class derived_wrapper : public 
 			impl::DerivedWrapper<A, tag::weight, 
			impl::DerivedWrapper<A, tag::max_num_binning, 
 			impl::DerivedWrapper<A, tag::error, 
 			impl::DerivedWrapper<A, tag::mean, 
 			impl::DerivedWrapper<A, tag::count, 
 		detail::foundation_wrapper<A> > > > > > {
			public:
				derived_wrapper(): 
					impl::DerivedWrapper<A, tag::weight, 
					impl::DerivedWrapper<A, tag::max_num_binning, 
					impl::DerivedWrapper<A, tag::error, 
					impl::DerivedWrapper<A, tag::mean, 
					impl::DerivedWrapper<A, tag::count, 
				detail::foundation_wrapper<A> > > > > >() {}

				derived_wrapper(A const & arg): 
					impl::DerivedWrapper<A, tag::weight, 
					impl::DerivedWrapper<A, tag::max_num_binning, 
					impl::DerivedWrapper<A, tag::error, 
					impl::DerivedWrapper<A, tag::mean, 
					impl::DerivedWrapper<A, tag::count, 
				detail::foundation_wrapper<A> > > > > >(arg) {}

                A & extract()  {
                    return this->m_data;
                }

	 			void operator()(void const * value, std::type_info const & value_info) {
					return call_impl<A>(value, value_info);
		        }

	 			void operator()(void const * value, std::type_info const & value_info, void const * weight, std::type_info const & weight_info) {
					return call_impl<A>(value, value_info, weight, weight_info);
		        }

                void save(hdf5::archive & ar) const { 
                	ar[""] = this->m_data; 
               	}
                void load(hdf5::archive & ar) { 
                	ar[""] >> this->m_data; 
            	}

				void print(std::ostream & os) const {
                    this->m_data.print(os);
                }

                void reset() {
                    this->m_data.reset();
                }

				base_wrapper * clone() const { 
					return new derived_wrapper<A>(this->m_data); 
				}
				base_wrapper * result() const { 
					return result_impl<A>();
				}
			private:

				bool equal(std::type_info const & info1, std::type_info const & info2) const {
		            return (&info1 == &info2 ||
			            #ifdef BOOST_AUX_ANY_TYPE_ID_NAME
			                std::strcmp(info1.name(), info2.name()) == 0
			            #else
			                info1 == info2
			            #endif
		            );
				}

				template<typename T> typename boost::enable_if<typename boost::is_scalar<typename value_type<T>::type>::type>::type call_impl(
					void const * value, std::type_info const & value_info
				) {
					if (equal(value_info, typeid(char))) add_value(this->m_data, *static_cast<char const *>(value));
					else if (equal(value_info, typeid(signed char))) add_value(this->m_data, *static_cast<signed char const *>(value));
					else if (equal(value_info, typeid(unsigned char))) add_value(this->m_data, *static_cast<unsigned char const *>(value));
					else if (equal(value_info, typeid(short))) add_value(this->m_data, *static_cast<short const *>(value));
					else if (equal(value_info, typeid(unsigned short))) add_value(this->m_data, *static_cast<unsigned short const *>(value));
					else if (equal(value_info, typeid(int))) add_value(this->m_data, *static_cast<int const *>(value));
					else if (equal(value_info, typeid(unsigned))) add_value(this->m_data, *static_cast<unsigned const *>(value));
					else if (equal(value_info, typeid(long))) add_value(this->m_data, *static_cast<long const *>(value));
					else if (equal(value_info, typeid(unsigned long))) add_value(this->m_data, *static_cast<unsigned long const *>(value));
					else if (equal(value_info, typeid(long long))) add_value(this->m_data, *static_cast<long long const *>(value));
					else if (equal(value_info, typeid(unsigned long long))) add_value(this->m_data, *static_cast<unsigned long long const *>(value));
					else if (equal(value_info, typeid(float))) add_value(this->m_data, *static_cast<float const *>(value));
					else if (equal(value_info, typeid(double))) add_value(this->m_data, *static_cast<double const *>(value));
					else if (equal(value_info, typeid(long double))) add_value(this->m_data, *static_cast<long double const *>(value));
					else if (equal(value_info, typeid(bool))) add_value(this->m_data, *static_cast<bool const *>(value));
					else if (equal(value_info, typeid(typename value_type<A>::type))) add_value(this->m_data, *static_cast<typename value_type<A>::type const *>(value));
					else throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
				}
				template<typename T> typename boost::disable_if<typename boost::is_scalar<typename value_type<T>::type>::type>::type call_impl(
					void const * value, std::type_info const & value_info
				) {
					if (!equal(value_info, typeid(typename value_type<A>::type)))
		                throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
		            add_value(this->m_data, *static_cast<typename value_type<A>::type const *>(value));
				}

				template<typename T> typename boost::enable_if<typename has_feature<T, tag::weight>::type>::type call_impl(
					void const * value, std::type_info const & value_info, void const * weight, std::type_info const & weight_info
				) {
					if (!equal(value_info, typeid(typename value_type<T>::type)))
		                throw std::runtime_error("wrong value type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
					if (!equal(weight_info, typeid(typename value_type<typename weight_type<T>::type>::type)))
		                throw std::runtime_error("wrong weight type added in accumulator_wrapper::add_value" + ALPS_STACKTRACE);
		            add_value(this->m_data, *static_cast<typename value_type<T>::type const *>(value), *static_cast<typename value_type<typename weight_type<T>::type>::type const *>(weight));
				}
				template<typename T> typename boost::disable_if<typename has_feature<T, tag::weight>::type>::type call_impl(
					void const * value, std::type_info const & value_info, void const * weight, std::type_info const & weight_info
				) {
	                throw std::runtime_error(std::string("The type ") + typeid(T).name() + " has no weight" + ALPS_STACKTRACE);
				}

				template<typename T> typename boost::enable_if<typename has_result_type<T>::type, base_wrapper *>::type result_impl() const {
					return new derived_wrapper<typename A::result_type>(this->m_data);
				}
				template<typename T> typename boost::disable_if<typename has_result_type<T>::type, base_wrapper *>::type result_impl() const {
	                throw std::runtime_error(std::string("The type ") + typeid(A).name() + " has no result_type" + ALPS_STACKTRACE);
	                return NULL;
				}
		};
	}
}

 #endif