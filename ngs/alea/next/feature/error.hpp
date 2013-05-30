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

#ifndef ALPS_NGS_ACCUMULATOR_ERROR_HPP
#define ALPS_NGS_ACCUMULATOR_ERROR_HPP

#include <alps/ngs/alea/next/feature.hpp>
#include <alps/ngs/alea/next/feature/mean.hpp>
#include <alps/ngs/alea/next/feature/count.hpp>

#include <alps/hdf5.hpp>
#include <alps/ngs/numeric.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/short_print.hpp>

#include <boost/mpl/if.hpp>
#include <boost/utility.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/type_traits/is_integral.hpp>

#include <stdexcept>

namespace alps {
	namespace accumulator {
		namespace tag {
			struct error;
		}

		template<typename T> struct error_type : public mean_type<T> {};

		template<typename T> struct has_feature<T, tag::error> {
	        template<typename R, typename C> static char helper(R(C::*)() const);
	        template<typename C> static char check(boost::integral_constant<std::size_t, sizeof(helper(&C::error))>*);
	        template<typename C> static double check(...);
			typedef boost::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
	    };

		template<typename T> typename error_type<T>::type error(T const & arg) {
			return arg.error();
		}

		namespace detail {

			template<typename A> typename boost::enable_if<
				  typename has_feature<A, tag::error>::type
				, typename error_type<A>::type
			>::type error_impl(A const & acc) {
				return error(acc);
			}

			template<typename A> typename boost::disable_if<
				  typename has_feature<A, tag::error>::type
				, typename error_type<A>::type
			>::type error_impl(A const & acc) {
			    throw std::runtime_error(std::string(typeid(A).name()) + " has no error-method" + ALPS_STACKTRACE);
				return typename error_type<A>::type();
			}
		}

		namespace impl {

			template<typename T, typename B> struct Accumulator<T, tag::error, B> : public B {

			    public:
				    typedef typename alps::accumulator::error_type<B>::type error_type;
                	typedef Result<T, tag::error, typename B::result_type> result_type;

			        template<typename ArgumentPack> Accumulator(ArgumentPack const & args): B(args), m_sum2(T()) {}

			        Accumulator(): B(), m_sum2(T()) {}
			        Accumulator(Accumulator const & arg): B(arg), m_sum2(arg.m_sum2) {}

			        error_type const error() const {
                        using std::sqrt;
                        using alps::ngs::numeric::sqrt;
                        using alps::ngs::numeric::operator/;
                        using alps::ngs::numeric::operator-;
                        using alps::ngs::numeric::operator*;

						// TODO: make library for scalar type
						typename alps::hdf5::scalar_type<error_type>::type cnt = B::count();
                        return sqrt((m_sum2 / cnt - B::mean() * B::mean()) / (cnt - 1));
			        }

			        void operator()(T const & val) {
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+=;
                        using alps::ngs::numeric::detail::check_size;

						B::operator()(val);
                        check_size(m_sum2, val);
                        m_sum2 += val * val;
			        }

					template<typename S> void print(S & os) const {
						B::print(os);
                        os << " +/-" << alps::short_print(error());
			        }

			        void save(hdf5::archive & ar) const {
						B::save(ar);
						ar["mean/error"] = error();
			        }

			        void load(hdf5::archive & ar) { // TODO: make archive const
                        using alps::ngs::numeric::operator*;
                        using alps::ngs::numeric::operator+;

						B::load(ar);
						error_type error;
						ar["mean/error"] >> error;
						// TODO: make library for scalar type
						typename alps::hdf5::scalar_type<error_type>::type cnt = B::count();
						m_sum2 = (error * error * (cnt - 1) + B::mean() * B::mean()) * cnt;
			        }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
	                    using alps::hdf5::get_extent;

                    	return B::can_load(ar)
                    		&& ar.is_data("mean/error") 
                    		&& boost::is_scalar<T>::value == ar.is_scalar("mean/error")
                    		&& (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/error"))
                    	;
                    }

			        void reset() {
						B::reset();
						m_sum2 = T();
			        }

			    private:
			        T m_sum2;
			};

			template<typename T, typename B> class Result<T, tag::error, B> : public B {

			    public:
					typedef typename alps::accumulator::error_type<B>::type error_type;

				    Result() 
				    	: B()
				    	, m_error(error_type()) 
				    {}

				    template<typename A> Result(A const & acc)
						: B(acc)
						, m_error(detail::error_impl(acc))
			        {}

			        error_type const error() const { 
			        	return m_error; 
			        }

					template<typename S> void print(S & os) const {
						B::print(os);
                        os << " +/-" << alps::short_print(error());
			        }

			        void save(hdf5::archive & ar) const {
						B::save(ar);
						ar["mean/error"] = error();
			        }

			        void load(hdf5::archive & ar) {
						B::load(ar);
						ar["mean/error"] >> m_error;
			        }

                    static std::size_t rank() { return B::rank() + 1; }
                    static bool can_load(hdf5::archive & ar) { // TODO: make archive const
	                    using alps::hdf5::get_extent;

                    	return B::can_load(ar) 
                    		&& ar.is_data("mean/error") 
                    		&& boost::is_scalar<T>::value == ar.is_scalar("mean/error")
                    		&& (boost::is_scalar<T>::value || get_extent(T()).size() == ar.dimensions("mean/error"))
                    	;
                    }

			    private:
			        error_type m_error;		        
			};

			template<typename B> class BaseWrapper<tag::error, B> : public B {
				public:
				    virtual bool has_error() const = 0;
	        };

			template<typename T, typename B> class ResultTypeWrapper<T, tag::error, B> : public B {
				public:
				    virtual typename error_type<B>::type error() const = 0;
	        };

			template<typename T, typename B> class DerivedWrapper<T, tag::error, B> : public B {
				public:
				    DerivedWrapper(): B() {}
				    DerivedWrapper(T const & arg): B(arg) {}

				    bool has_error() const { return has_feature<T, tag::error>::type::value; }

				    typename error_type<B>::type error() const { return detail::error_impl(this->m_data); }
	        };

		}
	}
}

 #endif
