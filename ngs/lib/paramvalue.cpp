/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#include <alps/ngs/detail/paramvalue.hpp>

namespace alps {
    namespace detail {

        template<typename T> struct paramvalue_reader 
			: public boost::static_visitor<> 
		{

            paramvalue_reader(): value() {}

            template <typename U> void operator()(U & v) const {
				// TODO: implement!
//                value = convert<T>(v);
            }

            void operator()(T & v) const { 
                value = v; 
            }

			// todo: customize for boost::python object ...

            mutable T value;
        };

        struct paramvalue_saver: public boost::static_visitor<> {

            paramvalue_saver(hdf5::archive & a)
                : ar(a) 
            {}

            template<typename T> void operator()(T const & v) const {
				// TODO: implement!
//                ar << make_pvp("", v);
            }
			
			// todo: customize for boost::python object ...

            hdf5::archive & ar;
        };

		#define ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL(T)								\
			paramvalue::operator T () const {										\
				paramvalue_reader< T > visitor;								\
				boost::apply_visitor(visitor, *this);                               \
				return visitor.value;												\
			}
		ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL)
		#undef ALPS_NGS_PARAMVALUE_OPERATOR_T_IMPL

		#define ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL(T)								\
			paramvalue & paramvalue::operator=( T const & arg) {					\
				paramvalue_base::operator=(arg);									\
				return *this;														\
			}
		ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL)
		#undef ALPS_NGS_PARAMVALUE_OPERATOR_EQ_IMPL

		void paramvalue::save(hdf5::archive & ar) const {
			boost::apply_visitor(
				paramvalue_saver(ar), static_cast<paramvalue_base const &>(*this)
			);
		}

		void paramvalue::load(hdf5::archive & ar) {
			// TODO: implement!
		}

	}

	#define ALPS_NGS_PARAMETERVALUE_CONVERT_IMPL(T)                             \
		template<> ALPS_DECL T convert(											\
			detail::paramvalue const & arg										\
		) {																		\
			detail::paramvalue_reader< T > visitor;								\
			boost::apply_visitor(visitor, arg);									\
			return visitor.value;												\
		}
	ALPS_NGS_FOREACH_PARAMETERVALUE_TYPE(ALPS_NGS_PARAMETERVALUE_CONVERT_IMPL)
	#undef ALPS_NGS_PARAMETERVALUE_CONVERT_IMPL

}
