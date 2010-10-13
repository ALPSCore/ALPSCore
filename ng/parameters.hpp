/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2010 by Lukas Gamper <gamperl@gmail.com>
 *                       Matthias Troyer <troyer@comp-phys.org>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <alps/hdf5.hpp>

#include <boost/utility.hpp>
#include <boost/variant.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/lexical_cast.hpp>

#include <map>

#ifndef ALPS_NG_PARAMETERS_HPP
#define ALPS_NG_PARAMETERS_HPP

namespace alps {
    namespace ng {
        namespace detail {

            typedef boost::mpl::vector<std::string, int, double> parameter_value_types;
            typedef boost::make_variant_over<parameter_value_types>::type parameter_value_base;

            template<typename T> struct parameter_value_reader : public boost::static_visitor<> {
                template <typename U> void operator()(U & v) const { value = boost::lexical_cast<T, U>(v); }
                void operator()(T & v) const { value = v; }
                mutable T value;
            };

            struct parameter_value_serializer: public boost::static_visitor<> {
                parameter_value_serializer(hdf5::oarchive & a, std::string const & p) : ar(a), path(p) {}
                template<typename T> void operator()(T & v) const { ar << make_pvp(path, v); }
                hdf5::oarchive & ar;
                std::string const & path;
            };
        }

        class parameter_value : public detail::parameter_value_base {
            public:
                parameter_value() {}
                template <typename T> parameter_value(T const & v): detail::parameter_value_base(v) {}
                parameter_value(parameter_value const & v): detail::parameter_value_base(static_cast<detail::parameter_value_base const &>(v)) {}

                template <typename T> typename boost::enable_if<typename boost::mpl::contains<detail::parameter_value_types, T>::type, parameter_value &>::type operator=(T const & v) {
                    detail::parameter_value_base::operator=(v);
                    return *this;
                }

                template <typename T> typename boost::disable_if<typename boost::mpl::contains<detail::parameter_value_types, T>::type, parameter_value &>::type operator=(T const & v) {
                    detail::parameter_value_base::operator=(boost::lexical_cast<std::string>(v));
                    return *this;
                }

                #define ALPS_NGS_CAST_OPERATOR(T)                                                                                                              \
                    operator T () const {                                                                                                                      \
                        detail::parameter_value_reader< T > visitor;                                                                                           \
                        boost::apply_visitor(visitor, *this);                                                                                                  \
                        return visitor.value;                                                                                                                  \
                    }
                ALPS_NGS_CAST_OPERATOR(short)
                ALPS_NGS_CAST_OPERATOR(int)
                ALPS_NGS_CAST_OPERATOR(long)
                ALPS_NGS_CAST_OPERATOR(float)
                ALPS_NGS_CAST_OPERATOR(double)
                ALPS_NGS_CAST_OPERATOR(bool)
                ALPS_NGS_CAST_OPERATOR(std::size_t)
                ALPS_NGS_CAST_OPERATOR(std::string)
                #undef ALPS_NGS_CAST_OPERATOR
        };

    // can we keep parameter ordering? like in curent class
        class parameters : public std::map<std::string, parameter_value> {
            public: 
                parameters(std::string const & input_file) {
                    hdf5::iarchive ar(input_file);
                    ar >> make_pvp("/parameters", *this);
                }

                parameter_value & operator[](std::string const & k) {
                    return std::map<std::string, parameter_value>::operator[](k);
                }

                parameter_value const & operator[](std::string const & k) const {
                    if (find(k) == end())
                        throw std::invalid_argument("unknown argument: "  + k);
                    return find(k)->second;
                }

                parameter_value value_or_default(std::string const & k, parameter_value const & v) const {
                    if (find(k) == end())
                        return parameter_value(v);
                    return find(k)->second;
                }

                bool defined(std::string const & k) const {
                    return find(k) != end();
                }

                void serialize(hdf5::oarchive & ar) const {
                    for (const_iterator it = begin(); it != end(); ++it)
                        boost::apply_visitor(detail::parameter_value_serializer(ar, it->first), it->second);
                }

                void serialize(hdf5::iarchive & ar) {
                    std::vector<std::string> list = ar.list_children(ar.get_context());
                    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                        std::string v;
                        using namespace ::alps;
                        ar >> make_pvp(*it, v);
                        insert(std::make_pair(*it, v));
                    }
                }
        };
    }
}

#endif
