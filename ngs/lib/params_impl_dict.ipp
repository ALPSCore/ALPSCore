/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_PARAMS_IMPL_DICT_IPP
#define ALPS_NGS_PARAMS_IMPL_DICT_IPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/param.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/detail/params_impl_base.hpp>

#include <boost/bind.hpp>
#include <boost/python/dict.hpp> 
#include <boost/python/call_method.hpp> 
#include <boost/python/stl_iterator.hpp>

#include <string>

namespace alps {

    namespace detail {

        class params_impl_dict : public params_impl_base {

            public:

                params_impl_dict(boost::python::object arg)
                    : values_(arg)
                {
                    if (std::string(arg.ptr()->ob_type->tp_name) == "params")
                        values_ = dynamic_cast<params_impl_dict &>(*boost::python::extract<params &>(arg)().get_impl()).values_;
                    boost::python::incref(values_.ptr());
                }

                ~params_impl_dict() {
                    boost::python::decref(values_.ptr());
                }

                std::size_t size() const {
                    return boost::python::extract<std::size_t>(values_.attr("__len__")());
                }

				// TODO: save using generic python hdf5 interface
                std::vector<std::string> keys() const {
                    std::vector<std::string> keys;
                    boost::python::extract<boost::python::dict> dict(values_);
                    const boost::python::object it = dict().iterkeys();
                    for (std::size_t i = 0; i < size(); ++i) {
                        boost::python::object key(it.attr("next")());
                        if (std::string(key.ptr()->ob_type->tp_name) == "bool")
                            keys.push_back(cast<std::string>(boost::python::extract<bool>(key)()));
                        if (std::string(key.ptr()->ob_type->tp_name) == "int")
                            keys.push_back(cast<std::string>(boost::python::extract<int>(key)()));
                        else if (std::string(key.ptr()->ob_type->tp_name) == "long")
                            keys.push_back(cast<std::string>(boost::python::extract<long>(key)()));
                        else if (std::string(key.ptr()->ob_type->tp_name) == "float")
                            keys.push_back(cast<std::string>(boost::python::extract<double>(key)()));
                        else if (std::string(key.ptr()->ob_type->tp_name) == "str")
                            keys.push_back(boost::python::extract<std::string>(key)());
                        else
                            throw std::invalid_argument("unknown type: "  + std::string(key.ptr()->ob_type->tp_name) + ALPS_STACKTRACE);
                    }
                    return keys;
                }

                param operator[](std::string const & key) {
                    return param(
                        boost::bind(&params_impl_dict::getter, boost::ref(*this), key),
                        boost::bind(&params_impl_dict::setter, boost::ref(*this), key, _1)
                    );
                }

                param const operator[](std::string const & key) const {
                    if (!defined(key))
                        throw std::invalid_argument("unknown argument: "  + key + ALPS_STACKTRACE);
                    return param(getter(key));
                }

                bool defined(std::string const & key) const {
                    return boost::python::call_method<bool>(values_.ptr(), "__contains__", key);
                }

				// TODO: save using generic python hdf5 interface
                void save(hdf5::archive & ar) const {
                    boost::python::extract<boost::python::dict> dict(values_);
                    const boost::python::object kit = dict().iterkeys();
                    const boost::python::object vit = dict().itervalues();
                    for (std::size_t i = 0; i < size(); ++i) {
                        boost::python::object key(kit.attr("next")());
                        boost::python::object value(vit.attr("next")());
                        std::string segment;
                        if (std::string(key.ptr()->ob_type->tp_name) == "bool")
                            segment = cast<std::string>(boost::python::extract<bool>(key)());
                        if (std::string(key.ptr()->ob_type->tp_name) == "int")
                            segment = cast<std::string>(boost::python::extract<int>(key)());
                        else if (std::string(key.ptr()->ob_type->tp_name) == "long")
                            segment = cast<std::string>(boost::python::extract<long>(key)());
                        else if (std::string(key.ptr()->ob_type->tp_name) == "float")
                            segment = cast<std::string>(boost::python::extract<double>(key)());
                        else if (std::string(key.ptr()->ob_type->tp_name) == "str")
                            segment = boost::python::extract<std::string>(key)();
                        else
                            throw std::invalid_argument("unknown key type: "  + std::string(key.ptr()->ob_type->tp_name) + ALPS_STACKTRACE);
                        if (std::string(value.ptr()->ob_type->tp_name) == "bool")
                            ar << make_pvp(segment, boost::python::extract<bool>(value)());
                        if (std::string(value.ptr()->ob_type->tp_name) == "int")
                            ar << make_pvp(segment, boost::python::extract<int>(value)());
                        else if (std::string(value.ptr()->ob_type->tp_name) == "long")
                            ar << make_pvp(segment, boost::python::extract<long>(value)());
                        else if (std::string(value.ptr()->ob_type->tp_name) == "float")
                            ar << make_pvp(segment, boost::python::extract<double>(value)());
                        else if (std::string(value.ptr()->ob_type->tp_name) == "str")
                            ar << make_pvp(segment, boost::python::extract<std::string>(value)());
                        else
                            throw std::invalid_argument("unknown value type: "  + std::string(value.ptr()->ob_type->tp_name) + ALPS_STACKTRACE);
                    }
                }

                void load(hdf5::archive & ar) {
                    std::vector<std::string> list = ar.list_children(ar.get_context());
                    for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
                        std::string value;
                        ar >> make_pvp(*it, value);
                        setter(*it, value);
                    }
                }

                params_impl_base * clone() {
                    return new params_impl_dict(*this);
                }

                boost::python::object native_copy() {
                    return boost::python::call_method<boost::python::object>(values_.ptr(), "__copy__");
                }

                boost::python::object native_getitem(boost::python::object const & key) {
                    return boost::python::call_method<boost::python::object>(values_.ptr(), "__getitem__", key);
                }

                void native_setitem(boost::python::object const & key, boost::python::object & value) {
                    boost::python::call_method<void>(values_.ptr(), "__setitem__", key, value);
                }
                
                void native_delitem(boost::python::object const & key) {
                    boost::python::call_method<void>(values_.ptr(), "__delitem__", key);
                }
                
                bool native_contains(boost::python::object const & key) {
                    return boost::python::call_method<bool>(values_.ptr(), "__contains__", key);
                }
                
                boost::python::object native_iter() {
                    boost::python::extract<boost::python::dict> dict(values_);
                    if (dict.check())
                        return dict().iterkeys();
                    else
                        return boost::python::call_method<boost::python::object>(values_.ptr(), "__iter__");
                }

				#ifdef ALPS_HAVE_MPI
					void broadcast(int root) {
						throw std::logic_error("no communicator available" + ALPS_STACKTRACE);
					}
				#endif

            private:

                params_impl_dict(params_impl_dict const & arg)
                    : values_(arg.values_)
                {
                    boost::python::incref(values_.ptr());
                }

                void setter(std::string key, std::string value) {
                    boost::python::call_method<void>(values_.ptr(), "__setitem__", key, value);
                }

                std::string getter(std::string key) const {
                	boost::python::object value(boost::python::call_method<boost::python::object>(values_.ptr(), "__getitem__", key));
                    if (std::string(value.ptr()->ob_type->tp_name) == "bool")
                    	return cast<std::string>(boost::python::extract<bool>(value)());
                    if (std::string(value.ptr()->ob_type->tp_name) == "int")
                    	return cast<std::string>(boost::python::extract<int>(value)());
                    else if (std::string(value.ptr()->ob_type->tp_name) == "long")
                    	return cast<std::string>(boost::python::extract<long>(value)());
                    else if (std::string(value.ptr()->ob_type->tp_name) == "float")
                    	return cast<std::string>(boost::python::extract<double>(value)());
                    else if (std::string(value.ptr()->ob_type->tp_name) == "str")
                    	return boost::python::extract<std::string>(value)();
                    else {
                        throw std::invalid_argument("unknown value type: "  + std::string(value.ptr()->ob_type->tp_name) + ALPS_STACKTRACE);
                        return "";
                    }
                }

                boost::python::object values_;
        };

    }
}

#endif
