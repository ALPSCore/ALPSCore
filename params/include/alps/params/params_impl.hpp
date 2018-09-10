/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
   @file params_impl.hpp
   Contains header-part implementation of alps::params
   NOT TO BE INCLUDED DIRECTLY!
*/

#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <locale> // FIXME: needed only for boolean conversions

namespace alps {
    namespace params_ns {

        namespace detail {

            // FIXME: move this to a *.cpp file
            template <typename T>
            struct parse_string {
                static boost::optional<T> apply(const std::string& in) {
                    T conv_result;
                    boost::optional<T> result;
                    if (boost::conversion::try_lexical_convert(in, conv_result)) {
                        result=conv_result;
                    }
                    return result;
                }
            };

            template <>
            struct parse_string<std::string> {
                static boost::optional<std::string> apply(const std::string& in) {
                    return in;
                }
            };

            template <>
            struct parse_string<bool> {
                static boost::optional<bool> apply(std::string in) {
                    std::locale c_locale("C");
                    for(char& c: in) {
                        c=tolower(c, c_locale);
                    }
                    boost::optional<bool> result;
                    if (in=="true" || in=="on" || in=="yes" || in=="1") result=true;
                    if (in=="false" || in=="off" || in=="no" || in=="0") result=false;
                    return result;
                }
            };

            template <typename T>
            struct parse_string< std::vector<T> > {
                static boost::optional< std::vector<T> > apply(const std::string& in) {
                    typedef std::vector<T> value_type;
                    typedef boost::optional<value_type> result_type;
                    typedef boost::optional<T> optional_el_type;
                    typedef std::string::const_iterator sit_type;
                    value_type result_vec;
                    result_type result;
                    sit_type it1=in.begin();
                    while (it1!=in.end()) {
                        sit_type it2=find(it1, in.end(), ',');
                        optional_el_type elem=parse_string<T>::apply(std::string(it1,it2));
                        if (!elem) return result;
                        result_vec.push_back(*elem);
                        if (it2!=in.end()) ++it2;
                        it1=it2;
                    }
                    result=result_vec;
                    return result;
                }
            };

        } // ::detail

        template <typename T>
        bool params::assign_to_name_(const std::string& name, const std::string& strval)
        {
            boost::optional<T> result=detail::parse_string<T>::apply(strval);
            if (result) {
                (*this)[name]=*result;
                return true;
            } else {
                return false;
            }
        }

        template <typename T>
        bool params::define_(const std::string& name, const std::string& descr)
        {
            if (this->exists(name) && !this->exists<T>(name))
                throw exception::type_mismatch(name, "Parameter already in dictionary with a different type");

            td_map_type::iterator td_it=td_map_.find(name); // FIXME: use lower-bound instead
            if (td_it!=td_map_.end()) {
                if (td_it->second.typestr() != detail::make_typestr::apply<T>()) throw exception::type_mismatch(name, "Parameter already defined with a different type");
                td_it->second.descr()=descr;
                return true;
            }
            td_map_.insert(std::make_pair(name, detail::td_type::make_pair<T>(descr, td_map_.size())));

            strmap::const_iterator it=raw_kv_content_.find(name);
            if (it==raw_kv_content_.end()) {
                if (this->exists(name)) return true;
                return false; // caller needs to decide whether the default is available
            }
            if (!assign_to_name_<T>(name, it->second)) {
                err_status_.push_back("Cannot parse parameter '"+name+"' as the requested type");
                (*this)[name].clear();
            }
            return true;
        }

        template <typename T>
        params& params::define(const std::string& name, const std::string& descr)
        {
            if (!define_<T>(name, descr)) {
                if (!this->exists<T>(name)) err_status_.push_back("Required parameter '"+name+"' is missing");
            }
            return *this;
        }

        template <typename T>
        params& params::define(const std::string& name, const T& defval, const std::string& descr)
        {
            if (!define_<T>(name, descr)) {
                (*this)[name]=defval;
            }
            return *this;
        }

        inline bool params::supplied(const std::string &name) const
        {
            return raw_kv_content_.count(name);
        }

        inline bool params::defaulted(const std::string& name) const
        {
            return exists(name) && !supplied(name);
        }

        inline bool params::defined(const std::string& name) const
        {
            return td_map_.count(name)!=0 || exists(name);
        }

        inline void swap(params& p1, params& p2)
        {
            using std::swap;
            swap(static_cast<dictionary&>(p1), static_cast<dictionary&>(p2));
            swap(p1.raw_kv_content_, p2.raw_kv_content_);
            swap(p1.td_map_, p2.td_map_);
            swap(p1.err_status_, p2.err_status_);
            swap(p1.origins_.data(), p2.origins_.data());
        }

        inline std::string params::get_origin_name() const
        {
            return origin_name(*this);
        }

    } // params_ns::
    using params_ns::origin_name;
    using params_ns::swap;

} // alps::
