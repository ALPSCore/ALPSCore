// ** Implementation of alps::params

#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
namespace alps {
    namespace params_ns {

        namespace detail {

            
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
                static boost::optional<bool> apply(const std::string& in) {
                    // FIXME: use C_locale and lowercase the string
                    boost::optional<bool> result;
                    if (in=="true") result=true;
                    if (in=="false") result=false;
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
                // FIXME: use pretty-typename!
                if (td_it->second.typestr() != typeid(T).name()) throw exception::type_mismatch(name, "Parameter already defined with a different type");
                td_it->second.descr()=descr;
                return true;
            }
            td_map_.insert(std::make_pair(name, detail::td_pair(typeid(T).name(), descr))); // FIXME: use pretty-typename!

            strmap::const_iterator it=raw_kv_content_.find(name);
            if (it==raw_kv_content_.end()) {
                if (this->exists(name)) return true;
                return false; // need to decide whether the default available
            }
            if (!assign_to_name_<T>(name, it->second)) {
                ++err_status_; // FIXME: record the problem: cannot parse
                (*this)[name].clear();
            }
            return true;
        }

        template <typename T>
        params& params::define(const std::string& name, const std::string& descr)
        {
            if (!define_<T>(name, descr)) {
                if (!this->exists<T>(name)) ++err_status_; // FIXME: record the problem: missing required param
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

    } // params_ns::
} // alps::
