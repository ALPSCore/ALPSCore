/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602
#define ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602

#include "alps/config.hpp"
// #include "alps/hdf5/archive.hpp"

#ifdef ALPS_HAVE_MPI
#include "alps/utilities/mpi.hpp"
#endif

#include <map>

#include "./params/dict_exceptions.hpp"
#include "./params/dict_value.hpp"

namespace alps {
    namespace params_ns {

        /// Python-like dictionary
        // FIXME: TODO: rewrite the whole thing using a proxy object 
        class dictionary {
          public:
            typedef dict_value value_type;

          private:
            typedef std::map<std::string, value_type> map_type;
            map_type map_;
            
          public:
            /// Virtual destructor to make dictionary inheritable
            virtual ~dictionary() {}
            
            bool empty() const { return map_.empty(); }
            std::size_t size() const { return map_.size(); }

            /// Access with intent to assign
            value_type& operator[](const std::string& key) {
                map_type::iterator it=map_.lower_bound(key);
                if (it==map_.end() ||
                    map_.key_comp()(key, it->first)) {
                    // it's a new element, we have to construct it here
                    // and copy it to the map, returning the ref to the inserted element
                    it=map_.insert(it,map_type::value_type(key, value_type(key)));
                }
                // return reference to the existing or the newly-created element
                return it->second;
            }

            /// Read-only access
            const value_type& operator[](const std::string& key) const {
                map_type::const_iterator it=map_.find(key);
                if (it==map_.end()) throw exception::uninitialized_value(key, "Attempt to read uninitialized value");
                return it->second;
            }

          private:
            /// Check if the key exists and has a value; return the iterator
            map_type::const_iterator find_nonempty_(const std::string& key) const {
                map_type::const_iterator it=map_.find(key);
                if (it!=map_.end() && !(it->second).empty())
                    return it;
                else
                    return map_.end();
            }
          public:
            
            /// Check if a key exists and has a value (without creating the key)
            bool exists(const std::string& key) const {
                return find_nonempty_(key)!=map_.end();
            }
            
            /// Check if a key exists and has a value of a particular type (without creating the key)
            template <typename T>
            bool exists(const std::string& key) const {
                map_type::const_iterator it=find_nonempty_(key);
                return it!=map_.end() && (it->second).isType<T>();
            }

        };


        /// Parse sectioned INI file or HDF5 or command line, provide the results as dictionary.
        /**
           1. Default-constructed `params` object cannot be re-associated with a file;
              therefore, is 100% equivalent to `dictionary` ("is-a" dictionary).
              
           2. Lexing of the file and of the command line occurs at construction.
              Command line overrides the file. INI file name is taken from the command line.

           3. Parsing of a specific parameter occurs at the time of its type definition.
              There is no way for parameters to appear after the file and cmdline are read.
              
           // 3. Command line parsing is rudimentary:
           //    3.1. First argument is always INI file or HDF5 file.
           //    3.2. Options arguments start with single or double dash: `[-]-option_pair`
           //    3.3. `option_pair` has format `key[=value]`
           //    3.4. Keys are case-insensitive and contain [A-Za-z0-9_-].
           //         (MAYBE: disallow [_] and convert [-] to [_]?)
           //    3.5. 
         */
        class params : public dictionary {
          private:
            typedef std::map<std::string,std::string> strmap;
            // typedef std::vector<std::string> strvec;
            
            struct td_pair {
                std::string typestr;
                std::string descr;
                td_pair(const std::string& t, const std::string& d) : typestr(t), descr(d) {}
            };
            typedef std::map<std::string, td_pair> td_map_type;
            
            
            strmap raw_kv_content_;
            td_map_type td_map_;
            
            int err_status_;

            void read_ini_file_(const std::string& inifile);

            template <typename T>
            bool assign_to_name_(const std::string& name, const std::string& strval);

            /// Does the job of define(), returns false if the name is missing in raw_argsand default must be checked
            template <typename T>
            bool define_(const std::string& name, const std::string& descr);
            
          public:
            /// Default ctor
            params() : dictionary(), raw_kv_content_(), td_map_(), err_status_(0) {}

            params(const std::string& inifile) : dictionary(), raw_kv_content_(), td_map_(), err_status_(0)  { read_ini_file_(inifile); }

            /// No-errors status
            bool ok() const { return 0==err_status_; }
            
            /// Defines a parameter; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const std::string& descr);

            /// Defines a parameter with a default; returns false on error, and records the error in the object
            template<typename T>
            params& define(const std::string& name, const T& defval, const std::string& descr);

            const std::string get_descr(const std::string& name) const;

            template <typename A>
            void save(const A&) const { throw std::logic_error("params::save() is not yet implemented"); }
            
            template <typename A>
            void load(const A&) { throw std::logic_error("params::load() is not yet implemented"); }
        };
        
    } // params_ns::
    typedef params_ns::params params;
} // alps::


// ** Implementation **
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
                // std::cout << "DEBUG: err_stat_=" << err_status_ <<" redefinition for '" << name << "' , td_it->second.typestr='" << td_it->second.typestr << "' td_it->second.descr='" << td_it->second.descr << std::endl;
                // FIXME: use pretty-typename!
                if (td_it->second.typestr != typeid(T).name()) throw exception::type_mismatch(name, "Parameter already defined with a different type");
                td_it->second.descr=descr;
                return true;
            }
            td_map_.insert(std::make_pair(name, td_pair(typeid(T).name(), descr))); // FIXME: use pretty-typename!

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
        

#endif /* ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602 */
