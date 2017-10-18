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

#include "./params_new/dict_exceptions.hpp"
#include "./params_new/dict_value.hpp"


namespace alps {
    namespace params_new_ns {

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
            strmap raw_kv_content_;
            strmap descriptions_;
            // strvec def_errors_;
            
            void read_ini_file_(const std::string& inifile);

            template <typename T>
            bool assign_to_name_(const std::string& name, const std::string& strval);
          public:
            /// Default ctor
            params() : dictionary(), raw_kv_content_() {}

            params(const std::string& inifile) : dictionary(), raw_kv_content_()  { read_ini_file_(inifile); }

            /// Defines a parameter; returns false on error, and records the error in the object
            template<typename T>
            bool define(const std::string& name, const std::string& descr);

            /// Defines a parameter with a default; returns false on error, and records the error in the object
            template<typename T>
            bool define(const std::string& name, const T& defval, const std::string& descr);

            const std::string get_descr(const std::string& name) const;
        };
        
    } // params_ns::
} // alps::


// ** Implementation **
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
namespace alps {
    namespace params_new_ns {

        // inline params::find_def_(

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
        bool params::define(const std::string& name, const std::string& descr)
        {
            if (this->exists(name) && !this->exists<T>(name))
                throw exception::type_mismatch(name, "Parameter already defined with a different type");

            descriptions_[name]=descr;
            strmap::const_iterator it=raw_kv_content_.find(name);
            if (it==raw_kv_content_.end()) {
                if (this->exists(name)) return true;
                return false; // FIXME: and record the problem
            }
            return assign_to_name_<T>(name, it->second);
        }

        template <typename T>
        bool params::define(const std::string& name, const T& defval, const std::string& descr)
        {
            if (this->exists(name) && !this->exists<T>(name))
                throw exception::type_mismatch(name, "Parameter already defined with a different type");
            
            descriptions_[name]=descr;
            strmap::const_iterator it=raw_kv_content_.find(name);
            if (it==raw_kv_content_.end()) {
                if (!this->exists(name)) (*this)[name]=defval;
            } else {
                if (!assign_to_name_<T>(name, it->second)) {
                    // FIXME: record the problem
                    (*this)[name].clear();
                    return false;
                }
            }
            return true;
        }

    } // params_ns::
} // alps::
        

#endif /* ALPS_PARAMS_HPP_INCLUDED_00f672a032d949a7aa0e760a6b6f0602 */
