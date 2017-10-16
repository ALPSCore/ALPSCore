/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file iniparser_interface.hpp
    Implements C++ class to work with C ini-file parser library */

#include <alps/params_new/iniparser_interface.hpp>

#include <iniparser.h>

#include <stdexcept>
#include <algorithm>
#include <iterator>

#include <boost/foreach.hpp>

#include <iostream> // DEBUG!!!
#include <cstdio> // DEBUG

namespace alps {
    namespace params_new_ns {
        namespace detail {

            class ini_dict_impl {
                const std::string& inifile_;
                dictionary* inidict_;

              public:
                typedef std::vector<std::string> stringvec;

                ini_dict_impl(const std::string& inifile) : inifile_(inifile), inidict_(iniparser_load(inifile.c_str()))
                {
                    if (!inidict_) throw std::runtime_error("Cannot read INI file " + inifile);

                    dictionary_dump(inidict_, stdout); //DEBUG!!!
                }
                
                ~ini_dict_impl() {
                    if (inidict_) iniparser_freedict(inidict_);
                }

                /// Returns vector of sections in the INI file
                stringvec list_sections() const;

                /// Returns vector of keys in the given section
                stringvec list_keys(const std::string& sec) const;

                /// Returns the value for a given key
                std::string get_value(const std::string& key) const;
            };

            ini_dict_impl::stringvec ini_dict_impl::list_sections() const
            {
                int nsec=iniparser_getnsec(inidict_);
                if (nsec<0) throw std::runtime_error("Cannot read number of sections in "+inifile_);

                stringvec vec(1, "");
                vec.reserve(nsec+1);

                for (int i=0; i<nsec; ++i) {
                    vec.push_back(std::string(iniparser_getsecname(inidict_, i)));
                }
                return vec;
            }

            ini_dict_impl::stringvec ini_dict_impl::list_keys(const std::string &sec) const
            {
                int nkeys=iniparser_getsecnkeys(inidict_, sec.c_str());
                if (nkeys<0) throw std::runtime_error("Cannot determin number of keys in sec '"
                                                      + sec + "' of inifile '"
                                                      + inifile_+"'");
                std::cout << "DEBUG: " << nkeys << " keys.\n";
                std::vector<const char*> vptr(nkeys, (char*)0);
                if (!vptr.empty() && iniparser_getseckeys(inidict_, sec.c_str(), &vptr.front())==0)
                {
                    throw std::runtime_error("Cannot retrieve keys from sec '"
                                             + sec + "' of inifile '"
                                             + inifile_+"'");
                }
                stringvec kvec;
                kvec.reserve(nkeys);
                copy(vptr.begin(), vptr.end(), std::back_inserter(kvec));

                return kvec;
            }

            std::string ini_dict_impl::get_value(const std::string& key) const
            {
                const char* const miss="MISSED";
                const char* val=iniparser_getstring(inidict_, key.c_str(), miss);
                if (miss==val) throw std::runtime_error("Key '"+key+"' is missing in file "+inifile_);
                return std::string(val);
            }


            /* iniparser implementation */

            iniparser::iniparser(const std::string& inifile) : ini_dict_ptr_(new ini_dict_impl(inifile))
            {  }

            iniparser::~iniparser()
            {  }

            iniparser::kv_container_type iniparser::operator()() const
            {
                typedef std::vector<std::string> stringvec;
                kv_container_type kv_vec;
                
                const stringvec sections=ini_dict_ptr_->list_sections();
                BOOST_FOREACH(const std::string& sec, sections) {
                    std::cout << "DEBUG: sec='" << sec << "'" << std::endl;
                    const stringvec keys=ini_dict_ptr_->list_keys(sec);
                    BOOST_FOREACH(const std::string& k, keys) {
                        std::cout << "DEBUG: key='" << k << "'" << std::endl;
                        kv_vec.push_back(std::make_pair(k, ini_dict_ptr_->get_value(k)));
                    }
                }
                return kv_vec;
            }
            
        } // ::detail
    } // ::param_ns
} // ::alps

