/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file iniparser_interface.cpp
    Implements C++ class to work with C ini-file parser library */

#include <alps/params/iniparser_interface.hpp>

#include <iniparser.h>

#include <stdexcept>
#include <algorithm>
#include <iterator>

#include <boost/foreach.hpp>

namespace alps {
    namespace params_ns {
        namespace detail {

            class ini_dict_impl {
                dictionary* inidict_;

              public:
                struct kv_type {
                    const char* key;
                    const char* val;
                    kv_type(const char* k, const char* v): key(k), val(v) {}
                };

                ini_dict_impl(const std::string& inifile)
                    : inidict_(iniparser_load(inifile.c_str()))
                {
                    if (!inidict_) throw std::runtime_error("Cannot read INI file " + inifile);
                }

                ~ini_dict_impl() {
                    if (inidict_) iniparser_freedict(inidict_);
                }

                /// Returns the number of entries in the dictionary object
                std::size_t size() const;

                /// Returns pointers to a given key and value
                kv_type get_kv(const std::size_t i) const;
            };

            std::size_t ini_dict_impl::size() const
            {
                int n=inidict_->n;
                if (n<0) throw std::runtime_error("Dictionary is invalid: negative number of entries");
                return static_cast<std::size_t>(n);
            }

            ini_dict_impl::kv_type ini_dict_impl::get_kv(const std::size_t i) const
            {
                if (i>=this->size()) throw std::out_of_range("Access beyond the end of the dictionary");
                return kv_type(inidict_->key[i], inidict_->val[i]);
            }


            /* iniparser implementation */

            iniparser::iniparser(const std::string& inifile) : ini_dict_ptr_(new ini_dict_impl(inifile))
            {  }

            iniparser::~iniparser()
            {  }

            iniparser::kv_container_type iniparser::operator()() const
            {
                kv_container_type kv_vec;

                // the actual vector size will likely be smaller due to sections
                std::size_t inidict_sz=ini_dict_ptr_->size();

                kv_vec.reserve(inidict_sz);
                for (std::size_t i=0; i<inidict_sz; ++i) {
                    ini_dict_impl::kv_type kv=ini_dict_ptr_->get_kv(i);
                    if (kv.val && kv.key) {
                        kv_vec.push_back(std::make_pair(std::string(kv.key),std::string(kv.val)));
                    }
                }
                return kv_vec;
            }

        } // ::detail
    } // ::param_ns
} // ::alps
