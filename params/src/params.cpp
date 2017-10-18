/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp
    Contains implementation of alps::params */

#include <alps/params/iniparser_interface.hpp>
#include <alps/params.hpp>

#include <boost/foreach.hpp>

namespace alps {
    namespace params_ns {
        
        void params::read_ini_file_(const std::string& inifile)
        {
            detail::iniparser parser(inifile);
            BOOST_FOREACH(const detail::iniparser::kv_pair& kv, parser()) {
                // FIXME!!! Check for duplicates and optionally warn!
                std::string key=kv.first;
                if (!key.empty() && key[0]==':') key.erase(0,1);
                raw_kv_content_[key]=kv.second;
            }
        }

        const std::string params::get_descr(const std::string& name) const
        {
            strmap::const_iterator it=descriptions_.find(name);
            return (descriptions_.end()==it)? std::string() : it->second;
        }
    } // ::params_ns
}// alps::
            
