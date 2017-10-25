/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params.cpp
    Contains implementation of alps::params */

#include <alps/params/iniparser_interface.hpp>
#include <alps/params.hpp>
#include <algorithm>

#include <boost/foreach.hpp>

namespace alps {
    namespace params_ns {

        namespace {
            template <typename M>
            struct compare {
                typedef typename M::value_type pair_type;
                bool operator()(const pair_type& lhs, const pair_type& rhs) const
                {
                    return (lhs.first==rhs.first) && lhs.second.equals(rhs.second);
                }
            };
        }
        
        bool dictionary::equals(const dictionary &rhs) const 
        {
            if (this->size()!=rhs.size()) return false;
            return std::equal(map_.begin(), map_.end(), rhs.map_.begin(), compare<map_type>());
        }

        
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
            td_map_type::const_iterator it=td_map_.find(name);
            return (td_map_.end()==it)? std::string() : it->second.descr;
        }

        bool params::operator==(const alps::params_ns::params& rhs) const
        {
            const params& lhs=*this;
            const dictionary& lhs_dict=*this;
            const dictionary& rhs_dict=rhs;
            return
                (lhs.raw_kv_content_ == rhs.raw_kv_content_) &&
                (lhs_dict==rhs_dict);
        }
        
    } // ::params_ns
}// alps::
            
