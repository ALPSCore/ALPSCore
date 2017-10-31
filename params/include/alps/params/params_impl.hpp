/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/** @file params_impl.hpp Contains implementations of inline and template functions for alps::params class */

#ifndef ALPS_PARAMS_PARAMS_IMPL_HPP_8243f4d88828473688ec07edbb7a9f76
#define ALPS_PARAMS_PARAMS_IMPL_HPP_8243f4d88828473688ec07edbb7a9f76

namespace alps {
    namespace params_ns {

        inline void params::certainly_parse(bool reassign) const
        {
            boost::program_options::options_description odescr;
            certainly_parse(odescr,reassign);
        }

        inline void params::possibly_parse() const
        {
            if (!is_valid_) certainly_parse();
        }

        inline void params::invalidate() {
            is_valid_=false;
        }

        inline void params::init()
        {
            is_valid_=false;
            this->define("help", "Provides help message");
        }

        inline params::params()
        {
            init();
        }

        inline params::params(unsigned int argc, const char* const* argv, const char* hdfpath)
        {
          init(argc,argv,hdfpath);
        }

        inline params::params(hdf5::archive ar, std::string const & path)
        {
            this->load(ar, path);
        }

      inline params::params(const std::string& inifile) : infile_(inifile)
        {
            preparse_ini();
            init();
        }

        inline bool params::is_restored() const
        {
            return bool(archname_);
        }

        inline std::string params::get_archive_name() const
        {
            if (archname_) return *archname_;
            throw not_restored("This instance of parameters was not restored from an archive");
        }

        inline std::size_t params::size() const
        {
            possibly_parse(); return optmap_.size();
        }

        inline const params::mapped_type& params::operator[](const std::string& k) const
        {
            possibly_parse();
            return const_cast<const options_map_type&>(optmap_)[k];
        }

        inline params::mapped_type& params::operator[](const std::string& k)
        {
            possibly_parse();
            return optmap_[k];
        }

        inline params::const_iterator params::begin() const
        {
            possibly_parse();
            return optmap_.begin();
        }

        inline params::const_iterator params::end() const
        {
            possibly_parse();
            return optmap_.end();
        }

        inline params::missing_params_iterator params::begin_missing() const
        {
            return detail::iterators::make_missing_params_iterator(this->begin(), this->end());
        }

        inline params::missing_params_iterator params::end_missing() const
        {
            return detail::iterators::make_missing_params_iterator(this->end(), this->end());
        }

        inline bool params::exists(const std::string& name) const
        {
            possibly_parse();
            options_map_type::const_iterator it=optmap_.find(name);
            return (it!=optmap_.end()) && !detail::is_option_missing(it->second);
        }

        template <typename T>
        inline bool params::exists(const std::string& name) const
        {
            possibly_parse();
            options_map_type::const_iterator it=optmap_.find(name);
            return (it!=optmap_.end()) && (it->second).is_convertible<T>();
        }

        inline bool params::defaulted(const std::string& name) const
        {
            possibly_parse();
            // FIXME: the implementation via set is a quick hack
            return exists(name) && defaulted_options_.count(name)!=0;
        }

        inline bool params::defined(const std::string& name) const
        {
            possibly_parse(); // it fills optmap_ (FIXME: may not be needed actually?)
            return optmap_.count(name)!=0 || descr_map_.count(name)!=0;
        }

        inline params& params::description(const std::string& helpline)
        {
            invalidate();
            helpmsg_=helpline;
            return *this;
        }

        inline bool params::help_requested() const
        {
            possibly_parse();
            return optmap_["help"];
        }

        template <typename T>
        inline params& params::define(const std::string& optname, T defval, const std::string& a_descr)
        {
            check_validity(optname);
            invalidate();
            typedef detail::description_map_type::value_type value_type;
#ifndef NDEBUG
            bool result=
#endif
                descr_map_.insert(value_type(optname, detail::option_description_type(a_descr,defval)))
#ifndef NDEBUG
                .second;
            assert(result && "The inserted element is always new");
#else
            ;
#endif
            return *this;
        }

        template <typename T>
        inline params& params::define(const std::string& optname, const std::string& a_descr)
        {
            check_validity(optname);
            invalidate();
            typedef detail::description_map_type::value_type value_type;
#ifndef NDEBUG
            bool result=
#endif
                descr_map_.insert(value_type(optname, detail::option_description_type(a_descr, (T*)0)))
#ifndef NDEBUG
                .second;
            assert(result && "The inserted element is always new");
#else
            ;
#endif
            return *this;
        }

/// Define a "trigger" option
        inline params& params::define(const std::string& optname, const std::string& a_descr)
        {
            check_validity(optname);
            invalidate();
            typedef detail::description_map_type::value_type value_type;
#ifndef NDEBUG
            bool result=
#endif
                descr_map_.insert(value_type(optname, detail::option_description_type(a_descr)))
#ifndef NDEBUG
                .second;
            assert(result && "The inserted element is always new");
#else
            ;
#endif
            return *this;
        }

        /// Used in apply() and foreach()
        template <typename F>
        struct params_apply_visitor : public boost::static_visitor<>
        {
            const std::string& name_;
            const detail::option_description_type& opt_descr_;
            const F& f_;

            params_apply_visitor(const std::string& name, const detail::option_description_type& opt_descr, const F& f) :
                name_(name), opt_descr_(opt_descr), f_(f) {}

            /// Applying to a None type --- always fails
            void operator()(const detail::None&) const
            {
                throw option_type::visitor_none_used("Attempt to use uninitialized option value");
            }

            /// Even though this overload must be defined, it should never be called
            void operator()(const boost::optional<detail::trigger_tag>& val) const
            {
              assert(false);
            }

            /// Triggers are to be treated specially, because T == bool for them,
            /// but opt_descr_.deflt_ contains boost::optional<trigger_tag>

            /// Called by apply_visitor()
            void operator()(const boost::optional<bool>& val) const
            {
                if(opt_descr_.is_trigger())
                    f_(name_, val, boost::optional<bool>(boost::none), opt_descr_.descr_);
                else
                    f_(name_, val, boost::get<boost::optional<bool> >(opt_descr_.deflt_), opt_descr_.descr_);
            }

            /// Called by apply_visitor()
            template <typename T>
            void operator()(const boost::optional<T>& val) const
            {
                f_(name_, val, boost::get<boost::optional<T> >(opt_descr_.deflt_), opt_descr_.descr_);
            }
        };

        template <typename F>
        inline void apply(const params& opts, const std::string& optname, F const& f)
        {
            opts.possibly_parse();

            // opts.optmap_ is mutable, so we have to use const_cast to make compiler choose the right overload of operator[]
            const options_map_type::mapped_type& o = const_cast<options_map_type const&>(opts.optmap_)[optname];
            // There is no const operator[] in description_map_type
            detail::description_map_type::const_iterator d_it = opts.descr_map_.find(optname);

            params_apply_visitor<F> v(optname, d_it->second, f);
            boost::apply_visitor(v, o.val_);
        }

        template <typename F>
        inline void foreach(const params& opts, F const& f)
        {
            opts.possibly_parse();

            for(options_map_type::const_iterator o_it = opts.optmap_.begin(); o_it != opts.optmap_.end(); o_it++)
            {
                detail::description_map_type::const_iterator d_it = opts.descr_map_.find(o_it->first);
                params_apply_visitor<F> v(o_it->first, d_it->second, f);
                boost::apply_visitor(v, o_it->second.val_);
            }
        }

    } // params_ns::
} // alps::

#endif /* ALPS_PARAMS_PARAMS_IMPL_HPP_8243f4d88828473688ec07edbb7a9f76 */
