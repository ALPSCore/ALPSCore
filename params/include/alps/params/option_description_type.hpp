/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_PARAMS_OPTION_DESCRIPTION_TYPE_27836b27fafb4e60a89a59b80016bebc
#define ALPS_PARAMS_OPTION_DESCRIPTION_TYPE_27836b27fafb4e60a89a59b80016bebc

namespace alps {
    namespace params_ns {
        namespace detail {

            /// Option (parameter) description class. Used to interface with boost::program_options
            class option_description_type {
              private:
                typedef boost::program_options::options_description po_descr;
                
                std::string descr_; ///< Parameter description
                variant_all_type deflt_; ///< To keep type and defaults(if any)

                /// Visitor class to add the stored description to boost::program_options
                struct add_option_visitor: public boost::static_visitor<> {
                    po_descr& odesc_;
                    const std::string& name_;
                    const std::string& strdesc_;

                    add_option_visitor(po_descr& a_po_descr, const std::string& a_name, const std::string& a_strdesc):
                        odesc_(a_po_descr), name_(a_name), strdesc_(a_strdesc) {}

                    void operator()(const None&) const
                    {
                        throw std::logic_error("add_option_visitor is called for an object containing None: should not happen!");
                    }
                    
                    /// Called by apply_visitor(), for a optional<T> bound type
                    template <typename T>
                    void operator()(const boost::optional<T>& a_val) const
                    {
                        if (a_val) {
                            // a default value is provided
                            do_define<T>::add_option(odesc_, name_, *a_val, strdesc_);
                        } else {
                            // no default value
                            do_define<T>::add_option(odesc_, name_, strdesc_);
                        }
                    }

                    /// Called by apply_visitor(), for a trigger_tag type
                    void operator()(const boost::optional<trigger_tag>& a_val) const
                    {
                        do_define<trigger_tag>::add_option(odesc_, name_, strdesc_);
                    }
                };
                    

                /// Visitor class to set option_type instance from boost::any; visitor is used ONLY to extract type information
                struct set_option_visitor: public boost::static_visitor<> {
                    option_type& opt_;
                    const boost::any& anyval_;

                    set_option_visitor(option_type& a_opt, const boost::any& a_anyval):
                        opt_(a_opt), anyval_(a_anyval) {}

                    /// Called by apply_visitor(), for None bound type
                    void operator()(const None&) const
                    {
                        throw std::logic_error("set_option_visitor is called for an object containing None: should not happen!");
                    }
                    
                    /// Called by apply_visitor(), for a optional<T> bound type
                    template <typename T>
                    void operator()(const boost::optional<T>& a_val) const
                    {
                        if (anyval_.empty()) {
                            opt_.reset<T>();
                        } else {
                            opt_.reset<T>(boost::any_cast<T>(anyval_));
                        }
                    }

                    /// Called by apply_visitor(), for a optional<std::string> bound type
                    void operator()(const boost::optional<std::string>& a_val) const
                    {
                        if (anyval_.empty()) {
                            opt_.reset<std::string>();
                        } else {
                            // The value may contain a string or a default value, which is hidden inside string_container
                            // (FIXME: this mess of a design must be fixed).
                            const std::string* ptr=boost::any_cast<std::string>(&anyval_);
                            if (ptr) {
                                opt_.reset<std::string>(*ptr);
                            } else {
                                opt_.reset<std::string>(boost::any_cast<string_container>(anyval_));
                            }
                        }
                    }

                    /// Called by apply_visitor(), for a trigger_tag type
                    void operator()(const boost::optional<trigger_tag>& ) const
                    {
                        opt_.reset<bool>(!anyval_.empty()); // non-empty value means the option is present
                    }
                };


                /// Visitor to save the default value to an archive
                class save_visitor : public boost::static_visitor<> {
                    alps::hdf5::archive& ar_;
                    const std::string& name_;

                    // template <typename>
                    // struct is_trigger { static const bool VALUE = false; };

                    // template <>
                    // struct is_trigger<trigger_tag> { static const bool VALUE = true; };

                    
                  public:
                    save_visitor(alps::hdf5::archive& ar, const std::string& name)
                        : ar_(ar), name_(name)
                    {}

                    /// Called when the variant contains optional<T>
                    template <typename T>
                    void operator()(const boost::optional<T>& val) const
                    {
                        // if the value is present, the option has a default
                        if (val) {
                            ar_[name_] << *val;
                            ar_[name_+"@has_default"] << true;
                        } else {
                            ar_[name_] << T();
                            ar_[name_+"@has_default"] << false;
                        } 
                        ar_[name_+"@is_trigger"] << false;
                   }

                    /// Called when the variant contains optional<trigger_tag>
                    void operator()(const boost::optional<trigger_tag>& val) const
                    {
                        // trigger options do not have defaults
                        ar_[name_] << false;
                        ar_[name_+"@has_default"] << false;
                        ar_[name_+"@is_trigger"] << true;
                    }

                    /// Called when the variant contains None
                    void operator()(const None&) const
                    {
                        throw std::logic_error("option_description_type::save_visitor "
                                               "is invoked for an object containing None: "
                                               "should not happen!\n" + ALPS_STACKTRACE);
                    }
                };

                /// Class for reading the initialization info from archive
                // FIXME: simplified version exists as option_type::reader,
                //        can it be made more general? and used here?
                class reader {
                    alps::hdf5::archive& ar_;
                    const std::string& path_;
                    bool is_scalar_;
                    bool is_trigger_;
                    bool has_default_;
                    std::string descr_;
                  public:
                    reader(alps::hdf5::archive& ar, const std::string& path)
                        : ar_(ar), path_(path),
                          is_scalar_(false),
                          is_trigger_(false),
                          has_default_(false)
                    {
                        const std::string trigger_attr=path+"@is_trigger";  
                        const std::string deflt_attr=path+"@has_default";
                        const std::string descr_attr=path+"@description";

                        if (ar_.is_attribute(trigger_attr)) {
                            ar_.read(trigger_attr, is_trigger_);
                        }
                        if (ar_.is_attribute(deflt_attr)) {
                            ar_.read(deflt_attr, has_default_);
                        }
                        ar_.read(descr_attr, descr_);
                        is_scalar_=ar_.is_scalar(path);
                    }

                    template <typename T>
                    bool can_read(const T*)
                    {
                        bool ok=!is_trigger_ && is_scalar_ && ar_.is_datatype<T>(path_);
                        return ok;
                    }

                    template <typename T>
                    bool can_read(const std::vector<T>*)
                    {
                        bool ok=!is_trigger_ && !is_scalar_ && ar_.is_datatype<T>(path_);
                        return ok;
                    }

                    bool can_read(const trigger_tag*)
                    {
                        bool ok=is_trigger_ && is_scalar_ && ar_.is_datatype<bool>(path_);
                        return ok;
                    }

                    template <typename T>
                    option_description_type read(const T*)
                    {
                        if (has_default_) {
                            T defval;
                            ar_[path_] >> defval;
                            return option_description_type(descr_, defval);
                        } else {
                            return option_description_type(descr_, (T*)0);
                        }
                    }

                    option_description_type read(const trigger_tag*)
                    {
                        if (!is_trigger_) {
                            throw std::logic_error("Invalid attemp to read option at path '"+
                                                   path_ + "' as a \"trigger\" option.\n"
                                                   + ALPS_STACKTRACE);
                        }
                        return option_description_type(descr_);
                    }
                };


#ifdef ALPS_HAVE_MPI
                // FIXME: copy&paste from option_type. Generalize and refactor out!
                class broadcast_send_visitor : public boost::static_visitor<> {
                    const alps::mpi::communicator& comm_;
                    const int root_;
                    public:
                    broadcast_send_visitor(const alps::mpi::communicator& c, int rt)
                        : comm_(c), root_(rt)
                    { }

                    // should work for optional<trigger_tag> also
                    template <typename T>
                    void operator()(const T& val) const {
                        // FIXME: if we make 2 versions of broadcast, sending and receiving...
                        assert(comm_.rank()==root_ && "Broadcast send from non-root?");
                        // FIXME: ...this cast won't be needed
                        alps::mpi::broadcast(comm_, const_cast<T&>(val), root_);
                    }

                    void operator()(const None&) const {
                        throw std::logic_error("Attempt to option_descripton_type::broadcast() None. Should not happen.\n"
                                               + ALPS_STACKTRACE);
                    }

                };
#endif /* ALPS_HAVE_MPI */


            public:
                /// Constructor for description without the default
                template <typename T>
                option_description_type(const std::string& a_descr, T*): descr_(a_descr), deflt_(boost::optional<T>(boost::none))
                { }

                /// Constructor for description with default
                template <typename T>
                option_description_type(const std::string& a_descr, T a_deflt): descr_(a_descr), deflt_(boost::optional<T>(a_deflt)) 
                { }

                /// Constructor for a trigger option
                option_description_type(const std::string& a_descr): descr_(a_descr), deflt_(boost::optional<trigger_tag>(trigger_tag())) 
                { }

                /// Factory method for loading from archive
                // FIXME: move it to *.cpp? template it over ar type?
                static option_description_type get_loaded(alps::hdf5::archive& ar, const std::string& key)
                {
                    reader rd(ar,key);
                    
                    // macro: try reading, return if ok
#define ALPS_LOCAL_TRY_LOAD(_r_,_d_,_type_)                           \
                    if (rd.can_read((_type_*)0)) return rd.read((_type_*)0);

                    // try reading for each defined type
                     BOOST_PP_SEQ_FOR_EACH(ALPS_LOCAL_TRY_LOAD, X, ALPS_PARAMS_DETAIL_ALLTYPES_SEQ);
#undef ALPS_LOCAL_TRY_LOAD
                    
                    throw std::runtime_error("No matching payload type in the archive "
                                             "for `option_description_type` for "
                                             "path='" + key + "'");
                }

                /// Adds to program_options options_description
                void add_option(boost::program_options::options_description& a_po_desc, const std::string& a_name) const
                {
                    boost::apply_visitor(add_option_visitor(a_po_desc,a_name,descr_), deflt_);
                }

                /// Sets option_type instance to a correct value extracted from boost::any
                void set_option(option_type& opt, const boost::any& a_val) const
                {
                    boost::apply_visitor(set_option_visitor(opt, a_val), deflt_);
                }                

                // Note the signature is different from a regular save()
                void save(hdf5::archive& ar, const std::string& name) const
                {
                    // throw std::logic_error("option_description_type::save() not implemented yet");
                    boost::apply_visitor(save_visitor(ar,name), deflt_);
                    ar[name+"@description"] << descr_;
                }

                // void load(hdf5::archive& ar)
                // {
                //     // throw std::logic_error("option_description_type::load() not implemented yet");
                //     ar["alps::params::option_description_type::descr_"] >> descr_;
                //     ar["alps::params::option_description_type::deflt_"] >> deflt_;
                // }

                /// Default ctor for internal use (in factory methods)
                // FIXME: had to make it public for map broadcast
                option_description_type() {}
                
#ifdef ALPS_HAVE_MPI
                // FIXME: copy&paste from option_type. Generalize and factor out!
                void broadcast(const alps::mpi::communicator& comm, int root)
                {
                    alps::mpi::broadcast(comm, descr_, root);
                    int root_which=deflt_.which();
                    alps::mpi::broadcast(comm, root_which, root);
                    if (root_which==0) { // CAUTION: relies of None being the first type!
                        if (comm.rank()==root) {
                            // Do nothing
                        } else {
                            deflt_=None();
                        }
                    } else { // not-null
                        if (comm.rank()==root) {
                            boost::apply_visitor(broadcast_send_visitor(comm,root), deflt_);
                        } else { // slave rank
                            // CAUTION: Fragile code!
#define ALPS_LOCAL_TRY_TYPE(_r_,_d_,_type_) {                           \
                                boost::optional<_type_> buf;            \
                                variant_all_type trial(buf);            \
                                if (trial.which()==root_which) {        \
                                    alps::mpi::broadcast(comm, buf, root); \
                                    deflt_=buf;                         \
                                }                                       \
                            } /* end macro */
                        
                            BOOST_PP_SEQ_FOR_EACH(ALPS_LOCAL_TRY_TYPE, X, ALPS_PARAMS_DETAIL_ALLTYPES_SEQ);
#undef ALPS_LOCAL_TRY_TYPE
                            assert(deflt_.which()==root_which && "The `which` value must be the same as on root");
                        } // done with slave rank 
                    }
                }
#endif
                
            };

            typedef std::map<std::string, option_description_type> description_map_type;

        } // detail::
    } // params_ns::

#ifdef ALPS_HAVE_MPI
    namespace mpi {
        inline
        void broadcast(const alps::mpi::communicator &comm, alps::params_ns::detail::option_description_type& val, int root)
        {
            val.broadcast(comm, root);
        }
    }
#endif

    
} // alps::

#endif /* ALPS_PARAMS_OPTION_DESCRIPTION_TYPE_27836b27fafb4e60a89a59b80016bebc */
