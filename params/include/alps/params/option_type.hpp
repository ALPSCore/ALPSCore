/** Designing a suitable type to hold parameters */

/** Requirements:

    1. Can hold `None`, a scalar type, a vector type.

    2. `None` cannot be assigned to anything.

    3. Vector type can be assigned only to the same type.

    4. Scalar type can be assigned to any compatible (by implicit cast) type.

    5. Anything can be assigned to `None`.

    6. Only the same type can be assigned to any vector or scalar type.

    7. Models of scalar type: `int`, `double`, `bool`.

    8. Models of vector type: `std::string`, `std::vector<T>`

    9. The parameter must hold its name (and possibly typename?), for error reporting purposes

*/

#ifndef ALPS_PARAMS_OPTION_TYPE_INCLUDED
#define ALPS_PARAMS_OPTION_TYPE_INCLUDED

#include <iostream>
#include <stdexcept>

#include "boost/variant.hpp"
#include "boost/utility.hpp" // for enable_if
#include "boost/type_traits.hpp" // for is_convertible

#include "boost/serialization/base_object.hpp"
#include "boost/serialization/map.hpp"
#include "boost/serialization/optional.hpp"
#include "boost/serialization/variant.hpp"


#include "alps/params/param_types.hpp" // Sequences of supported types

namespace alps {
    namespace params_ns {
        
        class option_type {

            friend class option_description_type;  // to interface with boost::program_options

            public: // FIXME: not everything is public

            /// "Empty value" type
            typedef detail::None None;

            variant_all_type val_; ///< Value of the option

            std::string name_; ///< The option name (FIXME: make it "functionally const")

            // std::string type_name_; ///< The option type

            /// Exception to be thrown by visitor class: type mismatch
            struct visitor_type_mismatch: public std::runtime_error {
                visitor_type_mismatch(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };
            
            /// Exception to be thrown by visitor class: None value used
            struct visitor_none_used: public std::runtime_error {
                visitor_none_used(const std::string& a_what)
                    : std::runtime_error(a_what) {}
            };
            
            /// General exception (base class)
            class exception_base : public std::runtime_error {
                const char* const name_; ///< name of the option that caused the error
            public:
                exception_base(const std::string& a_name, const std::string& a_reason)
                    : std::runtime_error("Option '"+a_name+"': "+a_reason),
                      name_(a_name.c_str())
                {}

                std::string name() const { return name_; }
            };
            
            /// Exception for mismatching types assignment
            struct type_mismatch : public exception_base {
                type_mismatch(const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {};
            };

            /// Exception for using uninitialized option value
            struct uninitialized_value : public exception_base {
                uninitialized_value (const std::string& a_name, const std::string& a_reason)
                    : exception_base(a_name, a_reason) {};
            };
            
            /// Visitor to assign a value of type T to a variant containing type U
            template <typename T>
            struct setter_visitor: public boost::static_visitor<>
            {
                const T& rhs; ///< The rhs value to be assigned

                /// Constructor save the value to be assigned
                setter_visitor(const T& a_rhs): rhs(a_rhs) {}

                /// Called when the bound type U is the same as T
                void apply(T& lhs) const
                {
                    lhs=rhs;
                }

                /// Called when the bound type U and rhs type T are distinct types
                template <typename U>
                void apply(U& lhs) const
                {
                    std::string msg="Attempt to assign type T=";
                    msg += typeid(T).name();
                    msg += " to the option_type object containing type U=";
                    msg += typeid(U).name();
                    throw visitor_type_mismatch(msg);
                }

                /// Called when the bound type U is None (should never happen, option_type::operator=() must take care of this)
                void apply(None& lhs) const
                {
                    throw std::logic_error("Should not happen: setting an option_type object containing None");
                }

                /// Called by apply_visitor()
                template <typename U>
                void operator()(U& lhs) const
                {
                    apply(lhs);
                }
            };
      
            /// Assignment operator: assigns a value of type T
            template <typename T>
            void operator=(const T& rhs)
            {
                if (val_.which()==0) { // NOTE:Caution -- relies on None being the first type!
                    val_=rhs;
                    return;
                }

                try {
                    boost::apply_visitor(setter_visitor<T>(rhs), val_);
                } catch (visitor_type_mismatch& exc) {
                    throw type_mismatch(name_, exc.what());
                }
            }

            /// Assignment operator specialization: assigns a value of type `char*`
            void operator=(const char* rhs)
            {
                *this=std::string(rhs);
            }

            /// Visitor to get a value (with conversion): returns type T, converts from the bound type U
            template <typename T>
            struct getter_visitor: public boost::static_visitor<T> {

                /// Simplest case: the values are of the same type
                T apply(const T& val) const {
                    return val; // no conversion 
                }

    
                /// Types are convertible (Both are scalar types)
                template <typename U>
                // T apply(const U& val, typename boost::enable_if< detail::both_scalar<T,U>, bool>::type =true) const {
                T apply(const U& val, typename boost::enable_if< boost::is_convertible<U,T>, bool>::type =true) const {
                    return val; // invokes implicit conversion 
                }

                /// Types are not convertible (One of the types is not a scalar)
                template <typename U>
                // T apply(const U& val, typename boost::disable_if< detail::both_scalar<T,U>, bool>::type =true) const {
                T apply(const U& val, typename boost::disable_if< boost::is_convertible<U,T>, bool>::type =true) const {
                    throw visitor_type_mismatch(
                        std::string("Attempt to assign incompatible type U=")
                        +typeid(U).name()
                        +" to type T="
                        +typeid(T).name());
                }

                /// Extracting None type --- always fails
                T apply(const None& val) const {
                    throw visitor_none_used("Attempt to use uninitialized option value");
                }

                /// Called by apply_visitor()
                template <typename U>
                T operator()(const U& val) const {
                    return apply(val);
                }
            };

            /// Conversion operator to a generic type T
            template <typename T>
            operator T() const
            {
                try {
                    return boost::apply_visitor(getter_visitor<T>(), val_);
                } catch (visitor_type_mismatch& exc) {
                    throw type_mismatch(name_,exc.what());
                } catch (visitor_none_used& exc) {
                    throw uninitialized_value(name_, exc.what());
                }
            }

            // /// Assignment from boost::any containing type T
            // template <typename T>
            // void assign_any(const boost::any& aval)
            // {
            //     val_=boost::any_cast<T>(aval);
            // }

            // /// Constructor preserving the option name
            option_type(const std::string& a_name):
                name_(a_name) {}

            /// A fake constructor to create uninitialized object for serialization --- DO NOT USE IT!!!
            // FIXME: can i avoid it?
            option_type()
                : name_("**UNINITIALIZED**") {}
                
        private:
            friend class boost::serialization::access;

            /// Interface to serialization
            template<class Archive> void serialize(Archive & ar, const unsigned int)
            {
                ar  & val_
                    & name_;
            }
                    
        };

        /// Equality operator for option_type
        template <typename T>
        bool operator==(const option_type& lhs, T rhs)
        {
            const T& lhs_t=lhs;
            return (rhs == lhs_t);
        }

        /// Equality operator for option_type
        template <typename T>
        bool operator==(T lhs, const option_type& rhs)
        {
            return (rhs == lhs);
        }

        /// Class "map of options" (needed to ensure that option is always initialized by the name)
        class options_map_type : public std::map<std::string, option_type> {
        public:
            /// Access to a constant object
            const mapped_type& operator[](const key_type& k) const
            {
                const_iterator it=find(k);
                if (it == end()) {
                    throw std::runtime_error("Attempt to access non-existing key '"+k+"'");
                }
                return it->second;
            }

            /// Access to the map with intention to assign an element
            mapped_type& operator[](const key_type& k)
            {
                iterator it=find(k);
                if (it==end()) {
                    // it's a new element, we have to construct it here
                    value_type newpair(k, option_type(k));
                    // ...and copy it to the map, returning the ref to the inserted element
                    it=insert(end(),newpair);
                }
                // return reference to the existing or the newly-created element
                return it->second;
            }

        private:
            friend class boost::serialization::access;

            /// Interface to serialization
            template<class Archive> void serialize(Archive & ar, const unsigned int)
            {
                ar & boost::serialization::base_object< std::map<key_type,mapped_type> >(*this);
            }
            
        };

        namespace detail {

            /// Tag type to indicate vector/list parameter (FIXME: make sure it works for output too)
            template <typename T>
            struct vector_tag {};

            /// Service class calling boost::program_options::add_options(), to work around lack of function template specializations
            /// T is the option type, U is the tag type used to treat parsing of vectors/lists specially
            template <typename T, typename U=T>
            struct do_define {
                /// Add option with a default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, T defval, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              boost::program_options::value<U>()->default_value(defval),
                                              a_descr.c_str());
                }

                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              boost::program_options::value<U>(),
                                              a_descr.c_str());
                }
            };

            /// Specialization of the service do_define class to define a vector/list option 
            template <typename T>
            struct do_define< std::vector<T> > {
                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    // std::cerr << "***DEBUG: calling do_define<std::vector>() ***" << std::endl;
                    do_define< std::vector<T>, vector_tag<T> >::add_option(a_opt_descr, optname, a_descr);
                }

                /// Add option with default value: should never be called (a plug to keep boost::variant happy)
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::vector<T>& defval, const std::string& a_descr)
                {
                    throw std::logic_error("Should not happen: setting default value for vector/list parameter");
                }
            };
          
            /// Specialization of the service do_define class to define a "trigger" (parameterless) option
            template <>
            struct do_define<trigger_tag> {
                /// Add option with no default value
                static void add_option(boost::program_options::options_description& a_opt_descr,
                                       const std::string& optname, const std::string& a_descr)
                {
                    a_opt_descr.add_options()(optname.c_str(),
                                              a_descr.c_str());
                }
            };
          
            
            /// Option (parameter) description class. Used to interface with boost::program_options
            class option_description_type {
                typedef boost::program_options::options_description po_descr;
                
                std::string descr_; ///< Parameter description
                variant_all_optional_type deflt_; ///< To keep type and defaults(if any)

                /// Visitor class to add the stored description to boost::program_options
                struct add_option_visitor: public boost::static_visitor<> {
                    po_descr& odesc_;
                    const std::string& name_;
                    const std::string& strdesc_;

                    add_option_visitor(po_descr& a_po_descr, const std::string& a_name, const std::string& a_strdesc):
                        odesc_(a_po_descr), name_(a_name), strdesc_(a_strdesc) {}

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
                    void operator()(const trigger_tag& a_val) const
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

                    /// Called by apply_visitor(), for a optional<T> bound type
                    template <typename T>
                    void operator()(const boost::optional<T>& a_val) const
                    {
                        if (anyval_.empty()) {
                            opt_.val_=None();
                        } else {
                            opt_.val_=boost::any_cast<T>(anyval_);
                        }
                    }

                    /// Called by apply_visitor(), for a trigger_tag type
                    void operator()(const trigger_tag& ) const
                    {
                        opt_.val_=!anyval_.empty(); // non-empty value means the option is present
                    }
                };
                    
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
                option_description_type(const std::string& a_descr): descr_(a_descr), deflt_(trigger_tag()) 
                { }

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

                /// Fake constructor to create uninitialized object for serialization --- DO NOT USE IT!!!
                // FIXME: can i avoid it?
                option_description_type()
                    : descr_("**UNINITIALIZED**") {}
                
            private:
                friend class boost::serialization::access;

                /// Interface to serialization
                template<class Archive> void serialize(Archive & ar, const unsigned int)
                {
                    ar  & descr_
                        & deflt_;
                }
            };

            typedef std::map<std::string, option_description_type> description_map_type;

        } // detail

            
        
    } // params_ns
} // alps

// // The following is needed for the serialization interface
// namespace boost {
//     namespace serialization {
//         /// Called to reconstruct option_description_type on deserialization
//         template <typename Archive>
//         inline void load_construct_data(Archive & ar,
//                                         alps::params_ns::detail::option_description_type* self,
//                                         const unsigned int)
//         {
//             // Calling the fake constructor 
//             ::new(self) alps::params_ns::detail::option_description_type(alps::params_ns::detail::option_description_type::serialization_init_tag());
//         }

//         /// Called to reconstruct option_description_type map elements on deserialization
//         template <typename Archive>
//         inline void load_construct_data(Archive & ar,
//                                         std::pair<std::string, alps::params_ns::detail::option_description_type>* self,
//                                         const unsigned int)
//         {
//             typedef std::pair<std::string, alps::params_ns::detail::option_description_type> pair;
//             // Calling the fake constructor 
//             ::new(self) pair("**UNINITIALIZED**",
//                              alps::params_ns::detail::option_description_type(alps::params_ns::detail::option_description_type::serialization_init_tag()));
//         }

//         /// Called to reconstruct option_type on deserialization
//         template <typename Archive>
//         inline void load_construct_data(Archive & ar,
//                                         alps::params_ns::option_type* self,
//                                         const unsigned int)
//         {
//             // Calling the fake constructor 
//             ::new(self) alps::params_ns::option_type(alps::params_ns::option_type::serialization_init_tag());
//         }

//         /// Called to reconstruct option_type map element on deserialization
//         template <typename Archive>
//         inline void load_construct_data(Archive & ar,
//                                         std::pair<std::string, alps::params_ns::option_type>* self,
//                                         const unsigned int)
//         {
//             typedef std::pair<std::string, alps::params_ns::option_type> pair;
//             // Calling the fake constructor 
//             ::new(self) pair("**UNINITIALIZED**",
//                              alps::params_ns::option_type(alps::params_ns::option_type::serialization_init_tag()));
//         }
        
//     } // serialization
// } // boost

        
#endif // ALPS_PARAMS_OPTION_TYPE_INCLUDED
