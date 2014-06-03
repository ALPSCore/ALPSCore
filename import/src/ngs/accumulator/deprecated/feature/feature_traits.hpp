/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

//The whole file holds the meta-template code to calculate, 
//from what base_types one needs to derive in order to get the 
//requested feautures

#ifndef ALPS_NGS_ALEA_FEATURES_FEATURE_TRAITS_HPP
#define ALPS_NGS_ALEA_FEATURES_FEATURE_TRAITS_HPP

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>

#include <iostream>

namespace alps {
    namespace accumulator {
        template<typename T, typename W = void> 
        struct type_holder {};


        namespace detail {
        // = = = = = = M E T A   T E M P A L T E   L I S T = = = = = = = = = = =
            template<typename stored_type, typename next_list_item> struct ListItem {
                typedef stored_type type;
                typedef next_list_item next;
            };
            
            struct ListEnd {};  //is used to mark the end of a list

        // = = = = = = = R E M O V E   V O I D   I N   L I S T = = = = = = = = = =
            template<typename list> struct RemoveVoid {
                typedef list type;
            };
            
            template<
                    typename stored_type
                  , typename next_list_item
                  > 
            struct RemoveVoid<ListItem<stored_type, next_list_item> > {
                typedef ListItem< stored_type, typename RemoveVoid<next_list_item>::type> type;
            };
            
            template<typename next_list_item> struct RemoveVoid<ListItem<void, next_list_item> > {
                typedef typename RemoveVoid<next_list_item>::type type;
            };

        // = = = = = = C O N S T R U C T   L I S T = = = = = = = = = = =
            template<
                  typename _0  = void
                , typename _1  = void
                , typename _2  = void
                , typename _3  = void
                , typename _4  = void
                , typename _5  = void
                , typename _6  = void
                , typename _7  = void
                , typename _8  = void
                , typename _9  = void
                , typename _10  = void
                , typename _11  = void
                , typename _12  = void
                , typename _13  = void
                , typename _14  = void
                , typename _15  = void
                , typename _16  = void
                , typename _17  = void
                , typename _18  = void
                , typename _19  = void
                , typename _20  = void
                , typename _21  = void
            > struct MakeList {
                typedef typename RemoveVoid<
                        ListItem<_0, 
                         ListItem<_1, 
                          ListItem<_2, 
                           ListItem<_3, 
                            ListItem<_4, 
                             ListItem<_5, 
                              ListItem<_6, 
                               ListItem<_7,
                                ListItem<_8, 
                                 ListItem<_9, 
                                  ListItem<_10, 
                                   ListItem<_11, 
                                    ListItem<_12, 
                                     ListItem<_13, 
                                      ListItem<_14, 
                                       ListItem<_15, 
                                        ListItem<_16, 
                                         ListItem<_17, 
                                          ListItem<_18, 
                                           ListItem<_19, 
                                            ListItem<_20, 
                                             ListItem<_21, 
                                  ListEnd
                         > > > > >  > > > > >
                         > > > > >  > > > > > 
                         > >
                        >::type type;
            };

        // = = = = = = = C O N C A T   T W O   L I S T S = = = = = = = = = =
            template <typename list1, typename list2> struct ConcatinateLists {
                typedef ListItem< typename list1::type, typename ConcatinateLists<typename list1::next, list2>::type> type;
            };
            
            template <typename list2> struct ConcatinateLists<ListEnd, list2> {
                typedef list2 type;
            };

        // = = = = = = = U N I Q U E   L I S T   W A L K E R = = = = = = = = = =
            //walks through the list and eliminates target
            template <typename target, typename list> struct UniqueListWalker {
                typedef ListItem< 
                    typename list::type
                  , typename UniqueListWalker<target, typename list::next>::type 
                > type;
            };
            
            template <typename target, typename list> struct UniqueListWalker<target, ListItem<target, list> > {
                typedef typename UniqueListWalker<target, list>::type type;
            };
            
            template <typename target> struct UniqueListWalker<target, ListEnd>{
                typedef ListEnd type;
            };
        // = = = = = = U N I Q U E   L I S T = = = = = = = = = = =
        //bc ValueType is at first position one uses typename list::type and after that UniqueList
            template <typename list> struct UniqueList {
                typedef ListItem<
                      typename list::type
                    , typename UniqueList<
                          typename UniqueListWalker<
                                typename list::type
                              , typename list::next
                          >::type
                      >::type
                > type;
            };
            
            template <> struct UniqueList<ListEnd> {
                typedef ListEnd type;
            };
            
        // = = = = = = = F I N D   V A L U E   T Y P E = = = = = = = = = =
            template<typename list> struct FindTypeHolder {
                typedef typename FindTypeHolder<typename list::next>::type type;
            };
            
            template<
                      typename stored_value_type
                    , typename stored_weight_type
                    , typename next_list_item
                    > 
            struct FindTypeHolder<
                                  ListItem<type_holder<stored_value_type, stored_weight_type>
                                         , next_list_item> 
                                > 
            {
                typedef type_holder<stored_value_type, stored_weight_type> type;
            };

            template<> struct FindTypeHolder<ListEnd> { //no type-holder-type found
                BOOST_STATIC_ASSERT_MSG(true, "No ValueType added!");
            };
            
            //takes a list and frontInserts the ValueType 
            template<typename list> struct TypeHolderFirst {
                typedef ListItem<typename FindTypeHolder<list>::type, list> type;
            };

        // = = = = = = = D E P E N D E N C I E S = = = = = = = = = =
            template<typename T> struct Dependencies { //trait that is overloaded for each properties
                typedef MakeList<>::type type;
            };

        // = = = = = = = = R E S O L V E   D E P E N D E N C I E S = = = = = = = = =
            template <typename list> struct ResolveDependencies {
                typedef typename ConcatinateLists<
                      typename ResolveDependencies< //resolve dependencies of the dependencies
                          typename Dependencies<typename list::type>::type
                      >::type
                    , ListItem<
                          typename list::type, 
                          typename ResolveDependencies<typename list::next>::type
                    >
                >::type type;
            };
            
            template <> struct ResolveDependencies<ListEnd> {
                typedef ListEnd type;
            };

        // = = = = = = D E R I V E   F R O M   I M P L E M E N T A T I O N S   F O R   A C C U M U L A T O R = = = = = = = = = = =
            template<typename property, typename base_type> struct AccumulatorImplementation {};

            template<typename list, typename base_type> struct DeriveAccumulatorProperties {
                typedef typename DeriveAccumulatorProperties<
                    typename list::next, AccumulatorImplementation<typename list::type, base_type> //the base_type is expanded here
                >::type type;
            };
            template<typename base_type> struct DeriveAccumulatorProperties<ListEnd, base_type> {
                typedef base_type type; //here base_type will be AccumulatorImplementation<property1, AccumulatorImplementation<property2, ... AccumulatorImplementation<propertyN, UselessBase> > >
            };

        // = = = = = = D E R I V E   F R O M   I M P L E M E N T A T I O N S   F O R   R E S U L T = = = = = = = = = = =
            template<typename property, typename base_type> struct ResultImplementation {};

            template<typename list, typename base_type> struct DeriveResultProperties {
                typedef typename DeriveResultProperties<
                    typename list::next, ResultImplementation<typename list::type, base_type> //the base_type is expanded here
                >::type type;
            };
            template<typename base_type> struct DeriveResultProperties<ListEnd, base_type> {
                typedef base_type type; //here base_type will be DeriveResultProperties<property1, DeriveResultProperties<property2, ... DeriveResultProperties<propertyN, UselessBase> > >
            };

        // = = = = = = S T A N D A R D   B A S E = = = = = = = = = = =
            struct UselessBase {};

        }
    }
}
#endif // ALPS_NGS_ALEA_FEATURES_FEATURE_TRAITS_HPP
