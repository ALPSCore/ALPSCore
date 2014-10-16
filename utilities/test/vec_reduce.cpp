/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gtest/gtest.h"
#include"alps/utilities/boost_mpi.hpp"
#include"alps/utilities/boost_array_mpi.hpp"

TEST(reduce, main)
{
#ifndef ALPS_HAVE_MPI
  return;
#else
    std::vector<double> vec_1(3,1.);
    std::vector<double> vec_2(3,1.);
    std::vector<double> vec_3(3,0.);
    boost::mpi::environment env;
    boost::mpi::communicator comm;
    int root=0;
    int size=comm.size();
    int rank=comm.rank();


   //EG TODO: debug this reduce. It seems to missbehave.
   //reduce(comm, vec_1, std::plus<double>(), root);    
   //if(rank==0)
   //  for(std::vector<double>::const_iterator it=vec_1.begin(); it!=vec_1.end();++it)
   //    ASSERT_EQ(*it, size);
   reduce(comm, vec_2, vec_3, std::plus<double>(), root);    
   if(rank==0)
     for(std::vector<double>::const_iterator it=vec_3.begin(); it!=vec_3.end();++it)
       ASSERT_EQ(*it, size);
#endif
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

