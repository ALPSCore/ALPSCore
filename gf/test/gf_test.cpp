/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "gf_test.hpp"

/// This generates some "outside" data to fill momentum mesh: 4 2-d points
alps::gf::momentum_index_mesh::container_type get_data_for_momentum_mesh()
{
    alps::gf::momentum_index_mesh::container_type points(boost::extents[4][2]);
    points[0][0]=0; points[0][1]=0; 
    points[1][0]=M_PI; points[1][1]=M_PI;
    points[2][0]=M_PI; points[2][1]=0; 
    points[3][0]=0; points[3][1]=M_PI;

    return points;
}

/// This generates some "outside" data to fill real-space mesh: 4 2-d points
alps::gf::real_space_index_mesh::container_type get_data_for_real_space_mesh()
{
  alps::gf::real_space_index_mesh::container_type points(boost::extents[4][2]);
  points[0][0]=0; points[0][1]=0;
  points[1][0]=0; points[1][1]=1;
  points[2][0]=1; points[2][1]=1;
  points[3][0]=1; points[3][1]=1;

  return points;
}
