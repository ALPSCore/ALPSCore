/***************************************************************************
* ALPS/model library
*
* example/example.C
*
* $Id$
*
* Copyright (C) 2003 by Simon Trebst <trebst@itp.phys.ethz.ch>
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include "plot.h"

double myrandom() { return -0.5 + double(std::rand())/RAND_MAX; }

int main(int argc, char* argv[]) {
  // create plot and specify title and labels
  alps::plot::Plot<double> MyPlot;
  MyPlot.set_name("Some fictious plot for some fictious model");
  MyPlot.set_labels("Temperature","Energy");
  MyPlot.show_legend(true);

  // create first set
  alps::plot::Set<double> EnergySet(alps::plot::xy);  
  EnergySet << "Free energy versus temperature";
  for(int i=0; i<10; ++i) {
    double x = i, y = .5+2.*i+myrandom();
    EnergySet << boost::tuples::make_tuple(x,y);
  }
  
  // create second set
  alps::plot::Set<double> GapSet(alps::plot::xdxydy);  
  GapSet << "Gap versus temperature";
  for(int i=0; i<10; ++i) {
    double x = i, dx = (i+1)*myrandom();
    double y = 12-i+myrandom(), dy = myrandom();
    GapSet << boost::tuples::make_tuple(x,dx,y,dy);
  }

  // write sets to plot
  MyPlot << EnergySet << GapSet;
  
  // output plot
  std::cout << MyPlot;
}   // main