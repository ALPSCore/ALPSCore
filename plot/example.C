/***************************************************************************
* ALPS/model library
*
* example/example.C
*
* $Id$
*
* Copyright (C) 2003 by Simon Trebst <trebst@itp.phys.ethz.ch>
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
* available from http://alps.comp-phys.org/. 

*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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
