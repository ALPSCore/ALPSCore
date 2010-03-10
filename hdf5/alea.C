#include<iostream>
#include <alps/alea.h>
#include <iostream>
#include <boost/filesystem/operations.hpp>

using namespace alps;
int main(void){

//	return 0;

  // int N=5000;
  // int M=50;
  ObservableSet measurements, measurements2;
  measurements<<SimpleObservable<double,DetailedBinning<double> >("double_detailed");
  measurements<<SimpleObservable<double,SimpleBinning<double> >("double_simple");
  measurements<<SimpleObservable<double,NoBinning<double> >("double_none");

  measurements<<SimpleObservable<float,DetailedBinning<float> >("float_detailed");
  measurements<<SimpleObservable<float,SimpleBinning<float> >("float_simple");
  measurements<<SimpleObservable<float,NoBinning<float> >("float_none");
  
  measurements<<SignedObservable<SimpleObservable<double,DetailedBinning<double> > >("signed_double_detailed");
  measurements<<SignedObservable<SimpleObservable<double,SimpleBinning<double> > >("signed_double_simple");
  measurements<<SignedObservable<SimpleObservable<double,NoBinning<double> > >("signed_double_none");

  measurements<<SignedObservable<SimpleObservable<double,DetailedBinning<double> > >("signed_float_detailed");
  measurements<<SignedObservable<SimpleObservable<double,SimpleBinning<double> > >("signed_float_simple");
  measurements<<SignedObservable<SimpleObservable<double,NoBinning<double> > >("signed_float_none");
  
  measurements<<SimpleObservable<std::valarray<double>,DetailedBinning<std::valarray<double> > >("double_detailed_vec");
  measurements<<SimpleObservable<std::valarray<double>,SimpleBinning<std::valarray<double> > >("double_simple_vec");
  measurements<<SimpleObservable<std::valarray<double>,NoBinning<std::valarray<double> > >("double_none_vec");
  
  measurements<<SignedObservable<SimpleObservable<std::valarray<double>,DetailedBinning<std::valarray<double> > > >("signed_double_detailed_vec");
  measurements<<SignedObservable<SimpleObservable<std::valarray<double>,SimpleBinning<std::valarray<double> > > >("signed_double_simple_vec");
  measurements<<SignedObservable<SimpleObservable<std::valarray<double>,NoBinning<std::valarray<double> > > >("signed_double_none_vec");

  /*measurements<<SimpleObservable<std::valarray<float>,DetailedBinning<std::valarray<float> > >("float_detailed_vec");
  measurements<<SimpleObservable<std::valarray<float>,SimpleBinning<std::valarray<float> > >("float_simple_vec");
  measurements<<SimpleObservable<std::valarray<float>,NoBinning<std::valarray<float> > >("float_none_vec");
  */
  measurements<<RealObservable("Sign");
  for(int i=0;i<100;++i){
    measurements["double_detailed"]<<0.5;
    measurements["double_simple"]<<0.5;
    measurements["double_none"]<<0.9;

    measurements["signed_double_detailed"]<<0.8;
    measurements["signed_double_simple"]<<0.7;
    measurements["signed_double_none"]<<0.6;
    measurements["Sign"]<<-1.;

    measurements["float_detailed"]<<(float)(0.3);
    measurements["float_simple"]<<(float)(0.3);
    measurements["float_none"]<<(float)(0.2);

    /*measurements["signed_float_detailed"]<<(float)(drand48());
    measurements["signed_float_simple"]<<(float)(drand48());
    measurements["signed_float_none"]<<(float)(drand48());*/

    std::valarray<double> double_vec(0.1, 2);
    measurements["double_detailed_vec"]<<double_vec;
    measurements["double_simple_vec"]<<double_vec;
    measurements["double_none_vec"]<<double_vec;
    
    measurements["signed_double_detailed_vec"]<<double_vec;
    measurements["signed_double_simple_vec"]<<double_vec;
    measurements["signed_double_none_vec"]<<double_vec;
  }
  
  {
    hdf5::oarchive ar("alea.h5");
    ar << make_pvp("/results", measurements);
  }

  measurements2<<SimpleObservable<double,DetailedBinning<double> >("double_detailed");
  measurements2<<SimpleObservable<double,SimpleBinning<double> >("double_simple");
  measurements2<<SimpleObservable<double,NoBinning<double> >("double_none");

  measurements2<<SignedObservable<SimpleObservable<double,DetailedBinning<double> > >("signed_double_detailed");
  measurements2<<SignedObservable<SimpleObservable<double,SimpleBinning<double> > >("signed_double_simple");
  measurements2<<SignedObservable<SimpleObservable<double,NoBinning<double> > >("signed_double_none");
  measurements2<<RealObservable("Sign");

  measurements2<<SimpleObservable<float,DetailedBinning<float> >("float_detailed");
  measurements2<<SimpleObservable<float,SimpleBinning<float> >("float_simple");
  measurements2<<SimpleObservable<float,NoBinning<float> >("float_none");
  
  measurements2<<SimpleObservable<std::valarray<double>,DetailedBinning<std::valarray<double> > >("double_detailed_vec");
  measurements2<<SimpleObservable<std::valarray<double>,SimpleBinning<std::valarray<double> > >("double_simple_vec");
  measurements2<<SimpleObservable<std::valarray<double>,NoBinning<std::valarray<double> > >("double_none_vec");
  
  measurements2<<SignedObservable<SimpleObservable<std::valarray<double>,DetailedBinning<std::valarray<double> > > >("signed_double_detailed_vec");
  measurements2<<SignedObservable<SimpleObservable<std::valarray<double>,SimpleBinning<std::valarray<double> > > >("signed_double_simple_vec");
  measurements2<<SignedObservable<SimpleObservable<std::valarray<double>,NoBinning<std::valarray<double> > > >("signed_double_none_vec");
  
  {
    hdf5::iarchive ar("alea.h5");
    ar >> make_pvp("/results", measurements2);
  }
  boost::filesystem::remove(boost::filesystem::path("alea.h5"));

  //SimpleObservable<double,DetailedBinning<double> >dd_obs("double_detailed");
  /*SimpleObservable<double,SimpleBinning<double> >ds_obs("double_simple");
  SimpleObservable<double,NoBinning<double> > dn_obs("double_none");*/
 
  //dd_obs.read_hdf5("/tmp/rugu.hdf5");
  //std::cout<<"dd obs: "<<dd_obs.mean()<<" "<<dd_obs.error()<<std::endl;

  RealObsevaluator double_detailed_eval_1(measurements["double_detailed"]);
  RealObsevaluator double_detailed_eval_2(measurements2["double_detailed"]);
  RealObsevaluator double_simple_eval_1(measurements["double_simple"]);
  RealObsevaluator double_simple_eval_2(measurements2["double_simple"]);
  RealObsevaluator double_none_eval_1(measurements["double_none"]);
  RealObsevaluator double_none_eval_2(measurements2["double_none"]);

  std::cout<<"mean double detailed: "<<double_detailed_eval_1.mean()-double_detailed_eval_2.mean()<<std::endl;
  std::cout<<"error double detailed: "<<double_detailed_eval_1.error()-double_detailed_eval_2.error()<<std::endl;
  std::cout<<"mean double simple: "<<double_simple_eval_1.mean()-double_simple_eval_2.mean()<<std::endl;
  std::cout<<"error double simple: "<<double_simple_eval_1.error()-double_simple_eval_2.error()<<std::endl;
  std::cout<<"mean double none: "<<double_none_eval_1.mean()-double_none_eval_2.mean()<<std::endl;
  std::cout<<"error double none: "<<double_none_eval_1.error()-double_none_eval_2.error()<<std::endl;
  
  SimpleObservableEvaluator<float> float_detailed_eval_1(measurements["float_detailed"]);
  SimpleObservableEvaluator<float> float_detailed_eval_2(measurements2["float_detailed"]);
  SimpleObservableEvaluator<float> float_simple_eval_1(measurements["float_simple"]);
  SimpleObservableEvaluator<float> float_simple_eval_2(measurements2["float_simple"]);
  SimpleObservableEvaluator<float> float_none_eval_1(measurements["float_none"]);
  SimpleObservableEvaluator<float> float_none_eval_2(measurements2["float_none"]);

  std::cout<<"mean float detailed: "<<float_detailed_eval_1.mean()-float_detailed_eval_2.mean()<<std::endl;
  std::cout<<"error float detailed: "<<float_detailed_eval_1.error()-float_detailed_eval_2.error()<<std::endl;
  std::cout<<"mean float simple: "<<float_simple_eval_1.mean()-float_simple_eval_2.mean()<<std::endl;
  std::cout<<"error float simple: "<<float_simple_eval_1.error()-float_simple_eval_2.error()<<std::endl;
  std::cout<<"mean float none: "<<float_none_eval_1.mean()-float_none_eval_2.mean()<<std::endl;
  std::cout<<"error float none: "<<float_none_eval_1.error()-float_none_eval_2.error()<<std::endl;
  
  RealObsevaluator sdouble_detailed_eval_1(measurements["signed_double_detailed"]);
  RealObsevaluator sdouble_detailed_eval_2(measurements2["signed_double_detailed"]);
  RealObsevaluator sdouble_simple_eval_1(measurements["signed_double_simple"]);
  RealObsevaluator sdouble_simple_eval_2(measurements2["signed_double_simple"]);
  RealObsevaluator sdouble_none_eval_1(measurements["signed_double_none"]);
  RealObsevaluator sdouble_none_eval_2(measurements2["signed_double_none"]);

  std::cout<<"signed mean double detailed: "<<sdouble_detailed_eval_1.mean()-sdouble_detailed_eval_2.mean()<<std::endl;
  std::cout<<"signed error double detailed: "<<sdouble_detailed_eval_1.error()-sdouble_detailed_eval_2.error()<<std::endl;
  std::cout<<"signed mean double simple: "<<sdouble_simple_eval_1.mean()-sdouble_simple_eval_2.mean()<<std::endl;
  std::cout<<"signed error double simple: "<<sdouble_simple_eval_1.error()-sdouble_simple_eval_2.error()<<std::endl;
  std::cout<<"signed mean double none: "<<sdouble_none_eval_1.mean()-sdouble_none_eval_2.mean()<<std::endl;
  std::cout<<"signed error double none: "<<sdouble_none_eval_1.error()-sdouble_none_eval_2.error()<<std::endl;
 
  /*SimpleObservableEvaluator<float> sfloat_detailed_eval_1(measurements["signed_float_detailed"]);
  SimpleObservableEvaluator<float> sfloat_detailed_eval_2(measurements2["signed_float_detailed"]);
  SimpleObservableEvaluator<float> sfloat_simple_eval_1(measurements["signed_float_simple"]);
  SimpleObservableEvaluator<float> sfloat_simple_eval_2(measurements2["signed_float_simple"]);
  SimpleObservableEvaluator<float> sfloat_none_eval_1(measurements["signed_float_none"]);
  SimpleObservableEvaluator<float> sfloat_none_eval_2(measurements2["signed_float_none"]);

  std::cout<<"mean float detailed: "<<sfloat_detailed_eval_1.mean()<<" "<<sfloat_detailed_eval_2.mean()<<std::endl;
  std::cout<<"error float detailed: "<<sfloat_detailed_eval_1.error()<<" "<<sfloat_detailed_eval_2.error()<<std::endl;
  std::cout<<"mean float simple: "<<sfloat_simple_eval_1.mean()<<" "<<sfloat_simple_eval_2.mean()<<std::endl;
  std::cout<<"error float simple: "<<sfloat_simple_eval_1.error()<<" "<<sfloat_simple_eval_2.error()<<std::endl;
  std::cout<<"mean float none: "<<sfloat_none_eval_1.mean()<<" "<<sfloat_none_eval_2.mean()<<std::endl;
  std::cout<<"error float none: "<<sfloat_none_eval_1.error()<<" "<<sfloat_none_eval_2.error()<<std::endl;*/
  
  RealVectorObsevaluator double_detailed_vec_eval_1(measurements["double_detailed_vec"]);
  RealVectorObsevaluator double_detailed_vec_eval_2(measurements2["double_detailed_vec"]);
  RealVectorObsevaluator double_simple_vec_eval_1(measurements["double_simple_vec"]);
  RealVectorObsevaluator double_simple_vec_eval_2(measurements2["double_simple_vec"]);
  RealVectorObsevaluator double_none_vec_eval_1(measurements["double_none_vec"]);
  RealVectorObsevaluator double_none_vec_eval_2(measurements2["double_none_vec"]);

  std::cout<<"mean double detailed vec: "<<double_detailed_vec_eval_1.mean()[0]-double_detailed_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"error double detailed vec: "<<double_detailed_vec_eval_1.error()[0]-double_detailed_vec_eval_2.error()[0]<<std::endl;
  std::cout<<"mean double simple vec: "<<double_simple_vec_eval_1.mean()[0]-double_simple_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"error double simple vec: "<<double_simple_vec_eval_1.error()[0]-double_simple_vec_eval_2.error()[0]<<std::endl;
  std::cout<<"mean double none vec: "<<double_none_vec_eval_1.mean()[0]-double_none_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"error double none vec: "<<double_none_vec_eval_1.error()[0]-double_none_vec_eval_2.error()[0]<<std::endl;
 
  RealVectorObsevaluator sdouble_detailed_vec_eval_1(measurements["signed_double_detailed_vec"]);
  RealVectorObsevaluator sdouble_detailed_vec_eval_2(measurements2["signed_double_detailed_vec"]);
  RealVectorObsevaluator sdouble_simple_vec_eval_1(measurements["signed_double_simple_vec"]);
  RealVectorObsevaluator sdouble_simple_vec_eval_2(measurements2["signed_double_simple_vec"]);
  RealVectorObsevaluator sdouble_none_vec_eval_1(measurements["signed_double_none_vec"]);
  RealVectorObsevaluator sdouble_none_vec_eval_2(measurements2["signed_double_none_vec"]);

  std::cout<<"signed mean double detailed vec: "<<sdouble_detailed_vec_eval_1.mean()[0]-sdouble_detailed_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"signed error double detailed vec: "<<sdouble_detailed_vec_eval_1.error()[0]-sdouble_detailed_vec_eval_2.error()[0]<<std::endl;
  std::cout<<"signed mean double simple vec: "<<sdouble_simple_vec_eval_1.mean()[0]-sdouble_simple_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"signed error double simple vec: "<<sdouble_simple_vec_eval_1.error()[0]-sdouble_simple_vec_eval_2.error()[0]<<std::endl;
  std::cout<<"signed mean double none vec: "<<sdouble_none_vec_eval_1.mean()[0]-sdouble_none_vec_eval_2.mean()[0]<<std::endl;
  std::cout<<"signed error double none vec: "<<sdouble_none_vec_eval_1.error()[0]-sdouble_none_vec_eval_2.error()[0]<<std::endl;
 
}
