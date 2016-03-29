#include <alps/mc/mcbase.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>

#include <boost/foreach.hpp>


// Simulation with deliberate chain length imbalance

class my_sim: public alps::mcbase {
    int rank_;
  public:
    typedef std::vector<double> value_type;
    // typedef double value_type;
    typedef alps::accumulators::FullBinningAccumulator<value_type> accumulator_type;

    static value_type generate() {
        return value_type(3, 1.0);
        // return value_type(1.0);
    }
    
    my_sim(const parameters_type& params, std::size_t rank)
        : alps::mcbase(params, 42+rank),
          rank_(rank)
    {
        measurements << accumulator_type("data");
    }

    void update() {}

    void measure() {
        // std::cerr << "DEBUG: i am rank=" << rank_ << std::endl;
        // number of measurements depends on rank
        for (int i=0; i<rank_*100+1; ++i) {
            measurements["data"] << generate();
        }
    }

    double fraction_completed() {
        usleep(rank_*200000);
        std::cerr << "Rank=" << rank_ << " measurements:\n" << measurements << std::endl;
        return 1.0;
    }
};

typedef alps::mcmpiadapter<my_sim> mpi_sim;

int test1(int argc, char** argv)
{
    alps::mpi::communicator comm;
    const bool is_master=(comm.rank()==0);

    usleep(comm.rank()*100000);
    std::cerr << "Rank #" << comm.rank() << " has pid " << getpid() << std::endl;
    sleep(15);

    
    alps::params p;
    if (is_master) {
        alps::params root_p(argc, (const char**)argv);
        p=root_p;
    }
    p.broadcast(comm,0);

    mpi_sim::define_parameters(p);
    
    if (is_master && p.help_requested(std::cerr)) { return 1; }

    mpi_sim sim(p, comm);

    std::string msg="Starting the simulation on rank="+boost::lexical_cast<std::string>(comm.rank())+"\n";
    std::cerr << msg << std::flush;
    
    sim.run(alps::stop_callback(1));

    msg="Collecting results on rank="+boost::lexical_cast<std::string>(comm.rank())+"\n";
    std::cerr << msg << std::flush;
    
    const mpi_sim::results_type res=sim.collect_results();

    if (is_master) {
        std::cerr << "Results:\n" << res;
    }

    return 0;
}

template <typename T>
void rectangularize(const T&) {}

template <typename T>
void rectangularize(std::vector< std::vector<T> >& vec)
{
    std::size_t mx_size=0;
    BOOST_FOREACH(std::vector<T>& val, vec) {
        // rectangularize(val);
        if (mx_size<val.size()) mx_size=val.size();
    }
    BOOST_FOREACH(std::vector<T>& val, vec) {
        val.resize(mx_size);
    }
}
    


int test2(int argc, char**argv)
{
    typedef std::vector<double> dbl_vec_type;
    typedef std::vector<dbl_vec_type> dbl_mtx_type;
    namespace ampi=alps::alps_mpi;
    using debug::operator<<;

    alps::mpi::communicator comm;

    dbl_vec_type vec1(3, 1.5);
    dbl_vec_type vec2(3, 2.25);
    dbl_mtx_type mtx1, mtx2;
    if (comm.rank()==0) {
        mtx1.resize(3);
        mtx1[0]=vec1;
        mtx1[1]=vec1;
        rectangularize(mtx1);
        ampi::reduce(comm, mtx1, mtx2, std::plus<double>(), 0);
        std::cout << "Result mtx2=" << mtx2 << std::endl;
    } else {
        mtx1.resize(3);
        mtx1[0]=vec2;
        mtx1[1]=vec2;
        mtx1[2]=vec2;
        rectangularize(mtx1);
        ampi::reduce(comm, mtx1, std::plus<double>(), 0);
    }
}

void test3()
{
    typedef std::vector<double> dbl_vec_type;
    typedef std::vector<dbl_vec_type> dbl_mtx_type;
    typedef std::vector<dbl_mtx_type> dbl_tn3_type;
    using debug::operator<<;
    
    dbl_vec_type vec1(3, 1.5);
    dbl_vec_type vec2(4, 2.25);
    dbl_mtx_type mtx1,mtx2;
    dbl_tn3_type tensor;
    

    mtx1.push_back(vec1);
    mtx1.push_back(vec1);
    mtx1.push_back(vec2);

    mtx2.push_back(vec2);
    mtx2.push_back(vec1);

    tensor.push_back(mtx1);
    tensor.push_back(mtx2);

    boolalpha(std::cout);
    std::cout << "Before: (" << alps::hdf5::is_vectorizable(mtx1) << ") mtx1=" << mtx1 << std::endl;
    rectangularize(mtx1);
    std::cout << "After: (" << alps::hdf5::is_vectorizable(mtx1) << ") mtx1=" << mtx1 << std::endl;
    
    std::cout << "Before: (" << alps::hdf5::is_vectorizable(tensor) << ") tensor=" << tensor << std::endl;
    rectangularize(tensor);
    std::cout << "After: (" << alps::hdf5::is_vectorizable(tensor) << ") tensor=" << tensor << std::endl;
}

int main(int argc, char** argv)
{
    alps::mpi::environment mpi_env(argc, argv);
    // test1(argc, argv);
    test2(argc, argv);
    // test3();

    return 0;
}
    
    

    
    
