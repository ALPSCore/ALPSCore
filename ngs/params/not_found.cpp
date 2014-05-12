#include <alps/ngs.hpp>

int main(int argc, char** argv){
    alps::params parms;
    parms["hello"]="world";

    try {
        std::cout<<parms["hello"]<<std::endl;
        std::cout<<parms["not_in_parms"]<<std::endl;
    } catch (std::exception const & e) {
        std::string w = e.what();
        std::cout << w.substr(0, w.find_first_of('\n')) << std::endl;
    }
    
    const alps::params p(parms);

    try {
        std::cout<<p["not_in_parms"]<<std::endl;
    } catch (std::exception const & e) {
        std::string w = e.what();
        std::cout << w.substr(0, w.find_first_of('\n')) << std::endl;
    }

    return 0;
}
