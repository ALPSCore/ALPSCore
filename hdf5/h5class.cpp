#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <boost/filesystem.hpp>

#include <alps/hdf5.hpp>

class userdefined_class {
    public:
        userdefined_class(): a(1), b(10, 3) {}
        void serialize(alps::hdf5::iarchive & ar) { 
            ar 
                >> alps::make_pvp("a", a) 
                >> alps::make_pvp("b", b)
            ; 
        }
        void serialize(alps::hdf5::oarchive & ar) const { 
            ar 
                << alps::make_pvp("a", a) 
                << alps::make_pvp("b", b)
            ; 
        }
        void dump() {
            std::cout << "a: " << a << " b: (" << b.size() << "): [";
            for (std::size_t i = 0; i < b.size();  ++i)
                std::cout << b[i] << ( i < b.size() - 1 ? ", " : "");
            std::cout << "]" << std::endl;
        }
    private:
        int a;
        std::vector<long> b;
};
int main() {
    {
        alps::hdf5::oarchive oar("class.h5");
        {
            userdefined_class value;
            oar << alps::make_pvp("/class/scalar", value);
        }
        {
            std::vector<userdefined_class> value(5);
            oar << alps::make_pvp("/class/vector", value);
        }
    }
    {
        alps::hdf5::iarchive iar("class.h5");
        {
            userdefined_class value;
            iar >> alps::make_pvp("/class/scalar", value);
            value.dump();
        }
        {
            std::vector<userdefined_class> value;
            iar >> alps::make_pvp("/class/vector", value);
            std::cout << "vector: " << value.size() << std::endl;
        }
    }
    boost::filesystem::remove(boost::filesystem::path("class.h5"));
}
