#include <alps/hdf5.hpp>

#include <vector>

#include <boost/filesystem.hpp>

typedef enum { PLUS, MINUS } enum1_type;
inline alps::hdf5::oarchive & serialize(alps::hdf5::oarchive & ar, std::string const & p, enum1_type const & v) {
    switch (v) {
        case PLUS: ar << alps::make_pvp(p, std::string("plus"));
        case MINUS: ar << alps::make_pvp(p, std::string("minus"));
    }
    return ar;
}
inline alps::hdf5::iarchive & serialize(alps::hdf5::iarchive & ar, std::string const & p, enum1_type & v) {
    std::string s;
    ar >> alps::make_pvp(p, s);
    v = (s == "plus" ? PLUS : MINUS);
    return ar;
}
template<typename T> void test_enum(T & v, std::vector<T> w, T c[2]) {
    {
        alps::hdf5::oarchive oar("enum.h5");
        oar << alps::make_pvp("/enum/scalar", v);
        oar << alps::make_pvp("/enum/vector", w);
        oar << alps::make_pvp("/enum/c_arr", c, 2);
    }
    {
        alps::hdf5::iarchive iar("enum.h5");
        iar >> alps::make_pvp("/enum/scalar", v);
        std::cout << v << std::endl;
    }
}
int main() {
    {
        enum1_type v = PLUS;
        std::vector<enum1_type> w(2, MINUS);
        enum1_type c[2] = { PLUS, MINUS };
        test_enum(v, w, c);
    }
    boost::filesystem::remove(boost::filesystem::path("enum.h5"));
}