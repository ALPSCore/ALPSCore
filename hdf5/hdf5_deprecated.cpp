/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
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
*****************************************************************************/

#include <alps/ngs/hdf5/deprecated.hpp>

#define SEED 42
#define VECTOR_SIZE 25

#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

double rng() {
    static boost::mt19937 rng(SEED);
    static boost::uniform_real<> dist_real(0., 1e9);
    static boost::variate_generator<boost::mt19937, boost::uniform_real<> > random_real(rng, dist_real);
    return random_real();
}

typedef enum { PLUS, MINUS } enum_type;

template<typename T> void initialize(T & v) {
    v = static_cast<T>(rng());
}
template<typename T> void initialize(std::complex<T> & v) {
    v = std::complex<T>(rng(), rng());
}
void initialize(std::string & v) {
    v = boost::lexical_cast<std::string>(rng());
}
void initialize(enum_type & v) {
    v = static_cast<std::size_t>(rng()) % 2 == 0 ? PLUS : MINUS;
}

template<typename T> class userdefined_class {
    public:
        userdefined_class(): b(VECTOR_SIZE) {
            initialize(a);
            for (std::size_t i = 0; i < VECTOR_SIZE; ++i)
                initialize(b[i]);
            initialize(c);
        }
        void serialize(alps::hdf5::oarchive & ar) const {
            ar
                << alps::make_pvp("a", a)
                << alps::make_pvp("b", b)
                << alps::make_pvp("c", c)
            ;
        }
        void serialize(alps::hdf5::iarchive & ar) { 
            ar
                >> alps::make_pvp("a", a)
                >> alps::make_pvp("b", b)
                >> alps::make_pvp("c", c)
            ;
        }
        bool operator==(userdefined_class<T> const & v) const {
            return a == v.a && b.size() == v.b.size() && (b.size() == 0 || std::equal(b.begin(), b.end(), v.b.begin())) && c == v.c;
        }
    private:
        T a;
        std::vector<T> b;
        enum_type c;
};

template<typename T> void initialize(userdefined_class<T> & v) {
    v = userdefined_class<T>();
}

void save(
      alps::hdf5::oarchive & ar
    , std::string const & path
    , enum_type const & value
    , std::vector<std::size_t> size = std::vector<std::size_t>()
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
    switch (value) {
        case PLUS: ar << alps::make_pvp(path, std::string("plus")); break;
        case MINUS: ar << alps::make_pvp(path, std::string("minus")); break;
    }
}
void load(
      alps::hdf5::iarchive & ar
    , std::string const & path
    , enum_type & value
    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
    , std::vector<std::size_t> offset = std::vector<std::size_t>()
) {
    std::string s;
    ar >> alps::make_pvp(path, s);
    value = (s == "plus" ? PLUS : MINUS);
}

int main() {
    std::string const filename = "test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    userdefined_class<double> read, write;
    {
        alps::hdf5::oarchive oar(filename);
        oar
            << alps::make_pvp("/data", write)
        ;
    }
    {
        alps::hdf5::iarchive iar(filename);
        iar
            >> alps::make_pvp("/data", read)
        ;
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
    std::cout << (write == read ? "SUCCESS" : "FAILURE") << std::endl;
    return write == read ? EXIT_SUCCESS : EXIT_FAILURE;
}
