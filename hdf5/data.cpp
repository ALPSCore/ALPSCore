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

#include "creator.hpp"

#ifndef SZIP_COMPRESS
    #define SZIP_COMPRESS false
#endif

template<typename base_type> struct test {
    static bool run(std::string const & filename, boost::mpl::true_) {
        std::vector<std::size_t> size_0;
        base_type* write_0_value = NULL;
        std::size_t length = MATRIX_SIZE;
        std::vector<std::size_t> size_1(1, MATRIX_SIZE);
        base_type write_1_value[MATRIX_SIZE];
        std::vector<std::size_t> size_2(2, MATRIX_SIZE);
        base_type write_2_value[MATRIX_SIZE][MATRIX_SIZE];
        std::vector<std::size_t> size_3(3, MATRIX_SIZE);
        base_type write_3_value[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE];
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i) {
            initialize(write_1_value[i]);
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j) {
                initialize(write_2_value[i][j]);
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)
                    initialize(write_3_value[i][j][k]);
            }
        }
        {
            alps::hdf5::oarchive oar(filename, SZIP_COMPRESS);
            oar
                << alps::make_pvp("/len", &write_1_value[0], length)
                << alps::make_pvp("/ptr_0", write_0_value, size_0)
                << alps::make_pvp("/ptr_1", &write_1_value[0], size_1)
                << alps::make_pvp("/ptr_2", &write_2_value[0][0], size_2)
                << alps::make_pvp("/ptr_3", &write_3_value[0][0][0], size_3)
            ;
        }
        {
            base_type* read_0_value = NULL;
            base_type read_1_len_value[MATRIX_SIZE], read_1_value[MATRIX_SIZE];
            base_type read_2_value[MATRIX_SIZE][MATRIX_SIZE];
            base_type read_3_value[MATRIX_SIZE][MATRIX_SIZE][MATRIX_SIZE];
            alps::hdf5::iarchive iar(filename);
            iar
                >> alps::make_pvp("/len", &read_1_len_value[0], length)
                >> alps::make_pvp("/ptr_0", read_0_value, size_0)
                >> alps::make_pvp("/ptr_1", &read_1_value[0], size_1)
                >> alps::make_pvp("/ptr_2", &read_2_value[0][0], size_2)
                >> alps::make_pvp("/ptr_3", &read_3_value[0][0][0], size_3)
            ;
            return write_0_value == read_0_value
                && equal(&write_1_value[0], &read_1_len_value[0], length)
                && equal(&write_1_value[0], &read_1_value[0], size_1[0])
                && equal(&write_2_value[0][0], &read_2_value[0][0], size_2[0] * size_2[1])
                && equal(&write_3_value[0][0][0], &read_3_value[0][0][0], size_3[0] * size_3[1] * size_3[2])
            ;
        }
    }
    static bool run(std::string const & filename, boost::false_type) {
        base_type random_write(creator<base_type>::random());
        base_type empty_write(creator<base_type>::empty());
        base_type special_write(creator<base_type>::special());
        bool result;
        {
            alps::hdf5::oarchive oar(filename, SZIP_COMPRESS);
            oar
                << alps::make_pvp("/random", random_write)
                << alps::make_pvp("/empty", empty_write)
                << alps::make_pvp("/special", special_write)
            ;
        }
        {
            alps::hdf5::iarchive iar(filename);
            base_type random_read(creator<base_type>::random(iar));
            base_type empty_read(creator<base_type>::empty(iar));
            base_type special_read(creator<base_type>::special(iar));
            iar
                >> alps::make_pvp("/random", random_read)
                >> alps::make_pvp("/empty", empty_read)
                >> alps::make_pvp("/special", special_read)
            ;
            result = equal(random_write, random_read) && equal(empty_write, empty_read) && equal(special_write, special_read);
            destructor<base_type>::apply(random_read);
            destructor<base_type>::apply(empty_read);
            destructor<base_type>::apply(special_read);
        }
        destructor<base_type>::apply(random_write);
        destructor<base_type>::apply(empty_write);
        destructor<base_type>::apply(special_write);
        return result;
    }
};

template<typename T> struct test<boost::shared_array<T> > {
    static bool run(std::string const & filename, boost::mpl::false_) {
        std::size_t length = MATRIX_SIZE;
        std::vector<std::size_t> size_1(1, MATRIX_SIZE);
        boost::shared_array<T> write_1_value(new T[MATRIX_SIZE]);
        std::vector<std::size_t> size_2(2, MATRIX_SIZE);
        boost::shared_array<T> write_2_value(new T[MATRIX_SIZE * MATRIX_SIZE]);
        std::vector<std::size_t> size_3(3, MATRIX_SIZE);
        boost::shared_array<T> write_3_value(new T[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE]);
        for (std::size_t i = 0; i < MATRIX_SIZE; ++i) {
            initialize(write_1_value[i]);
            for (std::size_t j = 0; j < MATRIX_SIZE; ++j) {
                initialize(write_2_value[i * MATRIX_SIZE + j]);
                for (std::size_t k = 0; k < MATRIX_SIZE; ++k)
                    initialize(write_3_value[(i * MATRIX_SIZE + j) * MATRIX_SIZE + k]);
            }
        }
        {
            alps::hdf5::oarchive oar(filename, SZIP_COMPRESS);
            oar
                << alps::make_pvp("/len", write_1_value, length)
                << alps::make_pvp("/ptr_1", write_1_value, size_1)
                << alps::make_pvp("/ptr_2", write_2_value, size_2)
                << alps::make_pvp("/ptr_3", write_3_value, size_3)
            ;
        }
        {
            boost::shared_array<T> read_1_len_value(new T[MATRIX_SIZE]), read_1_value(new T[MATRIX_SIZE]);
            boost::shared_array<T> read_2_value(new T[MATRIX_SIZE * MATRIX_SIZE]);
            boost::shared_array<T> read_3_value(new T[MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE]);
            alps::hdf5::iarchive iar(filename);
            iar
                >> alps::make_pvp("/len", read_1_len_value, length)
                >> alps::make_pvp("/ptr_1", read_1_value, size_1)
                >> alps::make_pvp("/ptr_2", read_2_value, size_2)
                >> alps::make_pvp("/ptr_3", read_3_value, size_3)
            ;
            return equal(write_1_value.get(), read_1_len_value.get(), length)
                && equal(write_1_value.get(), read_1_value.get(), size_1[0])
                && equal(write_2_value.get(), read_2_value.get(), size_2[0] * size_2[1])
                && equal(write_3_value.get(), read_3_value.get(), size_3[0] * size_3[1] * size_3[2])
            ;
        }
    }
};

int main() {
    std::string const filename = "test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    bool result = true;
    for (std::size_t i = 32; i && result; --i)
        result = test<boost::remove_pointer< TYPE >::type >::run(filename, boost::is_pointer< TYPE >::type());
    boost::filesystem::remove(boost::filesystem::path(filename));
    std::cout << (result ? "SUCCESS" : "FAILURE") << std::endl;
    return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
