// atm vli only works with GCC
#if defined(__GNUG__) && !defined(__ICC) && !defined(__FCC_VERSION)

#include <omp.h>
#include <alps/graph/vli.hpp>

// *****************************************************************************
//  Timer
// *****************************************************************************

class TimerOMP
{
public:
	TimerOMP(std::string name_) : name(name_), val(0.0), timer_start(0.0), timer_end(0.0){}

    ~TimerOMP() { std::cout << name << " " << val << std::endl; }
	
	void begin()
	{
		timer_start = omp_get_wtime(); 
	}
	
	void end()
	{
		timer_end = omp_get_wtime();
		val += timer_end - timer_start;
	}
	
private:
    std::string name;
    double val;
	double timer_start, timer_end;
};

// *****************************************************************************
//  Minimal polynomial class
// *****************************************************************************

#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace hp2c
{

template <typename T>
inline void multiply_add(T& c, T const& a, T const& b)
{
    c += a*b;
}

/**
  * A monomial class
  */

template <typename CoeffType>
struct monomial
{
    std::size_t j_exp;
    std::size_t h_exp;
    CoeffType coeff;

    /**
      * Constructor: Creates a monomial 1*J^j_exp*h^h_exp
      */
    explicit monomial(unsigned int j_exp = 0, unsigned int h_exp = 0)
        : j_exp(j_exp), h_exp(h_exp), coeff(1)
    {
    }

    monomial& operator *= (CoeffType const& c)
    {
        coeff *= c;
        return *this;
    }

    monomial& operator *= (int c)
    {
        coeff *= c;
        return *this;
    }
};

/**
  * Multiplication with some factor of arbitrary type T
  * (for which the monomial class has to provide a *= operator)
  * e.g. T = int
  */
template <typename CoeffType, typename T>
monomial<CoeffType> operator * (monomial<CoeffType> m, T const& t)
{
    m *= t;
    return m;
}

template <typename CoeffType, typename T>
monomial<CoeffType> operator * (T const& t, monomial<CoeffType> const& m)
{
    return m*t;
}

//
// *****************************************************************************
//

/**
  * A polynomial class
  */

template <typename CoeffType, unsigned int Order>
class polynomial;

/**
  * Multiplication of two polynomials
  */
template <typename CoeffType, unsigned int Order>
polynomial<CoeffType,2*Order> operator * (polynomial<CoeffType,Order> const& p1, polynomial<CoeffType,Order> const& p2)
{
    std::vector<CoeffType> result_coeffs(4*Order*Order);
    for(std::size_t je1 = 0; je1 < Order; ++je1){
        for(std::size_t je2 = 0; je2 < Order; ++je2){
            for(std::size_t he1 = 0; he1 < Order; ++he1){
				std::size_t I1 = je1*Order+he1;
               for(std::size_t he2 = 0; he2 < Order; ++he2){
					std::size_t Ir = (je1+je2)*2*Order + he1+he2;
					std::size_t I2 = je2*Order+he2;
//                     result_coeffs[ (je1+je2)*2*Order + he1+he2 ] += p1.coeffs[je1*max_order+he1] * p2.coeffs[je2*max_order+he2];
                     multiply_add(result_coeffs[Ir], p1.coeffs[I1], p2.coeffs[I2]);
               }
            }

        }
    }
    return polynomial<CoeffType,2*Order>(result_coeffs);
}

/**
  * Multiplication of a polynomial with a monomial
  */
template <typename CoeffType, unsigned int Order, typename T>
polynomial<CoeffType,Order> operator * (polynomial<CoeffType,Order> const& p, monomial<T> const& m)
{
    polynomial<CoeffType,Order> r;
    for(std::size_t je = 0; je < Order-m.j_exp; ++je)
        for(std::size_t he = 0; he < Order-m.h_exp; ++he)
            r(je+m.j_exp,he+m.h_exp) = p(je,he)  * m.coeff;
    return r;
}

template <typename CoeffType, unsigned int Order>
class polynomial
{
    private:
        std::vector<CoeffType> coeffs;
    public:
        enum {max_order = Order};
        typedef unsigned int exponent_type;

        friend polynomial<CoeffType,2*Order> operator *<> (polynomial<CoeffType,Order> const&, polynomial<CoeffType,Order> const&);

        /**
          * Constructor: Creates a polynomial
          * p(J,h) = c_0 + c_{1,0} * J^1 * h^0 + c_{2,0} * J^2 * h^0 + ...
          *             + c_{max_order-1,max_order-1} * (J*h)^(max_order-1)
          * where all c_* are of type CoeffType and set to 0.
          */
        polynomial()
            : coeffs(max_order*max_order,CoeffType(0))
        {
        }

        /**
          * Copy constructor
          */
        polynomial(polynomial const& p)
            :coeffs(p.coeffs)
        {
        }

        explicit polynomial(std::vector<CoeffType> const& v)
            : coeffs(v)
        {
            assert(v.size() == max_order*max_order);
        }

        /**
          * Assignment operator
          */
        polynomial& operator = (polynomial p)
        {
            swap(*this,p);
            return *this;
        }

        /**
          * Swap function
          */
        friend void swap(polynomial& p1, polynomial& p2)
        {
            swap(p1.coeffs,p2.coeffs);
        }

        /**
          * Prints polynomial to ostream o
          */
        void print(std::ostream& o) const
        {
            for(std::size_t je = 0; je < max_order; ++je)
            {
                for(std::size_t he = 0; he < max_order; ++he)
                {
                    if( coeffs[je*max_order+he] != CoeffType(0))
                    {
                        if(coeffs[je*max_order+he] > CoeffType(0))
                            o<<"+";
                        o<<coeffs[je*max_order+he]<<"*J^"<<je<<"*h^"<<he;
                    }
                }
            }
        }

        /**
          * Plus assign with a polynomial
          */
        polynomial& operator += (polynomial const& p)
        {
            typename std::vector<CoeffType>::iterator it    = coeffs.begin();
            typename std::vector<CoeffType>::iterator end   = coeffs.end();
            typename std::vector<CoeffType>::const_iterator p_it  = p.coeffs.begin();
            typename std::vector<CoeffType>::const_iterator p_end = p.coeffs.end();
            while( it != end && p_it != p_end)
            {
                *it += *p_it;
                ++it;
                ++p_it;
            }
            return *this;
        }

        /**
          * Plus assign with a coefficient
          * Adds the coefficient to the 0th order element of the polynomial
          */
        polynomial& operator += (CoeffType c)
        {
            coeffs[0] += c;
            return *this;
        }

        /**
          * Plus assign with a monomial
          */
        template <typename T>
        polynomial& operator += (monomial<T> const& m)
        {
            assert(m.j_exp < max_order);
            assert(m.h_exp < max_order);
            coeffs[m.j_exp*max_order+m.h_exp] += m.coeff;
            return *this;
        }

//        polynomial& operator *= (monomial<CoeffType> const& m)
//        {
//            polynomial p;
//            for(std::size_t je = 0; je < max_order; ++je)
//                for(std::size_t he = 0; he < max_order; ++he)
//                    p.coeffs[ (je+m.j_exp)*max_order + he+m.h_exp ] = m.coeff * this->coeffs[ je*max_order + he ];
//            swap(p);
//            return *this;
//        }

        /**
          * Multiplies assign with coefficient
          * Mutliplies all elements the argument
          */
        template <typename T>
        polynomial& operator *= (T const& c)
        {
            for(typename std::vector<CoeffType>::iterator it = coeffs.begin(); it != coeffs.end(); ++it)
                *it *= c;
            return *this;
        }

        void negate()
        {
            for(typename std::vector<CoeffType>::iterator it = coeffs.begin(); it != coeffs.end(); ++it)
                *it = -*it;
        }
        polynomial operator - () const
        {
            polynomial tmp(*this);
            tmp.negate();
            return tmp;
        }

        /**
         * Comparison with an CoeffType (0th order)
         */

        bool operator == (CoeffType const& c) const
        {
            if(coeffs[0] == c)
            {
                bool all_zero = true;
                for(typename std::vector<CoeffType>::const_iterator it = coeffs.begin(); it != coeffs.end(); ++it)
                    all_zero = all_zero && (*it == 0);
                return all_zero;
            }
            return false;
        }

        /**
         * Access coefficient of monomial J^j_exp*h^h_exp
         */
        inline CoeffType const& operator ()(unsigned int j_exp, unsigned int h_exp) const
        {
            assert(j_exp < max_order);
            assert(h_exp < max_order);
            return coeffs[j_exp*max_order+h_exp];
        }
        
        /**
         * Access coefficient of monomial J^j_exp*h^h_exp
         */
        inline CoeffType& operator ()(unsigned int j_exp, unsigned int h_exp)
        {
            assert(j_exp < max_order);
            assert(h_exp < max_order);
            return coeffs[j_exp*max_order+h_exp];
        }
};

/**
  * Stream operator
  */
template <typename CoeffType, unsigned int Order>
std::ostream& operator <<(std::ostream& o, polynomial<CoeffType,Order> const& p)
{
    p.print(o);
    return o;
}

/**
  * Multiplication of a monomial with a polynomial
  */
template <typename CoeffType, unsigned int Order, typename T>
polynomial<CoeffType,Order> operator * (monomial<T> const& m,polynomial<CoeffType,Order> const& p)
{
    return p * m;
}

/**
  * Multiplication of a polynomial with a factor
  */
template <typename CoeffType, unsigned int Order, typename T>
polynomial<CoeffType,Order> operator * (polynomial<CoeffType,Order> p, T const& c)
{
    p *= c;
    return p;
}

template <typename CoeffType, unsigned int Order, typename T>
polynomial<CoeffType,Order> operator * (T const& c, polynomial<CoeffType,Order> const& p)
{
    return p * c;
}

template <typename CoeffType, unsigned int Order>
polynomial<CoeffType,2*Order>  inner_product(std::vector<polynomial<CoeffType,Order> > const& a, std::vector<polynomial<CoeffType,Order> >const& b)
{
    assert( a.size() == b.size() );


#ifdef _OPENMP
    std::vector < polynomial<CoeffType,2*Order> > result(omp_get_max_threads());
    #pragma omp parallel for
    for(int i=0; i < static_cast<int>(a.size());++i){
        result[omp_get_thread_num()] += a[i]*b[i];
//        multiply_add(result[omp_get_thread_num()], a[i], b[i]);
    }
    
    for(int i=1; i < omp_get_max_threads();++i){
        result[0] += result[i];
    }
    
    return result[0];
#else
    polynomial<CoeffType,2*Order> result;
    typename std::vector<polynomial<CoeffType,Order> >::const_iterator it(a.begin()),  it_b(b.begin());
    while( it != a.end() )
    {
        result += *it * *it_b;
//        multiply_add(result,*it, *it_b);
        ++it;
        ++it_b;
    }

    return result;
#endif
}

}

// *****************************************************************************
// util
// *****************************************************************************
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

boost::mt11213b rng;

template <typename Vli>
struct max_int_value
{
};

template <typename Vli>
typename Vli::value_type rnd_digit()
{
    static boost::uniform_int<typename Vli::value_type> rnd(0,max_int_value<Vli>::value);
    return rnd(rng);
}

template <typename Vli>
void fill_random(Vli& v, std::size_t size)
{
    assert(size <= Vli::static_size);
    for(std::size_t i=0; i < size; ++i)
        v[i] = rnd_digit<Vli>();
}

template <typename Polynomial>
void fill_poly_random(Polynomial& p, std::size_t size)
{
    for(typename Polynomial::exponent_type i=0; i < Polynomial::max_order; ++i)
        for(typename Polynomial::exponent_type j=0; j < Polynomial::max_order; ++j)
            fill_random(p(i,j),size);
}

template <typename Vector>
void fill_vector_random(Vector& v, std::size_t size)
{
    for(typename Vector::size_type i=0; i < v.size(); ++i)
        fill_poly_random(v[i], size);
}


// *****************************************************************************
// Benchmark
// *****************************************************************************

// What is the maximal value that we can write in a digit (segment) of the vli
// TODO please check!
template <>
struct max_int_value<alps::graph::vli<256> >
{
    enum {value = 1ul<<63};
};

namespace hp2c
{
template <>
inline void multiply_add(alps::graph::vli<256>& c, alps::graph::vli<256> const& a, alps::graph::vli<256> const& b)
{
    c.madd(a,b);
}
}

int main()
{
	omp_set_num_threads(1);

    static const unsigned int Order = 21;
    typedef alps::graph::vli<256> vli_type;

    typedef hp2c::polynomial<vli_type,Order> polynomial_type;
    typedef hp2c::polynomial<vli_type,2*Order> polynomial_result_type;

    std::vector<polynomial_type> v1(16384);
    std::vector<polynomial_type> v2(16384);


    fill_vector_random(v1,1);
    fill_vector_random(v2,1);

    TimerOMP t1("VLI Lukas");
    t1.begin();
    polynomial_result_type p = hp2c::inner_product(v1,v2);
    t1.end();
    std::cout << p << std::endl;
}

#endif
