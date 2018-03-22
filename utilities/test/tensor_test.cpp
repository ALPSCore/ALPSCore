/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <gtest/gtest.h>
#include <complex>

#include "alps/numeric/tensors/tensor_base.hpp"

using namespace alps::numerics::detail;
using namespace alps::numerics;


TEST(TensorTest, StoragesComparison) {
  size_t N = 100;
  data_storage<double> s(N);
  data_view<double> v(s);
  // copy constructed
  data_view<double> v1(v);
  // duplicated
  data_view<double> v2(s);
  for(size_t i=0; i<N; ++i) {
    s.data(i) = i;
  }
  // self-comparison
  ASSERT_EQ(s, s);
  ASSERT_EQ(v, v);
  // left comparison;
  ASSERT_EQ(v, s);
  // right comparison;
  ASSERT_EQ(s, v);
  // comparison with views
  ASSERT_EQ(v1, v);
  ASSERT_EQ(v2, v);

}

TEST(TensorTest, TestInitialization) {
  tensor<std::complex<double>, 3> X(1,2,3);
  ASSERT_EQ(X.size(), 1*2*3u);
  tensor<double, 1> U( 50 );
  ASSERT_EQ(U.size(), 50u);
  std::vector<double> x(100);
  tensor<double, 2> UU(x.data(), 25, 4);
}

TEST(TensorTest, TestAssignments) {
  tensor<std::complex<double>, 3> X(std::array<size_t, 3>{{1,2,3}});
  X( 0,0,0 ) = 10.0;
  ASSERT_DOUBLE_EQ(X(0,0,0).real(), 10.0);
  ASSERT_DOUBLE_EQ(X(0,0,0).imag(), 0.0);

  tensor<double, 1> U(50);
  U(3) = 555.0;
  ASSERT_DOUBLE_EQ(U(3), 555.0);
  ASSERT_DOUBLE_EQ(U(0), 0.0);

  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXcd M2 = Eigen::MatrixXcd::Random(N, N);
  tensor<double, 2> T1({{N, N}});
  tensor<double, 2> T3(T1);
  tensor<std::complex<double>, 2> T2({{N, N}});
  for(size_t i = 0; i< N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T1(i,j) = M1(i,j);
    }
  }
  T2 = T1;
  ASSERT_EQ(T2, T1);
  tensor_view<std::complex<double>, 2> V = T2;
  ASSERT_EQ(V, T1);
}

TEST(TensorTest, TestCopyAssignments) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXcd M2 = Eigen::MatrixXcd::Random(N, N);
  tensor<double, 2> T1(N, N);
  tensor<double, 2> T3(T1);
  tensor<double, 2> T2(N, N);
  for(size_t i = 0; i< N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      T1(i,j) = M1(i,j);
    }
  }
  T2 = T1;
  ASSERT_EQ(T2, T1);
  tensor_view<double, 2> V1 = T1;
  tensor_view<double, 2> V2 = T2;
  ASSERT_EQ(V2, T1);
  V1(0,0) = -15.0;
  V2 = V1;
  ASSERT_EQ(T2(0,0), -15.0);
}

TEST(TensorTest, TestSlices) {
  tensor<std::complex<double>, 3> X(1, 2, 3);

  tensor_view<std::complex<double>, 1> slice1 = X(0, 1);
  ASSERT_EQ(slice1.dimension(), 1u);

  ASSERT_EQ(slice1.shape()[0], 3u);

  for (size_t i = 0; i<X.shape()[0]; ++i) {
    tensor_view<std::complex<double>, 2> slice2 = X(i);
    ASSERT_EQ(X.index(i), 6*i);
    ASSERT_EQ(slice2.dimension(), 2u);
    ASSERT_EQ(slice2.shape()[0], 2u);
    ASSERT_EQ(slice2.shape()[1], 3u);
    for (size_t j = 0; j < X.shape()[1]; ++j) {
      ASSERT_EQ(X.index(i, j), 6*i + j*3);
      for (size_t k = 0; k < X.shape()[2]; ++k) {
        X(i, j, k) = i*X.shape()[1]*X.shape()[2] + j*X.shape()[2] + k;
        ASSERT_DOUBLE_EQ(slice2(j, k).real(), X(i,j,k).real());
        ASSERT_DOUBLE_EQ(slice2(j, k).imag(), X(i,j,k).imag());
      }
    }
  }
}

TEST(TensorTest, TestSubSlices) {
  tensor<double, 4> X({{3,4,5,6}});
  for (size_t i = 0; i< X.shape()[0]; ++i) {
    tensor_view<double, 3> Y = X(i);
    for (size_t j = 0; j < X.shape()[1]; ++j) {
      tensor_view<double, 2> Z = Y (j);
      ASSERT_EQ(Z.storage().offset(), (i* X.shape()[1]+ j)* X.shape()[2]* X.shape()[3]);
      std::vector<double> XX(X.shape()[2]* X.shape()[3], 0.0);
      for (size_t k = 0; k < X.shape()[2]; ++k) {
        for (size_t l = 0; l < X.shape()[3]; ++l) {
          double value = i * X.shape()[1] * X.shape()[2] * X.shape()[3] + j * X.shape()[2] * X.shape()[3] + k * X.shape()[3] + l;
          X(i, j, k, l) = value;
          XX[k* X.shape()[3] + l] = value;
          ASSERT_DOUBLE_EQ(Y(j, k, l), X(i, j, k, l));
          ASSERT_DOUBLE_EQ(Z(k, l),    X(i, j, k, l));
        }
      }
      for (size_t k = 0; k < X.shape()[2]; ++k) {
        for (size_t l = 0; l < X.shape()[3]; ++l) {
          ASSERT_DOUBLE_EQ(XX[k* X.shape()[3] + l], X(i, j, k, l));
          ASSERT_DOUBLE_EQ(XX[k* X.shape()[3] + l], Z(k, l));
        }
      }
    }
  }
}

TEST(TensorTest, TestMultByScalar) {
  tensor<double, 2> XX({{3, 4}});
  for (size_t i = 0; i < XX.shape()[0]; ++i) {
    for (size_t j = 0; j < XX.shape()[1]; ++j) {
      double value = i * XX.shape()[0] + j;
      XX(i, j) = value;
    }
  }
  double mult = 12;
  tensor<double, 2> X = XX * mult;
  for (size_t i = 0; i < XX.shape()[0]; ++i) {
    for (size_t j = 0; j < XX.shape()[1]; ++j) {
      double value = i * XX.shape()[0] + j;
      ASSERT_DOUBLE_EQ(value*mult, X(i, j));
      ASSERT_DOUBLE_EQ(value, XX(i, j));
    }
  }
  XX *= 10;
  for (size_t i = 0; i < XX.shape()[0]; ++i) {
    for (size_t j = 0; j < XX.shape()[1]; ++j) {
      double value = i * XX.shape()[0] + j;
      ASSERT_DOUBLE_EQ(value*10, XX(i, j));
    }
  }
  auto Z = XX * X;
  for (size_t i = 0; i < XX.shape()[0]; ++i) {
    for (size_t j = 0; j < XX.shape()[1]; ++j) {
      ASSERT_DOUBLE_EQ(Z(i, j), X(i, j) * XX(i, j));
    }
  }
}

TEST(TensorTest, RemoteDataRef) {
  size_t n = 256;
  std::vector<double> X(n, 0.0);
  tensor_view<double, 1> Y(X.data(), {{X.size()}});
  tensor_view<double, 2> Z(X.data(), {{16, 16}});
  for(size_t i = 0; i< X.size(); ++i) {
    X[i] = i*0.5;
  }
  for(size_t i = 0; i< Y.shape()[0]; ++i) {
    ASSERT_DOUBLE_EQ(X[i], Y(i));
  }
  for(size_t i = 0; i< Z.shape()[0]; ++i) {
    tensor_view<double, 1> W = Z(i);
    for (size_t j = 0; j < Z.shape()[1]; ++j) {
      W(j) += 10;
      ASSERT_DOUBLE_EQ(X[i* Z.shape()[1] + j], Z(i, j));
      ASSERT_DOUBLE_EQ(X[i* Z.shape()[1] + j], W(j));
    }
  }
}

TEST(TensorTest, Inversion) {
  size_t n = 40;
  Eigen::MatrixXd M(n,n);
  M = Eigen::MatrixXd::Identity(n, n);

  M(0, n - 1) = 1.0;
  M(n - 1, 0) = -1.0;
  tensor<double, 2> X({{n, n}});
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      X(i, j) = M(i, j);
    }
  }
  M = M.inverse();
  X = X.inverse();
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(X(i, j), M(i, j));
    }
  }
  tensor<double, 2> Z = X.dot(X.inverse());
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(Z(i, j), i==j? 1.0 : 0.0);
    }
  }
  tensor<double, 2> Y = X.inverse();
  ASSERT_DOUBLE_EQ(Y(n-1, 0), -1.0);
  ASSERT_DOUBLE_EQ(Y(0, n-1), 1.0);
}

TEST(TensorTest, SliceInversion) {
  size_t N = 40;
  size_t K = 10;
  tensor<double, 3> X({{K, N, N}});

  for(size_t k = 0; k < K; ++k) {
    Eigen::MatrixXd M(N,N);
    M = Eigen::MatrixXd::Identity(N, N);
    M *= (k+1);
    M(0, N - 1) = 1.0;
    M(N - 1, 0) = -1.0;
    tensor<double, 2> Z = X(k);
    for(size_t i = 0; i< N; ++i){
      for (size_t j = 0; j < N; ++j) {
        Z(i, j) = M(i, j);
      }
    }
    M = M.inverse();
    Z = Z.inverse();
    for(size_t i = 0; i< N; ++i){
      for (size_t j = 0; j < N; ++j) {
        ASSERT_DOUBLE_EQ(Z(i, j), M(i, j));
      }
    }
  }
}

TEST(TensorTest, BasicArithmetics) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(N, N);
  tensor<double, 2> X({{N, N}});
  tensor<double, 2> Z({{N, N}});
  for(size_t i = 0; i< N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
      Z(i,j) = M2(i,j);
    }
  }
  auto M3 = M1 + M2;
  auto Y = X + Z;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j), M3(i, j));
    }
  }
  Y -= Z;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_NEAR(Y(i, j), X(i, j), 1e-10);
    }
  }
}

TEST(TensorTest, BasicArithmeticsView) {
  size_t N = 10;
  tensor<double, 3> W({{N,N,N}});
  W *= 0.0;
  for (size_t k = 0; k < N; ++k) {
    Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(N, N);
    tensor_view<double, 2> X = W(k);
    tensor<double, 2> Z({{N, N}});
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        X(i, j) = M1(i, j);
        Z(i, j) = M2(i, j);
      }
    }
    auto M3 = M1 + M2;
    tensor<double, 2> Y = X + Z;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_DOUBLE_EQ(Y(i, j), M3(i, j));
        ASSERT_DOUBLE_EQ(X(i, j), M1(i, j));
        ASSERT_DOUBLE_EQ(Z(i, j), M2(i, j));
      }
    }
    Y -= Z;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_NEAR(Y(i, j), X(i, j), 1e-10);
      }
    }
  }
}

TEST(TensorTest, DoubleScaleByComplex) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  std::complex<double> x = 5.0;
  tensor<double, 2> X({{N, N}});
  for(size_t i = 0; i< N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
    }
  }
  auto M2 = M1 * x;
  auto Y = X * x;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j).real(), M2(i, j).real());
      ASSERT_DOUBLE_EQ(Y(i, j).imag(), M2(i, j).imag());
    }
  }
  auto M3 = M1 / x;
  Y = X / x;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j).real(), M3(i, j).real());
      ASSERT_DOUBLE_EQ(Y(i, j).imag(), M3(i, j).imag());
    }
  }
}

TEST(TensorTest, DoubleScaleByComplexView) {
  size_t N = 10;
  size_t M = 10;
  std::complex<double> x = 5.0;
  tensor<double, 3> Z({{M, N, N}});
  for(size_t k = 0; k< M; ++k) {
    Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
    auto X =  Z(k);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        X(i, j) = M1(i, j);
      }
    }
    auto M3 = M1 * x;
    auto Y = X * x;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        ASSERT_DOUBLE_EQ(Y(i, j).real(), M3(i, j).real());
        ASSERT_DOUBLE_EQ(Y(i, j).imag(), M3(i, j).imag());
      }
    }
  }
}

TEST(TensorTest, DoublePlusComplex) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXcd M2 = Eigen::MatrixXcd::Random(N, N);
  tensor<double, 2> X({{N, N}});
  tensor<std::complex<double>, 2> Z({{N, N}});
  for(size_t i = 0; i< N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
      Z(i,j) = M2(i,j);
    }
  }
  auto M3 = M1 + M2;
  auto Y = Z + X;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_NEAR(Y(i, j).real(), M3(i, j).real(), 1e-12);
      ASSERT_NEAR(Y(i, j).imag(), M3(i, j).imag(), 1e-12);
    }
  }
  Y -= Z;
  for(size_t i = 0; i< N; ++i){
    for (size_t j = 0; j < N; ++j) {
      ASSERT_NEAR(Y(i, j).real(), X(i, j),1e-12);
    }
  }
}

TEST(TensorTest, DotProduct) {
  size_t n = 40;
  size_t l = 20;
  // same size
  Eigen::MatrixXd M(n,n);
  M = Eigen::MatrixXd::Random(n, n);
  tensor<double, 2> X1({{n, n}});
  tensor<double, 2> X2({{n, n}});
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      X1(i, j) = M(i, j);
      X2(i, j) = M(j, i);
    }
  }
  Eigen::MatrixXd M2 = M*(M.transpose());
  auto X3 = X1.dot(X2);
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(X3(i, j), M2(i, j));
    }
  }
  Eigen::MatrixXd N1(n,l);
  Eigen::MatrixXd N2(l,n);
  N1 = Eigen::MatrixXd::Random(n, l);
  N2 = Eigen::MatrixXd::Random(l, n);
  tensor<double, 2> Y1({{n, l}});
  tensor<double, 2> Y2({{l, n}});
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < l; ++j) {
      Y1(i, j) = N1(i, j);
      Y2(j, i) = N2(j, i);
    }
  }
  ASSERT_ANY_THROW(Y1.dot(X1));
  Eigen::MatrixXd N3 = N1*N2;
  auto Y3 = Y1.dot(Y2);
  ASSERT_EQ(Y3.shape()[0], n);
  ASSERT_EQ(Y3.shape()[1], n);
  for(size_t i = 0; i< n; ++i){
    for (size_t j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(Y3(i, j), N3(i, j));
    }
  }
}

TEST(TensorTest, StorageAssignments) {
  size_t N = 10;
  data_storage<double> storage_obj(N);
  std::vector<double> buffer_obj(N, 0.0);
  for(size_t i =0 ;i<buffer_obj.size(); ++i) {
    buffer_obj[i] = i*15.0;
  }
//  data_view<double> view_obj(buffer_obj.data(), buffer_obj.size());
  data_storage<std::complex<double> > storage_obj2 = data_view<double>(buffer_obj.data(), buffer_obj.size());
  ASSERT_EQ(storage_obj2.size(), buffer_obj.size());
  for(size_t i =0 ;i<buffer_obj.size(); ++i) {
    ASSERT_DOUBLE_EQ(buffer_obj[i], storage_obj2.data(i).real());
  }
  ASSERT_NO_THROW(storage_obj2 = storage_obj);

}

TEST(TensorTest, ConstTensor) {
  size_t N = 10;
  tensor <double, 3> X(N, N, N);
  const tensor <double, 3> & Y = X;
  Y( 1 ) ( 1 );
}

TEST(TensorTest, Reshape) {
  size_t N = 10;
  tensor <double, 3> X(N, N, N);
  std::array<size_t, 3> shape{1,100,5};
  X.reshape(shape);
  ASSERT_TRUE(X.shape()[0] == 1 && X.shape()[1] == 100 && X.shape()[2] == 5);
  X.reshape(10,10,10);
  ASSERT_TRUE(X.shape()[0] == 10 && X.shape()[1] == 10 && X.shape()[2] == 10);
}
