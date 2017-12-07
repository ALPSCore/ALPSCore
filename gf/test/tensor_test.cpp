//
// Created by iskakoff on 26/10/17.
//

#include <gtest/gtest.h>
#include <complex>

#include "alps/gf_new/tensors/TensorBase.h"

using namespace alps::gf::detail;


TEST(TensorTest, TestInitialization) {
  Tensor<std::complex<double>, 3> X(std::array<size_t, 3>{{1,2,3}});
  ASSERT_EQ(X.size(), 1*2*3);
  Tensor<double, 1> U( 50 );
  ASSERT_EQ(U.size(), 50);
}

TEST(TensorTest, TestAssignments) {
  Tensor<std::complex<double>, 3> X(std::array<size_t, 3>{{1,2,3}});
  X( 0,0,0 ) = 10.0;
  ASSERT_DOUBLE_EQ(X(0,0,0).real(), 10.0);
  ASSERT_DOUBLE_EQ(X(0,0,0).imag(), 0.0);

  Tensor<double, 1> U(50);
  U(3) = 555.0;
  ASSERT_DOUBLE_EQ(U(3), 555.0);
  ASSERT_DOUBLE_EQ(U(0), 0.0);
}


TEST(TensorTest, TestSlices) {
  Tensor<std::complex<double>, 3> X(std::array<size_t, 3>{{1,2,3}});

  TensorView<std::complex<double>, 1> slice1 = X(0, 1);
  ASSERT_EQ(slice1.dimension(), 1);

  ASSERT_EQ(slice1.sizes()[0], 3);

  for (int i = 0; i<X.sizes()[0]; ++i) {
    TensorView<std::complex<double>, 2> slice2 = X(i);
    ASSERT_EQ(X.index(std::integral_constant<int, 0>(), i), 6*i);
    ASSERT_EQ(slice2.dimension(), 2);
    ASSERT_EQ(slice2.sizes()[0], 2);
    ASSERT_EQ(slice2.sizes()[1], 3);
    for (int j = 0; j < X.sizes()[1]; ++j) {
      ASSERT_EQ(X.index(std::integral_constant<int, 0>(), i, j), 6*i + j*3);
      for (int k = 0; k < X.sizes()[2]; ++k) {
        X(i, j, k) = i*X.sizes()[1]*X.sizes()[2] + j*X.sizes()[2] + k;
        ASSERT_DOUBLE_EQ(slice2(j, k).real(), X(i,j,k).real());
        ASSERT_DOUBLE_EQ(slice2(j, k).imag(), X(i,j,k).imag());
      }
    }
  }
}


TEST(TensorTest, TestSubSlices) {
  Tensor<double, 4> X({{3,4,5,6}});
  for (int i = 0; i<X.sizes()[0]; ++i) {
    TensorView<double, 3> Y = X(i);
    for (int j = 0; j < X.sizes()[1]; ++j) {
      TensorView<double, 2> Z = Y (j);
      ASSERT_EQ(Z.data().offset(), (i*X.sizes()[1]+ j)*X.sizes()[2]*X.sizes()[3]);
      std::vector<double> XX(X.sizes()[2]*X.sizes()[3], 0.0);
      for (int k = 0; k < X.sizes()[2]; ++k) {
        for (int l = 0; l < X.sizes()[3]; ++l) {
          double value = i * X.sizes()[1] * X.sizes()[2] * X.sizes()[3] + j * X.sizes()[2] * X.sizes()[3] + k * X.sizes()[3] + l;
          X(i, j, k, l) = value;
          XX[k*X.sizes()[3] + l] = value;
          ASSERT_DOUBLE_EQ(Y(j, k, l), X(i, j, k, l));
          ASSERT_DOUBLE_EQ(Z(k, l),    X(i, j, k, l));
        }
      }
      for (int k = 0; k < X.sizes()[2]; ++k) {
        for (int l = 0; l < X.sizes()[3]; ++l) {
          ASSERT_DOUBLE_EQ(XX[k*X.sizes()[3] + l], X(i, j, k, l));
          ASSERT_DOUBLE_EQ(XX[k*X.sizes()[3] + l], Z(k, l));
        }
      }
    }
  }
}

TEST(TensorTest, TestMultByScalar) {
  Tensor<double, 2> XX({{3, 4}});
  for (int i = 0; i < XX.sizes()[0]; ++i) {
    for (int j = 0; j < XX.sizes()[1]; ++j) {
      double value = i * XX.sizes()[0] + j;
      XX(i, j) = value;
    }
  }
  double mult = 12;
  Tensor<double, 2> X = XX * mult;
  for (int i = 0; i < XX.sizes()[0]; ++i) {
    for (int j = 0; j < XX.sizes()[1]; ++j) {
      double value = i * XX.sizes()[0] + j;
      ASSERT_DOUBLE_EQ(value*mult, X(i, j));
      ASSERT_DOUBLE_EQ(value, XX(i, j));
    }
  }
  XX *= 10;
  for (int i = 0; i < XX.sizes()[0]; ++i) {
    for (int j = 0; j < XX.sizes()[1]; ++j) {
      double value = i * XX.sizes()[0] + j;
      ASSERT_DOUBLE_EQ(value*10, XX(i, j));
    }
  }

  ASSERT_ANY_THROW(XX * X);
}

TEST(TensorTest, RemoteDataRef) {
  size_t n = 256;
  std::vector<double> X(n, 0.0);
  TensorView<double, 1> Y(X.data(), {{X.size()}});
  TensorView<double, 2> Z(X.data(), {{16, 16}});
  for(int i = 0; i< X.size(); ++i) {
    X[i] = i*0.5;
  }
  for(int i = 0; i<Y.sizes()[0]; ++i) {
    ASSERT_DOUBLE_EQ(X[i], Y(i));
  }
  for(int i = 0; i<Z.sizes()[0]; ++i) {
    TensorView<double, 1> W = Z(i);
    for (int j = 0; j < Z.sizes()[1]; ++j) {
      W(j) += 10;
      ASSERT_DOUBLE_EQ(X[i*Z.sizes()[1] + j], Z(i, j));
      ASSERT_DOUBLE_EQ(X[i*Z.sizes()[1] + j], W(j));
    }
  }
}

TEST(TensorTest, Inversion) {
  size_t n = 40;
  Eigen::MatrixXd M(n,n);
  M = Eigen::MatrixXd::Identity(n, n);

  M(0, n - 1) = 1.0;
  M(n - 1, 0) = -1.0;
  Tensor<double, 2> X({{n, n}});
  for(int i = 0; i< n; ++i){
    for (int j = 0; j < n; ++j) {
      X(i, j) = M(i, j);
    }
  }
  M = M.inverse();
  X = X.inverse();
  for(int i = 0; i< n; ++i){
    for (int j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(X(i, j), M(i, j));
    }
  }
  Tensor<double, 2> Z = X.dot(X.inverse());
  for(int i = 0; i< n; ++i){
    for (int j = 0; j < n; ++j) {
      ASSERT_DOUBLE_EQ(Z(i, j), i==j? 1.0 : 0.0);
    }
  }
  Tensor<double, 2> Y = X.inverse();
  ASSERT_DOUBLE_EQ(Y(n-1, 0), -1.0);
  ASSERT_DOUBLE_EQ(Y(0, n-1), 1.0);
}

TEST(TensorTest, SliceInversion) {
  size_t N = 40;
  size_t K = 10;
  Tensor<double, 3> X({{K, N, N}});

  for(int k = 0; k < K; ++k) {
    Eigen::MatrixXd M(N,N);
    M = Eigen::MatrixXd::Identity(N, N);
    M *= (k+1);
    M(0, N - 1) = 1.0;
    M(N - 1, 0) = -1.0;
    Tensor<double, 2> Z = X(k);
    for(int i = 0; i< N; ++i){
      for (int j = 0; j < N; ++j) {
        Z(i, j) = M(i, j);
      }
    }
    M = M.inverse();
    Z = Z.inverse();
    for(int i = 0; i< N; ++i){
      for (int j = 0; j < N; ++j) {
        ASSERT_DOUBLE_EQ(Z(i, j), M(i, j));
      }
    }
  }
}

TEST(TensorTest, BasicArithmetics) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(N, N);
  Tensor<double, 2> X({{N, N}});
  Tensor<double, 2> Z({{N, N}});
  for(int i = 0; i< N; ++i) {
    for (int j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
      Z(i,j) = M2(i,j);
    }
  }
  auto M3 = M1 + M2;
  auto Y = X + Z;
  for(int i = 0; i< N; ++i){
    for (int j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j), M3(i, j));
    }
  }
  Y -= Z;
  for(int i = 0; i< N; ++i){
    for (int j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j), X(i, j));
    }
  }
}

TEST(TensorTest, BasicArithmeticsView) {
  size_t N = 10;
  Tensor<double, 3> W({{N,N,N}});
  W *= 0.0;
  for (int k = 0; k < N; ++k) {
    Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
    Eigen::MatrixXd M2 = Eigen::MatrixXd::Random(N, N);
    TensorView<double, 2> X = W(k);
    Tensor<double, 2> Z({{N, N}});
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        X(i, j) = M1(i, j);
        Z(i, j) = M2(i, j);
      }
    }
    auto M3 = M1 + M2;
    Tensor<double, 2> Y = X + Z;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        ASSERT_DOUBLE_EQ(Y(i, j), M3(i, j));
        ASSERT_DOUBLE_EQ(X(i, j), M1(i, j));
        ASSERT_DOUBLE_EQ(Z(i, j), M2(i, j));
      }
    }
    Y -= Z;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        ASSERT_NEAR(Y(i, j), X(i, j), 1e-10);
      }
    }
  }
}

TEST(TensorTest, DoubleScaleByComplex) {
  size_t N = 10;
  Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
  std::complex<double> x = 5.0;
  Tensor<double, 2> X({{N, N}});
  for(int i = 0; i< N; ++i) {
    for (int j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
    }
  }
  auto M3 = M1 * x;
  auto Y = X * x;
  for(int i = 0; i< N; ++i){
    for (int j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j).real(), M3(i, j).real());
      ASSERT_DOUBLE_EQ(Y(i, j).imag(), M3(i, j).imag());
    }
  }
}

TEST(TensorTest, DoubleScaleByComplexView) {
  size_t N = 10;
  size_t M = 10;
  std::complex<double> x = 5.0;
  Tensor<double, 3> Z({{M, N, N}});
  for(int k = 0; k< M; ++k) {
    Eigen::MatrixXd M1 = Eigen::MatrixXd::Random(N, N);
    auto X =  Z(k);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        X(i, j) = M1(i, j);
      }
    }
    auto M3 = M1 * x;
    auto Y = X * x;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
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
  Tensor<double, 2> X({{N, N}});
  Tensor<std::complex<double>, 2> Z({{N, N}});
  for(int i = 0; i< N; ++i) {
    for (int j = 0; j < N; ++j) {
      X(i,j) = M1(i,j);
      Z(i,j) = M2(i,j);
    }
  }
  auto M3 = M1 + M2;
  auto Y = Z + X;
  for(int i = 0; i< N; ++i){
    for (int j = 0; j < N; ++j) {
      ASSERT_DOUBLE_EQ(Y(i, j).real(), M3(i, j).real());
      ASSERT_DOUBLE_EQ(Y(i, j).imag(), M3(i, j).imag());
    }
  }
//  Y -= Z;
  for(int i = 0; i< N; ++i){
    for (int j = 0; j < N; ++j) {
//      ASSERT_DOUBLE_EQ(Y(i, j), X(i, j));
    }
  }
}
