#pragma once

#define DLLEXPORT extern "C" __declspec(dllexport)
//#define __ENABLE_CONSOLE_TEST_CODES__

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <library_types.h>

#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/for_each.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

/**
 * Calculate Principal Component Analysis with Eigen (C++ implementation)
 *
 *   - Since Eigen matrix is column-major, m_array_input's rows and columns are swapped from C# interface
 *
 * @param m_array_input  Pointer to an input matrix values (length: rows * columns): column-major matrix (M_input)
 * @param columns  Number of columns of an input matrix: Number of sample vectors
 * @param rows  Number of rows of an input matrix: Size of sample vector
 * @param v_array_mean_out  Pointer to an output mean vector (length: rows): Mean of sample vectors
 * @param m_array_cov_out  Pointer to an output covariance martix (length: rows * rows): Covariance matrix of sample vectors
 * @param v_array_eigenvalues_out  Pointer to an output eigenvalues vector (length: rows): Eigenvalues of covariance matrix
 * @param m_array_eigenvectors_out  Pointer to an output eigenvectors martix (length: rows * rows): Eigenvectors of covariance matrix (row-major)
 */
void solve_principal_component_analysis_with_eigen(double* m_array_input,
                                                   const int columns,
                                                   const int rows,
                                                   double* v_array_mean_out,
                                                   double* m_array_cov_out,
                                                   double* v_array_eigenvalues_out,
                                                   double* m_array_eigenvectors_out);

/**
 * Calculate Principal Component Analysis with cuBLAS and cuSOLVER (CUDA implementation)
 *
 *   - Since cuBLAS and cuSOLVER matrix is column-major, m_array_input's rows and columns are swapped from C# interface
 *
 * @param m_array_input  Pointer to an input matrix values (length: rows * columns): column-major matrix (M_input)
 * @param columns  Number of columns of an input matrix: Number of sample vectors
 * @param rows  Number of rows of an input matrix: Size of sample vector
 * @param v_array_mean_out  Pointer to an output mean vector (length: rows): Mean of sample vectors
 * @param m_array_cov_out  Pointer to an output covariance martix (length: rows * rows): Covariance matrix of sample vectors
 * @param v_array_eigenvalues_out  Pointer to an output eigenvalues vector (length: rows): Eigenvalues of covariance matrix
 * @param m_array_eigenvectors_out  Pointer to an output eigenvectors martix (length: rows * rows): Eigenvectors of covariance matrix (row-major)
 */
void solve_principal_component_analysis_with_cuda(const double* m_array_input,
                                                  const int columns,
                                                  const int rows,
                                                  double* v_array_mean_out,
                                                  double* m_array_cov_out,
                                                  double* v_array_eigenvalues_out,
                                                  double* m_array_eigenvectors_out);

