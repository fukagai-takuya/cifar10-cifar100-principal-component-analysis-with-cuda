//
// Comment out USE_SINGLE_FLOATINGPOINT_WITH_CUDA when "double" is used for CUDA
//
// #define USE_SINGLE_FLOATINGPOINT_WITH_CUDA

#ifdef USE_SINGLE_FLOATINGPOINT_WITH_CUDA
using data_type = float;
#else // USE_SINGLE_FLOATINGPOINT_WITH_CUDA
using data_type = double;
#endif // USE_SINGLE_FLOATINGPOINT_WITH_CUDA

#define DLLEXPORT extern "C" __declspec(dllexport)

#include <iostream>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_api.h>
#include <library_types.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

/**
 * Compute Eigendecomposition of a symmetric matrix with CUDA
 *
 *   - Since m_symmetric_in stores values of a symmetric matrix,
 *     it is not affected by the difference between column-major and row-major.
 *   - m_array_eigenvectors_out are storead in column-major order,
 *     since cuSOLVER expects dense matrices are assumed to be stored in column-major order in memory.
 *
 * @param m_symmetric_in  Pointer to an input symmetric matrix values
 * @param columns  Number of columns (or rows) of an input symmetric matrix
 * @param v_array_eigenvalues_out  Pointer to output eigenvalues
 * @param m_array_eigenvectors_out  Pointer to output eigenvectors
 * @param cudaDataType  Type of data (this function supports CUDA_R_32F and CUDA_R_64F only)
 */
void eigendecomposition_with_cuda(const data_type* m_symmetric_in,
                                  const int columns,
                                  data_type* v_array_eigenvalues_out,
                                  data_type* m_array_eigenvectors_out,
                                  cudaDataType_t cudaDataType)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    data_type *d_M_cov = nullptr; // device pointer for covariance matrix
    int size_of_matrix = columns * columns; // size of covariance and eigenvectors matrix
    
    data_type *d_V_lambda = nullptr; // device pointer for eigenvalues vector
    int size_of_vector = columns; // size of eigenvalues vector

    int *d_info = nullptr;
    int info = 0;

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace */

    // step 1: create cusolver handle, bind a stream
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    // step 2: allocate device memory and copy matrix data
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_M_cov), sizeof(data_type) * size_of_matrix));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V_lambda), sizeof(data_type) * size_of_vector));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_M_cov, m_symmetric_in, sizeof(data_type) * size_of_matrix, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; // lower triangle of d_M_cov is stored

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(cusolverH, params, jobz, uplo, size_of_vector, cudaDataType, d_M_cov, size_of_vector,
                                               cudaDataType, d_V_lambda, cudaDataType, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnDevice = %zu\n", workspaceInBytesOnDevice);
    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnHost = %zu\n", workspaceInBytesOnHost);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // step 4: compute eigendecomposition
    CUSOLVER_CHECK(cusolverDnXsyevd(cusolverH, params, jobz, uplo, size_of_vector, cudaDataType, d_M_cov, size_of_vector, cudaDataType, d_V_lambda,
                                    cudaDataType, d_work, workspaceInBytesOnDevice, h_work,  workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpyAsync(m_array_eigenvectors_out, d_M_cov, sizeof(data_type) * size_of_matrix, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_array_eigenvalues_out, d_V_lambda, sizeof(data_type) * size_of_vector, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xsyevd: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    // free resources
    CUDA_CHECK(cudaFree(d_M_cov));
    CUDA_CHECK(cudaFree(d_V_lambda));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    std::printf("\n");
}

/*
void eigendecomposition_with_cuda(const double* m_symmetric_in,
                                  const int columns,
                                  double* v_array_eigenvalues_out,
                                  double* m_array_eigenvectors_out)
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;

    double *d_M_cov = nullptr; // device pointer for covariance matrix
    int size_of_matrix = columns * columns; // size of covariance and eigenvectors matrix
    
    double *d_V_lambda = nullptr; // device pointer for eigenvalues vector
    int size_of_vector = columns; // size of eigenvalues vector

    int *d_info = nullptr;
    int info = 0;

    size_t workspaceInBytesOnDevice = 0; // size of workspace
    void *d_work = nullptr;              // device workspace
    size_t workspaceInBytesOnHost = 0;   // size of workspace
    void *h_work = nullptr;              // host workspace

    // step 1: create cusolver handle, bind a stream
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    // step 2: allocate device memory and copy matrix data
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_M_cov), sizeof(double) * size_of_matrix));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_V_lambda), sizeof(double) * size_of_vector));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_M_cov, m_symmetric_in, sizeof(double) * size_of_matrix, cudaMemcpyHostToDevice, stream));

    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; // lower triangle of d_M_cov is stored

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(cusolverH, params, jobz, uplo, size_of_vector, CUDA_R_64F, d_M_cov, size_of_vector,
                                               CUDA_R_64F, d_V_lambda, CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnDevice = %zu\n", workspaceInBytesOnDevice);
    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnHost = %zu\n", workspaceInBytesOnHost);

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // step 4: compute eigendecomposition
    CUSOLVER_CHECK(cusolverDnXsyevd(cusolverH, params, jobz, uplo, size_of_vector, CUDA_R_64F, d_M_cov, size_of_vector,
                                    CUDA_R_64F, d_V_lambda, CUDA_R_64F, d_work, workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpyAsync(m_array_eigenvectors_out, d_M_cov, sizeof(double) * size_of_matrix, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_array_eigenvalues_out, d_V_lambda, sizeof(double) * size_of_vector, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xsyevd: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    // free resources
    CUDA_CHECK(cudaFree(d_M_cov));
    CUDA_CHECK(cudaFree(d_V_lambda));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    std::printf("\n\n");
}
*/

/**
 * Performs the matrix-matrix multiplication with cuBLAS
 *
 * @param m_array_in  Pointer to an input matrix values (length: rows * columns): A
 * @param rows  Number of rows of an input matrix: Number of samples
 * @param columns  Number of columns of an input matrix: Size of sample data
 * @param m_array_out  Pointer to an output matrix values (length: columns * columns): A^t * A
 */
void matrix_multiplication_with_cuda(const data_type* m_array_in,
                                     const int rows,
                                     const int columns,
                                     data_type* m_array_out)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;  // A
    data_type *d_X = nullptr;  // X = A^t * A

    // step 1: create cublas handle, bind a stream
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // step 2: copy data to device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * rows * columns));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(data_type) * columns * columns));

    CUDA_CHECK(cudaMemcpyAsync(d_A, m_array_in, sizeof(data_type) * rows * columns, cudaMemcpyHostToDevice, stream));

    // step 3: compute: X = A^t * A
#ifdef USE_SINGLE_FLOATINGPOINT_WITH_CUDA
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, columns, columns, rows, &alpha, d_A, rows, d_A, rows, &beta, d_X, columns));
#else //  USE_SINGLE_FLOATINGPOINT_WITH_CUDA
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, columns, columns, rows, &alpha, d_A, rows, d_A, rows, &beta, d_X, columns));    
#endif //  USE_SINGLE_FLOATINGPOINT_WITH_CUDA
    
    // step 4: copy data to host
    CUDA_CHECK(cudaMemcpyAsync(m_array_out, d_X, sizeof(data_type) * columns * columns, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
}

/*
void matrix_multiplication_with_cuda(const double* m_array_in,
                                     const int rows,
                                     const int columns,
                                     double* m_array_out)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    const double alpha = 1.0;
    const double beta = 0.0;

    double *d_A = nullptr;  // A
    double *d_X = nullptr;  // X = A^t * A

    // step 1: create cublas handle, bind a stream
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    // step 2: copy data to device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * rows * columns));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_X), sizeof(double) * columns * columns));

    CUDA_CHECK(cudaMemcpyAsync(d_A, m_array_in, sizeof(double) * rows * columns, cudaMemcpyHostToDevice, stream));

    // step 3: compute: X = A^t * A
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, columns, columns, rows, &alpha, d_A, rows, d_A, rows, &beta, d_X, columns));

    // step 4: copy data to host
    CUDA_CHECK(cudaMemcpyAsync(m_array_out, d_X, sizeof(double) * columns * columns, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // free resources
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_X));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
}
*/


DLLEXPORT void solve_principal_component_analysis(double* m_array_input,
                                                  const int rows,
                                                  const int columns,
                                                  double* v_array_mean_out,
                                                  double* m_array_cov_out,
                                                  double* v_array_eigenvalues_out,
                                                  double* m_array_eigenvectors_out)
{
    cout << "--- C++ codes (START) ---" << endl;
    cout << "--- solve_principal_component_analysis (START) ---" << endl;

    // Map m_array_input to MatrixXd
    // rows: Number of sample row vectors
    // columns: Size of sample row vectors
    // - Make memory array row-major to map m_array_input (Eigen's default order is column-major)
    // - C# two dimensional array double[,] is row-major
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> m_input(m_array_input, rows, columns);

    cout << "m_input is prepared." << endl;

    // ---------------------------------------------------------------
    // Calculation of Covariance Matrix
    // ---------------------------------------------------------------

    // Map v_array_mean_out to VectorXd
    // - The size of mean vector is columns
    Map<VectorXd> v_mean(v_array_mean_out, columns);

    // Calculate the mean of sample row vectors
    v_mean = m_input.colwise().mean();

    cout << "v_mean" << endl;
    cout << v_mean(0) << ", ..., " << v_mean(columns - 1) << endl << endl;    
    
    // Subtract v_mean.transpose() from each sample row vector
    m_input = m_input.rowwise() - v_mean.transpose();

#ifdef USE_SINGLE_FLOATINGPOINT_WITH_CUDA
    
    // 1. Convert double array to float array to make calculation faster and reduce memory usage for GPGPU.
    // 2. Convert array order from row-major to column-major for cuBLAS
    float* m_array_input_float = new float[rows * columns];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // row-major to column-major
            m_array_input_float[j * rows + i] = (float) m_array_input[i * columns + j];
        }
    }
    
    float* m_array_cov_float = new float[columns * columns];
    matrix_multiplication_with_cuda(m_array_input_float, rows, columns, m_array_cov_float);

#else // USE_SINGLE_FLOATINGPOINT_WITH_CUDA

    // Convert array order from row-major to column-major for cuBLAS
    double* m_array_input_column_major = new double[rows * columns];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            // row-major to column-major
            m_array_input_column_major[j * rows + i] = m_array_input[i * columns + j];
        }
    }
    std::memcpy(m_array_input, m_array_input_column_major, sizeof(double) * rows * columns);
    delete[] m_array_input_column_major;
    matrix_multiplication_with_cuda(m_array_input, rows, columns, m_array_cov_out);
    
#endif // USE_SINGLE_FLOATINGPOINT_WITH_CUDA

    
    // ---------------------------------------------------------------
    // Eigendecomposition of a covariance matrix
    // ---------------------------------------------------------------
    
#ifdef USE_SINGLE_FLOATINGPOINT_WITH_CUDA
    
    // Convert double array to float array to make calculation faster and reduce memory usage for GPGPU.
    float* v_array_eigenvalues_float = new float[columns];
    float* m_array_eigenvectors_float = new float[columns * columns];
    
    eigendecomposition_with_cuda(m_array_cov_float, columns, v_array_eigenvalues_float, m_array_eigenvectors_float, CUDA_R_32F);

    for (int i = 0; i < columns; i++) {
        v_array_eigenvalues_out[i] = v_array_eigenvalues_float[i];
    }
    
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < columns; j++) {
            m_array_cov_out[i * columns + j] = m_array_cov_float[i * columns + j];
            m_array_eigenvectors_out[i * columns + j] = m_array_eigenvectors_float[i * columns + j];
        }
    }

#else // USE_SINGLE_FLOATINGPOINT_WITH_CUDA

    eigendecomposition_with_cuda(m_array_cov_out, columns, v_array_eigenvalues_out, m_array_eigenvectors_out, CUDA_R_64F);
    
#endif // USE_SINGLE_FLOATINGPOINT_WITH_CUDA

    // Map v_array_eigenvalues_out to VectorXd
    Map<VectorXd> eigenvalues(v_array_eigenvalues_out, columns);
    // Apply reverse() to make the order of eigenvalues decreasing order
    eigenvalues = eigenvalues.reverse().eval();

    // Map m_array_eigenvectors_out to MatrixXd
    // - Both cuSOLVER and Eigen expects dense matrices are stored in column-major order in memory.
    //   So, m_array_eigenvectors_out is mapped with Eigen's default mapping.
    // - eigenvectors matrix is transposed at the end of this function to make the order of m_array_eigenvectors_out row-major
    Map<MatrixXd> eigenvectors(m_array_eigenvectors_out, columns, columns);

    // Apply reverse() to eigenvectors since the order of eigenvalues are reversed
    eigenvectors = eigenvectors.rowwise().reverse().eval();

    // ---------------------------------------------------------------
    // Test codes (Start)
    // ---------------------------------------------------------------
    // Map m_array_cov_out to MatrixXd
    // - The size of rows are the same as the input columns
    // - Since m_array_cov_out stores values of a symmetric matrix,
    //   it is not affected by the difference between column-major and row-major.
    Map<MatrixXd> m_cov(m_array_cov_out, columns, columns);

    cout << "m_cov is prepared." << endl;
    cout << m_cov(0, 0) << ", ..., " << m_cov(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << m_cov(columns - 1, 0) << ", ..., " << m_cov(columns - 1, columns - 1) << endl << endl;
    
    cout << "eigenvalues" << endl;
    cout << eigenvalues(0) << ", ..., " << eigenvalues(columns - 1) << endl << endl;

    cout << "eigenvectors" << endl;
    cout << eigenvectors(0, 0) << ", ..., " << eigenvectors(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << eigenvectors(columns - 1, 0) << ", ..., " << eigenvectors(columns - 1, columns - 1) << endl << endl;

    MatrixXd M_cov_V = m_cov * eigenvectors;
    cout << "Check Result: M_cov * V" << endl;
    cout << M_cov_V(0, 0) << ", ..., " << M_cov_V(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << M_cov_V(columns - 1, 0) << ", ..., " << M_cov_V(columns - 1, columns - 1) << endl << endl;

    MatrixXd V_L = eigenvectors * eigenvalues.asDiagonal();
    cout << "Check Result: V * L" << endl;
    cout << V_L(0, 0) << ", ..., " << V_L(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << V_L(columns - 1, 0) << ", ..., " << V_L(columns - 1, columns - 1) << endl << endl;

    MatrixXd VVt = eigenvectors * eigenvectors.transpose();
    cout << "Check Result: V V^t" << endl;
    cout << VVt(0, 0) << ", ..., " << VVt(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << VVt(columns - 1, 0) << ", ..., " << VVt(columns - 1, columns - 1) << endl << endl;
    
    MatrixXd VLVt = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    cout << "Check Result: V L V^t" << endl;
    cout << VLVt(0, 0) << ", ..., " << VLVt(0, columns - 1) << endl;
    cout << "..." << endl;    
    cout << VLVt(columns - 1, 0) << ", ..., " << VLVt(columns - 1, columns - 1) << endl << endl;
    // ---------------------------------------------------------------
    // Test codes (End)
    // ---------------------------------------------------------------
    
    // Make the order of m_array_eigenvectors_out row-major
    eigenvectors = eigenvectors.transpose().eval();
    
    cout << "--- solve_principal_component_analysis (END) ---" << endl;
    cout << "--- C++ codes (END) ---" << endl;
}
