#include "PrincipalComponentAnalysis.h"

using namespace std;
using namespace Eigen;

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


template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
    T rows; // --- Number of rows
    __host__ __device__ linear_index_to_row_index(T rows) : rows(rows) {}
    __host__ __device__ T operator()(T i) { return i % rows; }
};


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
                                                  double* m_array_eigenvectors_out)
{
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // Create cublas handle, bind a stream
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    thrust::device_vector<double> d_M_input(rows * columns); // M_input
    thrust::device_vector<double> d_V_ones(columns, 1.f);    // V_ones = [1, ..., 1]^t
    thrust::device_vector<double> d_V_mean(rows);            // V_mean =  1 / columns * M_input * [1, ..., 1]^t
    thrust::device_vector<double> d_M_cov(rows * rows);      // M_cov = A * A^t (A = M_input - [V_mean, ..., V_mean])
    thrust::device_vector<double> d_M_eingenvectors_row_major(rows * rows);  // Eigenvectors of M_cov (row-major matrix)

    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_M_input.data()), m_array_input, sizeof(double) * rows * columns,
                               cudaMemcpyHostToDevice, stream));

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::cout << "M_input: " << d_M_input[0] << ", ..., " << d_M_input[(columns - 1) * rows] << std::endl;
    std::cout << "M_input: " << "......" << std::endl;
    std::cout << "M_input: " << d_M_input[rows - 1] << ", ..., " << d_M_input[rows * columns - 1] << std::endl << std::endl;
#endif
    
    // -------------------------------------------------------------------------------------------
    // Calculation of Mean Vector: V_mean = 1 / columns * M_input * [1, ..., 1]^t
    // -------------------------------------------------------------------------------------------
    double alpha = 1.0 / columns;
    double beta = 0.0;
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, rows, columns, &alpha, thrust::raw_pointer_cast(d_M_input.data()), rows, 
                             thrust::raw_pointer_cast(d_V_ones.data()), 1, &beta, thrust::raw_pointer_cast(d_V_mean.data()), 1));

#ifdef __ENABLE_CONSOLE_TEST_CODES__    
    std::cout << "V_mean: " << d_V_mean[0] << ", ..., " << d_V_mean[rows - 1] << std::endl << std::endl;
#endif
    
    // -------------------------------------------------------------------------------------------
    // Subtract mean colum vector from input Matrix columns: A = M_input - [V_mean, ..., V_mean]
    // -------------------------------------------------------------------------------------------
    thrust::transform(d_M_input.begin(), d_M_input.end(),
                      thrust::make_permutation_iterator(d_V_mean.begin(),
                                                        thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                                                                        linear_index_to_row_index<int>(rows))),
                      d_M_input.begin(),
                      thrust::minus<double>());

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::cout << "M_input - V_mean: " << d_M_input[0] << ", ..., " << d_M_input[(columns - 1) * rows] << std::endl;
    std::cout << "M_input - V_mean: " << "......" << std::endl;
    std::cout << "M_input - V_mean: " << d_M_input[rows - 1] << ", ..., " << d_M_input[rows * columns - 1] << std::endl << std::endl;
#endif
    
    // -------------------------------------------------------------------------------------------
    // Calculation of Covariance Matrix: M_cov = A * A^t (A = M_input - [V_mean, ..., V_mean])
    // -------------------------------------------------------------------------------------------
    alpha = 1.0;
    beta = 0.0;
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, rows, rows, columns, &alpha,
                             thrust::raw_pointer_cast(d_M_input.data()), rows,
                             thrust::raw_pointer_cast(d_M_input.data()), rows, &beta,
                             thrust::raw_pointer_cast(d_M_cov.data()), rows));

    // Copy data to host
    CUDA_CHECK(cudaMemcpyAsync(v_array_mean_out, thrust::raw_pointer_cast(d_V_mean.data()), sizeof(double) * rows,
                               cudaMemcpyDeviceToHost, stream));    
    CUDA_CHECK(cudaMemcpyAsync(m_array_cov_out, thrust::raw_pointer_cast(d_M_cov.data()), sizeof(double) * rows * rows,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // ---------------------------------------------------------------
    // Eigendecomposition of a covariance matrix
    // ---------------------------------------------------------------
    cusolverDnHandle_t cusolverH = NULL;
    cusolverDnParams_t params = NULL;

    thrust::device_vector<double> d_V_eigenvalues(rows);
    thrust::device_vector<int> d_info(1);
    int info = 0;

    size_t workspaceInBytesOnDevice = 0; // size of workspace
    void *d_work = nullptr;              // device workspace
    size_t workspaceInBytesOnHost = 0;   // size of workspace
    void *h_work = nullptr;              // host workspace

    // Create cusolver handle, bind a stream
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    // Query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // Compute eigenvalues and eigenvectors
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER; // Lower triangle of d_M_cov is stored
    
    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(cusolverH, params, jobz, uplo, rows, CUDA_R_64F, thrust::raw_pointer_cast(d_M_cov.data()), rows,
                                               CUDA_R_64F, thrust::raw_pointer_cast(d_V_eigenvalues.data()), CUDA_R_64F,
                                               &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnDevice = %zu\n", workspaceInBytesOnDevice);
    std::printf("cusolverDnXsyevd_bufferSize: workspaceInBytesOnHost = %zu\n", workspaceInBytesOnHost);
#endif

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));

    if (workspaceInBytesOnHost > 0) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // Compute eigendecomposition
    CUSOLVER_CHECK(cusolverDnXsyevd(cusolverH, params, jobz, uplo, rows, CUDA_R_64F,
                                    thrust::raw_pointer_cast(d_M_cov.data()), rows, CUDA_R_64F,
                                    thrust::raw_pointer_cast(d_V_eigenvalues.data()), CUDA_R_64F, d_work, workspaceInBytesOnDevice,
                                    h_work, workspaceInBytesOnHost, thrust::raw_pointer_cast(d_info.data())));

    // Apply thrust::reverse() to make the order of eigenvalues decreasing order
    thrust::reverse(d_V_eigenvalues.begin(), d_V_eigenvalues.end());

    // Since the order of eigenvalues is reversed,
    // reverse the column order of eigenvectors matrix (column-major matrix)
    int m_eigenvectors_size = rows * rows;
    int m_eigenvectors_rows = rows;
    int m_eigenvectors_columns = rows;    
    double *d_M_eigenvectors_ptr = (double *)thrust::raw_pointer_cast(d_M_cov.data());
    auto counting = thrust::make_counting_iterator<int>(0);
    thrust::for_each(thrust::cuda::par.on(stream), counting,
                     counting + (m_eigenvectors_size / 2), [=] __device__(int idx) {
                       int dest_row = idx % m_eigenvectors_rows;
                       int dest_col = idx / m_eigenvectors_rows;
                       int src_row = dest_row;
                       int src_col = (m_eigenvectors_columns - dest_col) - 1;
                       int src_idx = src_col * m_eigenvectors_rows + src_row;
                       double temp = d_M_eigenvectors_ptr[idx];
                       d_M_eigenvectors_ptr[idx] = d_M_eigenvectors_ptr[src_idx];
                       d_M_eigenvectors_ptr[src_idx] = temp;
                     });

    // Make the order of eigenvectors matrix (column-major matrix) row-major matrix : Transpose
    alpha = 1.0;
    beta = 0.0;
    CUBLAS_CHECK(cublasDgeam(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, rows, rows, &alpha,
                             d_M_eigenvectors_ptr, rows, &beta,
                             d_M_eigenvectors_ptr, rows,
                             thrust::raw_pointer_cast(d_M_eingenvectors_row_major.data()), rows));

    CUDA_CHECK(cudaMemcpyAsync(m_array_eigenvectors_out, thrust::raw_pointer_cast(d_M_eingenvectors_row_major.data()),
                               sizeof(double) * rows * rows,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(v_array_eigenvalues_out, thrust::raw_pointer_cast(d_V_eigenvalues.data()), sizeof(double) * rows,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, thrust::raw_pointer_cast(d_info.data()), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::printf("after Xsyevd: info = %d\n", info);
#endif
    
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

#ifdef __ENABLE_CONSOLE_TEST_CODES__    
    std::printf("\n\n");
#endif
    
    // free resources
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
}
