#include "PrincipalComponentAnalysis.h"

using namespace std;
using namespace Eigen;

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
                                                   double* m_array_eigenvectors_out)
{
  
    // Map m_array_input to MatrixXd
    // - Since Eigen matrix is column-major, rows and columns are swapped from C# interface
    // - M_input (column-major) is a transposed matrix of C# m_array_input matrix (row-major)
    Map<MatrixXd> M_input(m_array_input, rows, columns);

    // Map v_array_mean_out to VectorXd
    Map<VectorXd> V_mean(v_array_mean_out, rows); 

    // Map m_array_cov_out to MatrixXd
    Map<MatrixXd> M_cov(m_array_cov_out, rows, rows);

    // Map v_array_eigenvalues_out to VectorXd
    Map<VectorXd> eigenvalues(v_array_eigenvalues_out, rows);

    // Map m_array_eigenvectors_out to MatrixXd
    // - Since map m_array_cov_out is row-major, map as a row-major matrix
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> eigenvectors(m_array_eigenvectors_out, rows, rows);

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::cout << "M_input is prepared." << std::endl;        
    std::cout << M_input(0, 0) << ", ..., " << M_input(0, columns - 1) << std::endl;
    std::cout << "......" << std::endl;
    std::cout << M_input(rows - 1, 0) << ", ..., " << M_input(rows - 1, columns - 1) << std::endl << std::endl;
#endif
    
    // Calculate the mean of sample row vectors
    V_mean = M_input.rowwise().mean();

#ifdef __ENABLE_CONSOLE_TEST_CODES__    
    std::cout << "V_mean is prepared."  << std::endl;
    std::cout << V_mean(0) << ", ..., " << V_mean(rows - 1) << std::endl << std::endl;
#endif

    // Subtract v_mean.transpose() from each sample row vector
    M_input = M_input.colwise() - V_mean;

#ifdef __ENABLE_CONSOLE_TEST_CODES__
    std::cout << "M_input - V_mean is prepared." << std::endl;    
    std::cout << M_input(0, 0) << ", ..., " << M_input(0, columns - 1) << std::endl;
    std::cout << "......" << std::endl;
    std::cout << M_input(rows - 1, 0) << ", ..., " << M_input(rows - 1, columns - 1) << std::endl << std::endl;
#endif
    
    // Calculate covariance matrix
    // - Ignore the effect of constant multiplication
    // - Skip division by N or N - 1 (N is sample size)
    M_cov = M_input * M_input.transpose();

#ifdef __ENABLE_CONSOLE_TEST_CODES__    
    std::cout << "M_cov is prepared." << std::endl;
    std::cout << M_cov(0, 0) << ", ..., " << M_cov(0, rows - 1) << std::endl;
    std::cout << "..." << endl;    
    std::cout << M_cov(rows - 1, 0) << ", ..., " << M_cov(rows - 1, rows - 1) << std::endl << std::endl;
#endif
    
    // ---------------------------------------------------------------
    // Calculate the Eigendecomposition of a covariance matrix
    // ---------------------------------------------------------------
    SelfAdjointEigenSolver<MatrixXd> esolver(M_cov);
    eigenvalues = esolver.eigenvalues();
    eigenvectors = esolver.eigenvectors();

    // Apply reverse() to make the order of eigenvalues decreasing order
    eigenvalues = eigenvalues.reverse().eval();

#ifdef __ENABLE_CONSOLE_TEST_CODES__        
    std::cout << "eigenvalues" << std::endl;
    std::cout << eigenvalues(0) << ", ..., " << eigenvalues(rows - 1) << std::endl << std::endl;
#endif
    
    // Apply reverse() to eigenvectors because the order of eigenvalues are reversed
    eigenvectors = eigenvectors.rowwise().reverse().eval();

#ifdef __ENABLE_CONSOLE_TEST_CODES__        
    std::cout << "eigenvectors" << endl;
    std::cout << eigenvectors(0, 0) << ", ..., " << eigenvectors(0, rows - 1) << std::endl;
    std::cout << "..." << endl;    
    std::cout << eigenvectors(rows - 1, 0) << ", ..., " << eigenvectors(rows - 1, rows - 1) << std::endl << std::endl;
#endif    
}
