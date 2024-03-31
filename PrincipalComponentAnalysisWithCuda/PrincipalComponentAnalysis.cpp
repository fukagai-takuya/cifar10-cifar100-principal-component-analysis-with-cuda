#include "PrincipalComponentAnalysis.h"

using namespace std;
using namespace Eigen;

/**
 *  Specify calculation method
 *    - Use Eigen for the Calculation of Principal Component Analysis
 */
const int CALCULATE_WITH_EIGEN = 0;

/**
 *  Specify calculation method
 *    - Use CUDA for the Calculation of Principal Component Analysis
 */
const int CALCULATE_WITH_CUDA = 1;

/**
 * Calculate Principal Component Analysis
 *
 *   - Since Eigen, cuBLAS and cuSOLVER matrix is column-major, m_array_input's rows and columns are swapped from C# interface
 *
 * @param m_array_input  Pointer to an input matrix values (length: rows * columns): column-major matrix
 * @param columns  Number of columns of an input matrix: Number of sample vectors
 * @param rows  Number of rows of an input matrix: Size of sample vector
 * @param v_array_mean_out  Pointer to an output mean vector (length: rows): Mean of sample vectors
 * @param m_array_cov_out  Pointer to an output covariance martix (length: rows * rows): Covariance matrix of sample vectors
 * @param v_array_eigenvalues_out  Pointer to an output eigenvalues vector (length: rows): Eigenvalues of covariance matrix
 * @param m_array_eigenvectors_out  Pointer to an output eigenvectors martix (length: rows * rows): Eigenvectors of covariance matrix (row-major)
 * @param calculation_method  0:Eigen, 1:CUDA
 */
DLLEXPORT void solve_principal_component_analysis(double* m_array_input,
                                                  const int columns,
                                                  const int rows,
                                                  double* v_array_mean_out,
                                                  double* m_array_cov_out,
                                                  double* v_array_eigenvalues_out,
                                                  double* m_array_eigenvectors_out,
                                                  const int calculation_method = CALCULATE_WITH_EIGEN)
{
    cout << "--- C++ codes (START) ---" << endl;
    cout << "--- solve_principal_component_analysis (START) ---" << endl;

    if (calculation_method == CALCULATE_WITH_EIGEN) {
      
        solve_principal_component_analysis_with_eigen(m_array_input,
                                                      columns,
                                                      rows,
                                                      v_array_mean_out,
                                                      m_array_cov_out,
                                                      v_array_eigenvalues_out,
                                                      m_array_eigenvectors_out);
        
    } else if (calculation_method == CALCULATE_WITH_CUDA) {
      
        solve_principal_component_analysis_with_cuda(m_array_input,
                                                     columns,
                                                     rows,
                                                     v_array_mean_out,
                                                     m_array_cov_out,
                                                     v_array_eigenvalues_out,
                                                     m_array_eigenvectors_out);
        
    }

    // ---------------------------------------------------------------
    // Test codes (Start)
    // ---------------------------------------------------------------
#ifdef __ENABLE_CONSOLE_TEST_CODES__    
    // Map m_array_cov_out to MatrixXd
    // - The size of rows are the same as the input rows
    // - Since m_array_cov_out stores values of a symmetric matrix,
    //   it is not affected by the difference between column-major and row-major.
    Map<MatrixXd> M_cov(m_array_cov_out, rows, rows);
    cout << "M_cov is prepared." << endl;
    cout << M_cov(0, 0) << ", ..., " << M_cov(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << M_cov(rows - 1, 0) << ", ..., " << M_cov(rows - 1, rows - 1) << endl << endl;

    // Map v_array_eigenvalues_out to VectorXd
    Map<VectorXd> eigenvalues(v_array_eigenvalues_out, rows);
    cout << "eigenvalues" << endl;
    cout << eigenvalues(0) << ", ..., " << eigenvalues(rows - 1) << endl << endl;

    // Map m_array_eigenvectors_out to MatrixXd
    // - Since m_array_eigenvectors_out is row-major map to row-major matrix
    Map<Matrix<double, Dynamic, Dynamic, RowMajor>> eigenvectors(m_array_eigenvectors_out, rows, rows);
    cout << "eigenvectors" << endl;
    cout << eigenvectors(0, 0) << ", ..., " << eigenvectors(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << eigenvectors(rows - 1, 0) << ", ..., " << eigenvectors(rows - 1, rows - 1) << endl << endl;

    MatrixXd M_cov_V = M_cov * eigenvectors;
    cout << "Check Result: M_cov * V" << endl;
    cout << M_cov_V(0, 0) << ", ..., " << M_cov_V(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << M_cov_V(rows - 1, 0) << ", ..., " << M_cov_V(rows - 1, rows - 1) << endl << endl;

    MatrixXd V_L = eigenvectors * eigenvalues.asDiagonal();
    cout << "Check Result: V * L" << endl;
    cout << V_L(0, 0) << ", ..., " << V_L(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << V_L(rows - 1, 0) << ", ..., " << V_L(rows - 1, rows - 1) << endl << endl;

    MatrixXd VVt = eigenvectors * eigenvectors.transpose();
    cout << "Check Result: V V^t" << endl;
    cout << VVt(0, 0) << ", ..., " << VVt(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << VVt(rows - 1, 0) << ", ..., " << VVt(rows - 1, rows - 1) << endl << endl;
    
    MatrixXd VLVt = eigenvectors * eigenvalues.asDiagonal() * eigenvectors.transpose();
    cout << "Check Result: V L V^t" << endl;
    cout << VLVt(0, 0) << ", ..., " << VLVt(0, rows - 1) << endl;
    cout << "..." << endl;    
    cout << VLVt(rows - 1, 0) << ", ..., " << VLVt(rows - 1, rows - 1) << endl << endl;
    // ---------------------------------------------------------------
    // Test codes (End)
    // ---------------------------------------------------------------
#endif    
    cout << "--- solve_principal_component_analysis (END) ---" << endl;
    cout << "--- C++ codes (END) ---" << endl;
}
