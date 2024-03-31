using ConsoleCheckPcaWithCuda.Utilities;

namespace ConsoleCheckPcaWithCuda;

internal class Program
{
    private const int CALCULATE_WITH_EIGEN = 0;
    private const int CALCULATE_WITH_CUDA = 1;
    
    static void Main(string[] args)
    {
        int rows = 100;
        int columns = 5;

        int seed = 1;
        Random rnd = new Random(seed);
        
        double[,] X_input = new double[rows, columns];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                X_input[i, j] = rnd.NextDouble();
            }
        }

        //
        // Use CUDA (cuBLAS, cuSOLVER) or Eigen (C++ library) to calculate PCA
        //
        double[] vectorMean = new double[columns];
        double[,] X_cov = new double[columns, columns];
        double[] vectorEigenvalues = new double[columns];
        double[,] matrixEigenvectors = new double[columns, columns];
            
        unsafe
        {
            fixed (double* m_array_input = X_input,
                   v_array_mean_out = vectorMean,
                   m_array_cov_out = X_cov,
                   v_array_eigenvalues_out = vectorEigenvalues,
                   m_array_eigenvectors_out = matrixEigenvectors)
            {                        
                CudaCppDllFunctions.solve_principal_component_analysis(m_array_input,
                                                                       rows,
                                                                       columns,
                                                                       v_array_mean_out,
                                                                       m_array_cov_out,
                                                                       v_array_eigenvalues_out,
                                                                       m_array_eigenvectors_out,
                                                                       CALCULATE_WITH_CUDA);
            }
        }

        Console.WriteLine($"vectorMean: {vectorMean[0]}, ..., {vectorMean[columns - 1]}");
        Console.WriteLine("");
            
        Console.WriteLine($"X_cov: {X_cov[0, 0]}, ..., {X_cov[0, columns - 1]}");
        Console.WriteLine("...");
        Console.WriteLine($"X_cov: {X_cov[columns - 1, 0]}, ..., {X_cov[columns - 1, columns - 1]}");            
        Console.WriteLine("");

        Console.WriteLine($"vectorEigenvalues: {vectorEigenvalues[0]}, ..., {vectorEigenvalues[columns - 1]}");
        Console.WriteLine("");
            
        Console.WriteLine($"matrixEigenvectors: {matrixEigenvectors[0, 0]}, ..., {matrixEigenvectors[0, columns - 1]}");
        Console.WriteLine("...");
        Console.WriteLine($"matrixEigenvectors: {matrixEigenvectors[columns - 1, 0]}, ..., {matrixEigenvectors[columns - 1, columns - 1]}");
        Console.WriteLine("");
    }
}
