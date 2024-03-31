using System.Runtime.InteropServices;
using System.Security;

namespace CifarPrincipalComponentAnalysis.Utilities;

internal static unsafe class CudaCppDllFunctions
{
    [DllImport("PrincipalComponentAnalysisWithCuda.dll"), SuppressUnmanagedCodeSecurity]
    public static extern void solve_principal_component_analysis([In] double* m_array_input,
                                                                 int rows,
                                                                 int columns,
                                                                 [Out] double* v_array_mean_out,
                                                                 [Out] double* m_array_cov_out,
                                                                 [Out] double* v_array_eigenvalues_out,
                                                                 [Out] double* m_array_eigenvectors_out,
                                                                 int calculation_method);
}
