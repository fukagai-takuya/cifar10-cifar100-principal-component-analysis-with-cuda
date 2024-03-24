using CifarPrincipalComponentAnalysis.ViewModels;
using CifarPrincipalComponentAnalysis.Views;
using Prism.Services.Dialogs;

namespace CifarPrincipalComponentAnalysis.Utilities
{
    public class CustomMaterialDesignMessageBox
    {
        private static IDialogService? _dialogService;

        public static void SetDialogService(IDialogService dialogService)
        {
            _dialogService = dialogService;
        }

        public static void Show(string message)
        {
            var p = new DialogParameters();
            p.Add(nameof(CustomMaterialDesignMessageBoxViewModel.Message), message);
            _dialogService?.Show(nameof(CustomMaterialDesignMessageBoxView), p, null);
        }
    }
}
