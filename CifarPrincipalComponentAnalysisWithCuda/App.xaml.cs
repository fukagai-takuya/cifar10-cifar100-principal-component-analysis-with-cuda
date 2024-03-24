using CifarPrincipalComponentAnalysis.ViewModels;
using CifarPrincipalComponentAnalysis.Views;
using Prism.Ioc;
using System.Windows;

namespace CifarPrincipalComponentAnalysis
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App
    {
        protected override Window CreateShell()
        {
            return Container.Resolve<MainWindow>();
        }

        protected override void RegisterTypes(IContainerRegistry containerRegistry)
        {
            containerRegistry.RegisterDialogWindow<CustomPrismDialogWindow>();
            containerRegistry.RegisterDialog<CustomMaterialDesignMessageBoxView, CustomMaterialDesignMessageBoxViewModel>();
        }
    }
}
