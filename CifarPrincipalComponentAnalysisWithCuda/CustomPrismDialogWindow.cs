using System.Windows;
using MahApps.Metro.Controls;
using Prism.Services.Dialogs;
 
namespace CifarPrincipalComponentAnalysis
{
    public partial class CustomPrismDialogWindow : MetroWindow, IDialogWindow
    {
        public IDialogResult Result { get; set; }

        private void CustomPrismDialogWindow_Loaded(object sender, RoutedEventArgs e)
        {
            if (DataContext is IDialogAware)
                Title = ((IDialogAware)DataContext).Title;

            Loaded -= CustomPrismDialogWindow_Loaded;
        }

        public CustomPrismDialogWindow()
        {
            WindowStartupLocation = System.Windows.WindowStartupLocation.CenterOwner;
            Loaded += CustomPrismDialogWindow_Loaded;
        }
    }
}
