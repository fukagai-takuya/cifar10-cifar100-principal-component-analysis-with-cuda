using Prism.Commands;
using Prism.Mvvm;
using Prism.Services.Dialogs;

namespace CifarPrincipalComponentAnalysis.ViewModels
{
    public class CustomMaterialDesignMessageBoxViewModel : BindableBase, IDialogAware
    {
        public string Title => "Message Box";
        public DelegateCommand OKButton {get;}

        public CustomMaterialDesignMessageBoxViewModel()
        {
            OKButton = new DelegateCommand(OKButtonExecute);
        }

        public event Action<IDialogResult>? RequestClose;

        private string _message = string.Empty;
        public string Message
        {
            get { return _message; }
            set { SetProperty(ref _message, value); }
        }

        public bool CanCloseDialog()
        {
            return true;
        }

        public void OnDialogClosed()
        {
        }

        public void OnDialogOpened(IDialogParameters parameters)
        {
            Message = parameters.GetValue<string>(nameof(Message));
        }

        private void OKButtonExecute()
        {
            RequestClose?.Invoke(new DialogResult(ButtonResult.OK));
        }
    }
}
