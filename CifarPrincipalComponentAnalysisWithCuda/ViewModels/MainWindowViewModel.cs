using CifarPrincipalComponentAnalysis.Models;
using CifarPrincipalComponentAnalysis.Utilities;
using Prism.Commands;
using Prism.Mvvm;
using Prism.Services.Dialogs;
using System.Windows.Media;

namespace CifarPrincipalComponentAnalysis.ViewModels
{
    public class MainWindowViewModel : BindableBase
    {
        private CifarDataset _cifarDataset;
        private readonly Cifar10 _cifar10;
        private readonly Cifar100 _cifar100;

        private string _title = "Cifar10/Cifar100 Image Principal Component Analysis";
        public string Title
        {
            get => _title;
            set => SetProperty(ref _title, value);
        }

        private ImageSource? _imageSourceObj;
        public ImageSource? ImageSourceObj
        {
            get => _imageSourceObj;
            set
            {
                if (value == null)
                    return;

                SetProperty(ref _imageSourceObj, value);
            }
        }

        public PicturesRowsColumnsEnum PicturesRowsColumns
        {
            get => _cifarDataset.PicturesRowsColumns;
            set
            {
                if (value == _cifarDataset.PicturesRowsColumns)
                {
                    return;
                }
                else if (value == PicturesRowsColumnsEnum.PICTURES_20)
                {
                    _cifarDataset.ImageColumnNum = 20;
                    _cifarDataset.ImageRowNum = 20;
                }
                else if (value == PicturesRowsColumnsEnum.PICTURES_10)
                {
                    _cifarDataset.ImageColumnNum = 10;
                    _cifarDataset.ImageRowNum = 10;
                }
                else if (value == PicturesRowsColumnsEnum.PICTURES_5)
                {
                    _cifarDataset.ImageColumnNum = 5;
                    _cifarDataset.ImageRowNum = 5;
                }
                else if (value == PicturesRowsColumnsEnum.PICTURES_3)
                {
                    _cifarDataset.ImageColumnNum = 3;
                    _cifarDataset.ImageRowNum = 3;
                }
                else if (value == PicturesRowsColumnsEnum.PICTURES_1)
                {
                    _cifarDataset.ImageColumnNum = 1;
                    _cifarDataset.ImageRowNum = 1;
                }

                SetProperty(ref _cifarDataset.PicturesRowsColumns, value);
                UpdateImageSourceWithPageNumber();
            }
        }

        public bool IsVisiblePicture20
        {
            get => _cifarDataset.IsVisiblePicture20;
            set => SetProperty(ref _cifarDataset.IsVisiblePicture20, value);
        }

        public bool IsVisiblePicture10
        {
            get => _cifarDataset.IsVisiblePicture10;
            set => SetProperty(ref _cifarDataset.IsVisiblePicture10, value);
        }

        public bool IsVisiblePicture5
        {
            get => _cifarDataset.IsVisiblePicture5;
            set => SetProperty(ref _cifarDataset.IsVisiblePicture5, value);
        }

        public DelegateCommand ReadDatasetButtonCommand { get; }
        public bool IsEnabledReadDataset
        {
            get => _cifarDataset.IsEnabledReadDataset;
            set => SetProperty(ref _cifarDataset.IsEnabledReadDataset, value);
        }

        public DelegateCommand SelectDatasetButtonCommand { get; }
        public bool IsEnabledSelectDataset
        {
            get => _cifarDataset.IsEnabledSelectDataset;
            set => SetProperty(ref _cifarDataset.IsEnabledSelectDataset, value);
        }

        public string SelectDatasetButtonContent
        {
            get => _cifarDataset.SelectDatasetButtonContent;
            set => SetProperty(ref _cifarDataset.SelectDatasetButtonContent, value);
        }

        public Dictionary<DatasetEnum, string> DatasetDictionary { get; } = new Dictionary<DatasetEnum, string>()
        {
            {DatasetEnum.CIFAR10, "cifar-10"},
            {DatasetEnum.CIFAR100, "cifar-100"}
        };

        private DatasetEnum _selectedDataset;
        public DatasetEnum SelectedDataset
        {
            get => _selectedDataset;
            set
            {
                _selectedDataset = value;

                if (_selectedDataset == DatasetEnum.CIFAR10)
                {
                    _cifarDataset = _cifar10;
                }
                else if (_selectedDataset == DatasetEnum.CIFAR100)
                {
                    _cifarDataset = _cifar100;
                }

                RaisePropertyChanged(nameof(IsEnabledReadDataset));
                RaisePropertyChanged(nameof(IsEnabledSelectDataset));
                RaisePropertyChanged(nameof(SelectDatasetButtonContent));

                RaisePropertyChanged(nameof(IsVisiblePicture20));
                RaisePropertyChanged(nameof(IsVisiblePicture10));
                RaisePropertyChanged(nameof(IsVisiblePicture5));
                RaisePropertyChanged(nameof(PicturesRowsColumns));

                RaisePropertyChanged(nameof(IsEnabledShowDataImages));
                RaisePropertyChanged(nameof(IsEnabledShowTestImages));
                RaisePropertyChanged(nameof(IsEnabledShowPcaFilters));

                RaisePropertyChanged(nameof(IsEnabledSelectNumberOfEigenvectors));
                RaisePropertyChanged(nameof(SelectNumberOfEigenvectorsOpacity));
                RaisePropertyChanged(nameof(SelectedNumberOfEigenvectors));

                RaisePropertyChanged(nameof(IsVisiblePreviousButton));
                RaisePropertyChanged(nameof(IsVisibleNextButton));

                UpdateImageSourceWithPageNumber();
            }
        }

        string _selectedPath = "";

        public DelegateCommand ShowDataImageButtonCommand { get; }

        public bool IsEnabledShowDataImages
        {
            get => _cifarDataset.IsEnabledShowDataImages;
            set => SetProperty(ref _cifarDataset.IsEnabledShowDataImages, value);
        }

        public bool IsShowingDataImages
        {
            get => _cifarDataset.IsShowingDataImages;
            set
            {
                _cifarDataset.IsShowingDataImages = value;

                if (_cifarDataset.IsShowingDataImages == true)
                {
                    _cifarDataset.IsShowingTestImages = false;
                    _cifarDataset.IsShowingPcaFilters = false;
                }

                RaisePropertyChanged(nameof(IsEnabledSelectNumberOfEigenvectors));
                RaisePropertyChanged(nameof(SelectNumberOfEigenvectorsOpacity));
            }
        }

        public DelegateCommand ShowTestImageButtonCommand { get; }

        public bool IsEnabledShowTestImages
        {
            get => _cifarDataset.IsEnabledShowTestImages;
            set => SetProperty(ref _cifarDataset.IsEnabledShowTestImages, value);
        }

        public bool IsShowingTestImages
        {
            get => _cifarDataset.IsShowingTestImages;
            set
            {
                _cifarDataset.IsShowingTestImages = value;

                if (_cifarDataset.IsShowingTestImages == true)
                {
                    _cifarDataset.IsShowingDataImages = false;
                    _cifarDataset.IsShowingPcaFilters = false;
                }

                RaisePropertyChanged(nameof(IsEnabledSelectNumberOfEigenvectors));
                RaisePropertyChanged(nameof(SelectNumberOfEigenvectorsOpacity));
            }
        }

        public DelegateCommand ShowPcaFiltersButtonCommand { get; }

        public bool IsEnabledShowPcaFilters
        {
            get => _cifarDataset.IsEnabledShowPcaFilters;
            set => SetProperty(ref _cifarDataset.IsEnabledShowPcaFilters, value);
        }

        public bool IsShowingPcaFilters
        {
            get => _cifarDataset.IsShowingPcaFilters;
            set
            {
                _cifarDataset.IsShowingPcaFilters = value;

                if (_cifarDataset.IsShowingPcaFilters == true)
                {
                    _cifarDataset.IsShowingDataImages = false;
                    _cifarDataset.IsShowingTestImages = false;
                }

                RaisePropertyChanged(nameof(IsEnabledSelectNumberOfEigenvectors));
                RaisePropertyChanged(nameof(SelectNumberOfEigenvectorsOpacity));
            }
        }

        public bool IsEnabledSelectNumberOfEigenvectors
        {
            get
            {
                if (_cifarDataset.HasPcaFilters && (IsShowingDataImages || IsShowingTestImages))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        public double SelectNumberOfEigenvectorsOpacity
        {
            get
            {
                if (IsEnabledSelectNumberOfEigenvectors)
                {
                    return 1.0;
                }
                else
                {
                    return 0.5;
                }
            }
        }

        private Dictionary<int, string> _numberOfEigenvectorsDictionary = new Dictionary<int, string>();
        public Dictionary<int, string> NumberOfEigenvectorsDictionary => _numberOfEigenvectorsDictionary;

        public int SelectedNumberOfEigenvectors
        {
            get => _cifarDataset.SelectedNumberOfEigenvectors;
            set
            {
                _cifarDataset.SelectedNumberOfEigenvectors = value;

                if (_cifarDataset.SelectedNumberOfEigenvectors != 0)
                {
                    IsVisiblePicture20 = false;
                    IsVisiblePicture10 = false;
                    IsVisiblePicture5 = false;
                    PicturesRowsColumns = PicturesRowsColumnsEnum.PICTURES_3;
                }
                else
                {
                    IsVisiblePicture20 = true;
                    IsVisiblePicture10 = true;
                    IsVisiblePicture5 = true;
                }

                UpdateImageSourceWithPageNumber();
            }
        }

        public DelegateCommand PreviousButtonCommand { get; }
        public DelegateCommand NextButtonCommand { get; }

        public bool IsVisiblePreviousButton
        {
            get => _cifarDataset.IsVisiblePreviousButton;
            set => SetProperty(ref _cifarDataset.IsVisiblePreviousButton, value);
        }

        public bool IsVisibleNextButton
        {
            get => _cifarDataset.IsVisibleNextButton;
            set => SetProperty(ref _cifarDataset.IsVisibleNextButton, value);
        }

        public string PageNumber => _cifarDataset.PageNumber;

        public MainWindowViewModel(IDialogService dialogService)
        {
            CustomMaterialDesignMessageBox.SetDialogService(dialogService);
            _cifar10 = new Cifar10();
            _cifar100 = new Cifar100();
            _cifarDataset = _cifar10;

            // Set Number of Eigenvectors given by PCA
            _numberOfEigenvectorsDictionary.Add(0, "Cifar Image");
            for (int i = 0; i < _cifarDataset.ImageDataSize; i++)
            {
                int numberOfEigenvectors = i + 1;
                _numberOfEigenvectorsDictionary.Add(numberOfEigenvectors, numberOfEigenvectors.ToString());
            }

            ReadDatasetButtonCommand = new DelegateCommand(ReadDatasetButtonExecute).ObservesCanExecute(() => IsEnabledReadDataset);
            SelectDatasetButtonCommand = new DelegateCommand(SelectDatasetButtonExecute).ObservesCanExecute(() => IsEnabledSelectDataset);

            ShowDataImageButtonCommand = new DelegateCommand(ShowCifarImages).ObservesCanExecute(() => IsEnabledShowDataImages);
            ShowTestImageButtonCommand = new DelegateCommand(ShowCifarTestImages).ObservesCanExecute(() => IsEnabledShowTestImages);
            ShowPcaFiltersButtonCommand = new DelegateCommand(ShowPcaFiltersButtonExecute).ObservesCanExecute(() => IsEnabledShowPcaFilters);

            PreviousButtonCommand = new DelegateCommand(PreviousButtonExecute);
            NextButtonCommand = new DelegateCommand(NextButtonExecute);
        }

        private void ReadDatasetButtonExecute()
        {
            if (String.IsNullOrEmpty(_selectedPath))
            {
                CustomMaterialDesignMessageBox.Show("A dataset folder has not been selected yet.");
                return;
            }

            bool isReadingSuccess = _cifarDataset.ReadDataset(_selectedPath);
            if (!isReadingSuccess)
                return;

            IsEnabledReadDataset = false;
            IsEnabledSelectDataset = false;
            IsEnabledShowDataImages = true;
            IsEnabledShowTestImages = true;
            IsEnabledShowPcaFilters = true;

            IsVisiblePreviousButton = true;
            IsVisibleNextButton = true;

            ShowCifarImages();
        }

        private void SelectDatasetButtonExecute()
        {
            var folderDialog = new Microsoft.Win32.OpenFolderDialog
            {
                Title = "Select Folder"
            };

            if (folderDialog.ShowDialog() == true)
            {
                _selectedPath = folderDialog.FolderName;
                SelectDatasetButtonContent = "Selected";
                CustomMaterialDesignMessageBox.Show($"{_selectedPath} has been selected.");
            }
        }

        private void ShowPcaFiltersButtonExecute()
        {
            if (_cifarDataset.HasPcaFilters)
            {
                ShowPcaFilters();
            }
            else
            {
                IsEnabledShowPcaFilters = false;

                Thread thread = new Thread(PrincipalComponentAnalysisExecute);
                thread.Priority = ThreadPriority.Normal;
                thread.Start();
            }
        }

        private void PrincipalComponentAnalysisExecute()
        {
            bool isPcaSuccess = _cifarDataset.GetPcaEigenVectors();
            if (isPcaSuccess)
            {
                System.Windows.Application.Current.Dispatcher.Invoke(() =>
                {
                    IsEnabledShowPcaFilters = true;
                    ShowPcaFilters();
                });
            }
        }

        private void PreviousButtonExecute()
        {
            ImageSourceObj = _cifarDataset.GetPreviousImage();
            RaisePropertyChanged(nameof(PageNumber));
        }

        private void NextButtonExecute()
        {
            ImageSourceObj = _cifarDataset.GetNextImage();
            RaisePropertyChanged(nameof(PageNumber));
        }

        private void ShowCifarImages()
        {
            IsShowingDataImages = true;
            UpdateImageSourceWithPageNumber();
        }

        private void ShowCifarTestImages()
        {
            IsShowingTestImages = true;
            UpdateImageSourceWithPageNumber();
        }

        private void ShowPcaFilters()
        {
            IsShowingPcaFilters = true;
            UpdateImageSourceWithPageNumber();
        }

        private void UpdateImageSourceWithPageNumber()
        {
            ImageSourceObj = _cifarDataset.GetWriteableBitmap();
            RaisePropertyChanged(nameof(PageNumber));
        }
    }
}
