// #define CHECK_PCA
#define USE_EIGEN_AND_CUDA

using Accord.Math;
using Accord.Math.Decompositions;
using Accord.Statistics;
using System.IO;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using CifarPrincipalComponentAnalysis.Utilities;

namespace CifarPrincipalComponentAnalysis.Models
{
    public abstract class CifarDataset
    {
        public bool IsEnabledReadDataset = true;
        public bool IsEnabledSelectDataset = true;

        public bool IsEnabledShowDataImages = false;
        public bool IsShowingDataImages = false;

        public bool IsEnabledShowTestImages = false;
        public bool IsShowingTestImages = false;

        public bool IsEnabledShowPcaFilters = false;
        public bool IsShowingPcaFilters = false;

        public bool IsVisiblePreviousButton = false;
        public bool IsVisibleNextButton = false;

        public bool IsVisiblePicture20 = true;
        public bool IsVisiblePicture10 = true;
        public bool IsVisiblePicture5 = true;

        public string SelectDatasetButtonContent = "Select Dataset";
        public int SelectedNumberOfEigenvectors = 0;

        public bool HasPcaFilters
        {
            get
            {
                if (_pcaFilters != null)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
        }

        public PicturesRowsColumnsEnum PicturesRowsColumns = PicturesRowsColumnsEnum.PICTURES_20;

        public string PageNumber
        {
            get
            {
                if (_isReadingCompleted)
                {
                    return $"page {ImageIndex / (ImageRowNum * ImageColumnNum) + 1}";
                }
                else
                {
                    return "";
                }
            }
        }

        protected List<SmallImageData>? _dataImages;
        protected List<SmallImageData>? _testImages;

        private List<SmallImageData>? _pcaFilters;

        private List<SmallImageData>? DataImages
        {
            get
            {
                if (IsShowingDataImages)
                {
                    return _dataImages;
                }
                else if (IsShowingTestImages)
                {
                    return _testImages;
                }
                else
                {
                    return _pcaFilters;
                }
            }
        }

        public int ImageIndex
        {
            get
            {
                if (IsShowingDataImages)
                {
                    return _dataImageIndex;
                }
                else if (IsShowingTestImages)
                {
                    return _testImageIndex;
                }
                else
                {
                    return _pcaFilterIndex;
                }
            }
            set
            {
                if (IsShowingDataImages)
                {
                    _dataImageIndex = value;
                }
                else if (IsShowingTestImages)
                {
                    _testImageIndex = value;
                }
                else
                {
                    _pcaFilterIndex = value;
                }
            }
        }

        protected const int ImageWidth = 32;  // pixel
        protected const int ImageHeight = 32; // pixel
        protected const int ImageChannels = 3; // number of image channels (RGB)
        public int ImageDataSize => ImageWidth * ImageHeight * ImageChannels;

        protected const int ImageMargin = 2;

        protected const double DpiX = 96;
        protected const double DpiY = 96;

        protected const int ImageAreaSize = ImageWidth * ImageHeight;
        protected const int ImageByteSize = ImageAreaSize * 4; // PixelFormats.Pbgra32 has 4 bytes for each pixel
        protected const int ImageStride = ImageWidth * 4;      // PixelFormats.Pbgra32 has 4 bytes for each pixel

        protected readonly byte[] _pixels = new byte[ImageByteSize];

        protected int _dataImageIndex = 0;
        protected int _testImageIndex = 0;
        protected int _pcaFilterIndex = 0;

        protected int _imageColumnNum;
        public int ImageColumnNum
        {
            get => _imageColumnNum;
            set => _imageColumnNum = value;
        }

        protected int _imageRowNum;
        public int ImageRowNum
        {
            get => _imageRowNum;
            set => _imageRowNum = value;
        }

        protected int _areaWidth;
        protected int _areaHeight;

        protected WriteableBitmap _writeableBitmap;
        protected bool _isReadingCompleted;
        public bool IsReadingCompleted => _isReadingCompleted;

        private const string Debug_X_cov_File = "Debug_X_cov.bin";
        private const string PCA_Mean_File = "PCA_Mean.bin";
        private const string PCA_Eigenvalues_File = "PCA_Eigenvalues.bin";
        private const string PCA_Eigenvectors_File = "PCA_Eigenvectors.bin";

        private double[]? _vectorMean;
        private double[]? _vectorEigenvalues;
        private double[,]? _matrixEigenvectors;

        private string _selectedPath = "";

        public CifarDataset()
        {
            ImageColumnNum = 20;
            ImageRowNum = 20;

            _areaWidth = (ImageWidth + ImageMargin * 2) * ImageColumnNum; // pixel
            _areaHeight = (ImageHeight + ImageMargin * 2) * ImageRowNum;  // pixel

            _writeableBitmap = new WriteableBitmap(_areaWidth, _areaHeight, DpiX, DpiY, PixelFormats.Pbgra32, null);
        }

        public bool ReadDataset(string selectedPath)
        {
            string[] files = Directory.GetFiles(selectedPath);
            List<string> fileNames = new List<string>();

            foreach (string file in files)
            {
                fileNames.Add(Path.GetFileName(file));
            }

            if (CheckDirectory(selectedPath, fileNames) == false)
            {
                return false;
            }

            if (ReadFiles(selectedPath) == false)
            {
                return false;
            }

            string pca_Mean_File = Path.Combine(selectedPath, PCA_Mean_File);
            if (File.Exists(pca_Mean_File))
            {
                try
                {
                    byte[] byteArray = ReadByteArrayAsBinary(pca_Mean_File, ImageDataSize * sizeof(double));
                    _vectorMean = new double[ImageDataSize];
                    Buffer.BlockCopy(byteArray, 0, _vectorMean, 0, byteArray.Length);
                }
                catch (Exception ex)
                {
                    CustomMaterialDesignMessageBox.Show($"Failed to read a binary file({pca_Mean_File}). {ex}");
                    return false;
                }
            }

            string pca_Eigenvectors_File = Path.Combine(selectedPath, PCA_Eigenvectors_File);
            if (File.Exists(pca_Eigenvectors_File))
            {
                try
                {
                    byte[] byteArray = ReadByteArrayAsBinary(pca_Eigenvectors_File, ImageDataSize * ImageDataSize * sizeof(double));
                    _matrixEigenvectors = new double[ImageDataSize, ImageDataSize];
                    Buffer.BlockCopy(byteArray, 0, _matrixEigenvectors, 0, byteArray.Length);
                    PreparePcaFilterImages();
                }
                catch (Exception ex)
                {
                    CustomMaterialDesignMessageBox.Show($"Failed to read a binary file({pca_Eigenvectors_File}). {ex}");
                    return false;
                }
            }

#if CHECK_PCA

            string pca_Eigenvalues_File = Path.Combine(selectedPath, PCA_Eigenvalues_File);
            if (File.Exists(pca_Eigenvalues_File))
            {
                try
                {
                    byte[] byteArray = ReadByteArrayAsBinary(pca_Eigenvalues_File, ImageDataSize * sizeof(double));
                    _vectorEigenvalues = new double[ImageDataSize];
                    Buffer.BlockCopy(byteArray, 0, _vectorEigenvalues, 0, byteArray.Length);
                }
                catch (Exception ex)
                {
                    CustomMaterialDesignMessageBox.Show($"Failed to read a binary file({pca_Eigenvalues_File}). {ex}");
                    return false;
                }
            }

            string debug_X_cov_File = Path.Combine(selectedPath, Debug_X_cov_File);
            if (File.Exists(debug_X_cov_File))
            {
                try
                {
                    byte[] byteArray = ReadByteArrayAsBinary(debug_X_cov_File, ImageDataSize * ImageDataSize * sizeof(double));
                    double[,] X_cov = new double[ImageDataSize, ImageDataSize];
                    Buffer.BlockCopy(byteArray, 0, X_cov, 0, byteArray.Length);
                }
                catch (Exception ex)
                {
                    CustomMaterialDesignMessageBox.Show($"Failed to read a binary file({debug_X_cov_File}). {ex}");
                    return false;
                }
            }

            if (_matrixEigenvectors != null)
            {
                double[,] Pt_dot_P = _matrixEigenvectors.Transpose().Dot(_matrixEigenvectors);
            }

#endif // CHECK_PCA

            _isReadingCompleted = true;
            _selectedPath = selectedPath;
            return true;
        }

        public WriteableBitmap GetPreviousImage()
        {
            ImageIndex -= (ImageRowNum * ImageColumnNum);
            if (ImageIndex < 0)
            {
                // Show last page
                if (DataImages == null)
                    throw new NotSupportedException("DataImages must not be null here.");

                int remainder = DataImages.Count % (ImageRowNum * ImageColumnNum);
                if (remainder > 0)
                {
                    ImageIndex = DataImages.Count - remainder;
                }
                else
                {
                    ImageIndex += DataImages.Count;
                }
            }
            WriteImages();
            return _writeableBitmap;
        }

        public WriteableBitmap GetNextImage()
        {
            ImageIndex += (ImageRowNum * ImageColumnNum);
            if (ImageIndex >= DataImages.Count)
            {
                ImageIndex -= DataImages.Count;
            }
            WriteImages();
            return _writeableBitmap;
        }

        private void WriteImages()
        {
            _areaWidth = (ImageWidth + ImageMargin * 2) * ImageColumnNum; // pixel
            _areaHeight = (ImageHeight + ImageMargin * 2) * ImageRowNum;  // pixel
            _writeableBitmap = new WriteableBitmap(_areaWidth, _areaHeight, DpiX, DpiY, PixelFormats.Pbgra32, null);

            for (int column = 0; column < ImageColumnNum; column++)
            {
                for (int row = 0; row < ImageRowNum; row++)
                {
                    WriteSmallImage(column, row);
                }
            }
        }

        private void WriteSmallImage(int column, int row)
        {
            int remainder = ImageIndex % (ImageRowNum * ImageColumnNum);
            int initialIndex = ImageIndex - remainder;
            int n = row * ImageColumnNum + column + initialIndex;

            if (n >= DataImages.Count)
            {
                return;
            }

            if ((IsShowingDataImages || IsShowingTestImages) && SelectedNumberOfEigenvectors != 0)
            {
                SetImageRepsentedByEigenvectors(DataImages[n]);
            }
            else
            {
                for (int i = 0; i < ImageAreaSize; i++)
                {
                    _pixels[4 * i] = DataImages[n].BlueChannelData[i];      // blue
                    _pixels[4 * i + 1] = DataImages[n].GreenChannelData[i]; // green
                    _pixels[4 * i + 2] = DataImages[n].RedChannelData[i];   // red
                    _pixels[4 * i + 3] = (byte) 255;  // alpha
                }
            }

            int x = ImageMargin + (ImageWidth + ImageMargin * 2) * column;
            int y = ImageMargin + (ImageHeight + ImageMargin * 2) * row;
            _writeableBitmap.WritePixels(new Int32Rect(0, 0, ImageWidth, ImageHeight), _pixels, ImageStride, x, y);
        }

        private void SetImageRepsentedByEigenvectors(SmallImageData imageData)
        {
            double[] image = new double[ImageDataSize];

            for (int i = 0; i < ImageAreaSize; i++)
            {
                image[3 * i] = (double) imageData.BlueChannelData[i]; // blue
                image[3 * i + 1] = (double) imageData.GreenChannelData[i]; // green
                image[3 * i + 2] = (double) imageData.RedChannelData[i]; // red
            }
            image = image.Subtract(_vectorMean);

            double[] principalComponentScores = new double[SelectedNumberOfEigenvectors];
            for (int i = 0; i < SelectedNumberOfEigenvectors; i++)
            {
                principalComponentScores[i] = image.Dot(_matrixEigenvectors.GetColumn(i));
            }

            // double initial values are 0
            double[] imageRepsentedByEigenvectors = new double[ImageDataSize];
            for (int i = 0; i < SelectedNumberOfEigenvectors; i++)
            {
                imageRepsentedByEigenvectors = imageRepsentedByEigenvectors.Add(_matrixEigenvectors.GetColumn(i).Multiply(principalComponentScores[i]));
            }

            imageRepsentedByEigenvectors = imageRepsentedByEigenvectors.Add(_vectorMean);

            for (int i = 0; i < ImageAreaSize; i++)
            {
                _pixels[4 * i] = (byte) imageRepsentedByEigenvectors[3 * i]; // blue
                _pixels[4 * i + 1] = (byte) imageRepsentedByEigenvectors[3 * i + 1]; // green
                _pixels[4 * i + 2] = (byte) imageRepsentedByEigenvectors[3 * i + 2]; // red
                _pixels[4 * i + 3] = (byte) 255;  // alpha
            }
        }

        public WriteableBitmap GetWriteableBitmap()
        {
            if (_isReadingCompleted == false)
            {
                // returns empty image
                return _writeableBitmap;
            }

            WriteImages();
            return _writeableBitmap;
        }

        private void ShowMessageBoxFromBackground(string message)
        {
            System.Windows.Application.Current.Dispatcher.Invoke(() =>
            {
                CustomMaterialDesignMessageBox.Show(message);
            });
        }
        

        public bool GetPcaEigenVectors()
        {
            if (_pcaFilters == null)
            {
                try
                {
                    CalculatePCA();
                }
                catch (Exception ex)
                {
                    ShowMessageBoxFromBackground($"Failed to calculate PCA. {ex}");
                    return false;
                }
            }

            if (_pcaFilters == null)
                return false;

            IsShowingPcaFilters = true;
            WriteImages();

            return true;
        }

        private void CalculatePCA()
        {
            if (_dataImages == null)
                return;

            Console.WriteLine($"PCA calculation is started.");
            ShowMessageBoxFromBackground($"PCA calculation is started.");

            // Check PCA calculation time
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            
            int numberOfImages = _dataImages.Count;

            double[,]? X_input = new double[numberOfImages, ImageDataSize];
            for (int row = 0; row < numberOfImages; row++)
            {
                for (int column_i = 0; column_i < ImageAreaSize; column_i++)
                {
                    X_input[row, 3 * column_i] = (double) _dataImages[row].BlueChannelData[column_i]; // blue
                    X_input[row, 3 * column_i + 1] = (double) _dataImages[row].GreenChannelData[column_i]; // green
                    X_input[row, 3 * column_i + 2] = (double) _dataImages[row].RedChannelData[column_i]; // red
                }
            }

            double[,]? X_cov;

#if USE_EIGEN_AND_CUDA
            //
            // Use Eigen (C++ library) and CUDA (cuSOLVER) to calculate PCA
            //
            _vectorMean = new double[ImageDataSize];
            X_cov = new double[ImageDataSize, ImageDataSize];
            _vectorEigenvalues = new double[ImageDataSize];
            _matrixEigenvectors = new double[ImageDataSize, ImageDataSize];
            
            unsafe
            {
                fixed (double* m_array_input = X_input,
                       v_array_mean_out = _vectorMean,
                       m_array_cov_out = X_cov,
                       v_array_eigenvalues_out = _vectorEigenvalues,
                       m_array_eigenvectors_out = _matrixEigenvectors)
                {                        
                    CudaCppDllFunctions.solve_principal_component_analysis(m_array_input,
                                                                           numberOfImages,
                                                                           ImageDataSize,
                                                                           v_array_mean_out,
                                                                           m_array_cov_out,
                                                                           v_array_eigenvalues_out,
                                                                           m_array_eigenvectors_out);
                }
            }

#else
            //
            // Use Accord to calculate PCA
            //
            
            // column means
            _vectorMean = X_input.Mean(dimension: 0);

            X_input = X_input.Subtract(_vectorMean, dimension: (VectorType)0);
            X_cov = X_input.Transpose().Dot(X_input);

            X_input = null;
            GC.Collect();

            EigenvalueDecomposition evd = new EigenvalueDecomposition(X_cov, false, true);
            
            _vectorEigenvalues = evd.RealEigenvalues;
            _matrixEigenvectors = evd.Eigenvectors;

#endif // USE_EIGEN_AND_CUDA

            sw.Stop();
            Console.WriteLine($"PCA calculation processing time is {sw.Elapsed}");

#if CHECK_PCA
            #region Console messages to check the PCA calculation results
            
            Console.WriteLine($"_vectorMean: {_vectorMean[0]}, ..., {_vectorMean[ImageDataSize - 1]}");
            Console.WriteLine("");
            
            Console.WriteLine($"X_cov: {X_cov[0, 0]}, ..., {X_cov[0, ImageDataSize - 1]}");
            Console.WriteLine("...");
            Console.WriteLine($"X_cov: {X_cov[ImageDataSize - 1, 0]}, ..., {X_cov[ImageDataSize - 1, ImageDataSize - 1]}");            
            Console.WriteLine("");

            Console.WriteLine($"_vectorEigenvalues: {_vectorEigenvalues[0]}, ..., {_vectorEigenvalues[ImageDataSize - 1]}");
            Console.WriteLine("");
            
            Console.WriteLine($"_matrixEigenvectors: {_matrixEigenvectors[0, 0]}, ..., {_matrixEigenvectors[0, ImageDataSize - 1]}");
            Console.WriteLine("...");
            Console.WriteLine($"_matrixEigenvectors: {_matrixEigenvectors[ImageDataSize - 1, 0]}, ..., {_matrixEigenvectors[ImageDataSize - 1, ImageDataSize - 1]}");
            Console.WriteLine("");

            double[,] PtP = _matrixEigenvectors.Transpose().Dot(_matrixEigenvectors);
            Console.WriteLine($"P^T P: {PtP[0, 0]}, ..., {PtP[0, ImageDataSize - 1]}");
            Console.WriteLine("...");
            Console.WriteLine($"P^T P: {PtP[ImageDataSize - 1, 0]}, ..., {PtP[ImageDataSize - 1, ImageDataSize - 1]}");            
            Console.WriteLine("");

            double[,] matrixL = new double[ImageDataSize, ImageDataSize];
            for (int i = 0; i < ImageDataSize; i++)
                matrixL[i, i] = _vectorEigenvalues[i];

            double[,] PLPt = _matrixEigenvectors.Dot(matrixL).Dot(_matrixEigenvectors.Transpose());
            Console.WriteLine($"PLP^T: {PLPt[0, 0]}, ..., {PLPt[0, ImageDataSize - 1]}");
            Console.WriteLine("...");
            Console.WriteLine($"PLP^T: {PLPt[ImageDataSize - 1, 0]}, ..., {PLPt[ImageDataSize - 1, ImageDataSize - 1]}");            
            Console.WriteLine("");

            #endregion Console messages to check the PCA calculation results
#endif // CHECK_PCA
            
            #region Save Mean, X_cov, Eigenvalues, Eigenvectors
            
            byte[] byteArray = new byte[_vectorMean.Length * sizeof(double)];
            Buffer.BlockCopy(_vectorMean, 0, byteArray, 0, byteArray.Length);
            SaveByteArrayAsBinary(Path.Combine(_selectedPath, PCA_Mean_File), byteArray);

            byteArray = new byte[X_cov.Length * sizeof(double)];
            Buffer.BlockCopy(X_cov, 0, byteArray, 0, byteArray.Length);
            SaveByteArrayAsBinary(Path.Combine(_selectedPath, Debug_X_cov_File), byteArray);
            
            byteArray = new byte[_vectorEigenvalues.Length * sizeof(double)];
            Buffer.BlockCopy(_vectorEigenvalues, 0, byteArray, 0, byteArray.Length);
            SaveByteArrayAsBinary(Path.Combine(_selectedPath, PCA_Eigenvalues_File), byteArray);

            byteArray = new byte[_matrixEigenvectors.Length * sizeof(double)];
            Buffer.BlockCopy(_matrixEigenvectors, 0, byteArray, 0, byteArray.Length);
            SaveByteArrayAsBinary(Path.Combine(_selectedPath, PCA_Eigenvectors_File), byteArray);
            
            #endregion Save Mean, X_cov, Eigenvalues, Eigenvectors

            PreparePcaFilterImages();
        }

        private static void SaveByteArrayAsBinary(string path, byte[] data)
        {
            using(Stream stream = File.Open(path, FileMode.Create))
            {
                using(BinaryWriter binWriter = new BinaryWriter(stream))
                {
                    binWriter.Write(data);
                }
            }
        }

        private static byte[] ReadByteArrayAsBinary(string path, int byteSize)
        {
            using(Stream stream = File.Open(path, FileMode.Open))
            {
                using(BinaryReader binReader = new BinaryReader(stream))
                {
                    return binReader.ReadBytes(byteSize);
                }
            }
        }

        private void PreparePcaFilterImages()
        {
            _pcaFilters = new List<SmallImageData>();

            for (int column = 0; column < _matrixEigenvectors.Columns(); column++)
            {
                double[] eigenVector = _matrixEigenvectors.GetColumn(column);

                // change the range of the eigen vector values to display as the bitmaps: "Min: 0, Max: 255"
                double minValue = eigenVector.Min();
                eigenVector = eigenVector.Subtract(minValue);

                double maxValueModifier = eigenVector.Max() / 255.0;
                eigenVector = eigenVector.Divide(maxValueModifier);

                int label = column;
                string className = label.ToString();

                byte[] blueChannelData = new byte[ImageAreaSize];
                byte[] greenChannelData = new byte[ImageAreaSize];
                byte[] redChannelData = new byte[ImageAreaSize];

                for (int row = 0; row < eigenVector.Length; row += 3)
                {
                    int index = row / 3;
                    blueChannelData[index] = (byte) eigenVector[row];
                    greenChannelData[index] = (byte) eigenVector[row + 1];
                    redChannelData[index] = (byte) eigenVector[row + 2];
                }

                _pcaFilters.Add(new SmallImageData(label, className, ImageWidth, ImageHeight, redChannelData, greenChannelData, blueChannelData));
            }
        }

        protected abstract bool CheckDirectory(string selectedPath, List<string> fileNames);
        protected abstract bool ReadFiles(string selectedPath);
    }
}
